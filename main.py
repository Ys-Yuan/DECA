import sys
import os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing import Process, Queue, Event, Manager
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, HfArgumentParser
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from src.utils.argument import ModelArguments, DataArguments, TrainingArguments, DSArguments, LoRAArguments
from src.utils.topology import Graph, closest_factors
from src.data_loader.manager import *
from src.agent.deca_agent import DECAAgent
import pickle
import transformers
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@dataclass 
class AgentConfig:
    model_args: ModelArguments
    lora_args: LoRAArguments
    ds_args: DSArguments
    data_args: DataArguments
    training_args: TrainingArguments
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

AGENT_GROUPS = {
    DECAAgent: ["deca","deca_avg","deca_tsm"]
}

AGENT_MAP = {
    key: agent
    for agent, keys in AGENT_GROUPS.items()
    for key in keys
}

def train(agent_config, tokenizer_r, tokenizer_l):

    tokenizer_r.padding_side = "right"
    rank = int(os.environ["LOCAL_RANK"])
    data_manager = DataManager(data_config=agent_config.data_args, model_config=agent_config.model_args, tokenizer=tokenizer_r)
    data_modules = data_manager._get_data_loader(num_clients=agent_config.ds_args.num_clients, partition_method=agent_config.data_args.partition_method)
    dist.init_process_group(backend="nccl", rank=rank, world_size=agent_config.ds_args.num_clients)
    graph_topology = Graph(agent_config.ds_args.topology, agent_config.ds_args.num_clients)
    tokenizer_l.padding_side = "left"
    data_manager = DataManager(data_config=agent_config.data_args, model_config=agent_config.model_args, tokenizer=tokenizer_l)
    glb_data_modules = data_manager._get_data_loader(num_clients=1, partition_method="uniform")

    method = agent_config.model_args.method_type
    agent = AGENT_MAP[method](
        config=agent_config, 
        rank=rank, 
        world_size=agent_config.ds_args.num_clients,
        graph=graph_topology, 
        train_dataset=data_modules[rank]["train"], 
        eval_dataset=data_modules[rank]["test"],
        glb_eval_dataset=glb_data_modules[0]["test"],
        tokenizer=tokenizer_r
    )
    try:
        agent.run_training(num_epochs=agent_config.ds_args.num_rounds)
    except Exception as e:
        agent.logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':

    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, LoRAArguments, DataArguments, TrainingArguments, DSArguments))
    model_args, lora_args, data_args, training_args, ds_args = parser.parse_args_into_dataclasses()
    agent_config = AgentConfig(model_args, lora_args, ds_args, data_args, training_args)
    
    set_seed(agent_config.ds_args.seed)
    tokenizer_r = AutoTokenizer.from_pretrained(agent_config.model_args.base_model_path, trust_remote_code=True)
    tokenizer_l = AutoTokenizer.from_pretrained(agent_config.model_args.base_model_path, trust_remote_code=True)
    if tokenizer_r.pad_token is None:
        tokenizer_r.pad_token = tokenizer_r.eos_token
    if tokenizer_l.pad_token is None:
        tokenizer_l.pad_token = tokenizer_l.eos_token

    train(agent_config, tokenizer_r, tokenizer_l)
    
