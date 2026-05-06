import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import copy
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from utils.util import *
from utils.topology import Graph
from src.utils.metrics import TextQualityEvaluator
from utils.argument import ModelArguments, DataArguments, TrainingArguments, DSArguments, LoRAArguments
from agent.agent_util import Timer, DirectoryManager, CheckpointManager, SystemMonitor, LossPlotter, ResponseHandler, _group_random_max_k, _group_fixed_k
from optimizer.deca_optimizer import BlockAdamW, MeZOBlockAdamW
import wandb
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DecentralizedConfig:
    model_args: ModelArguments
    data_args: DataArguments
    ds_args: DSArguments
    training_args: TrainingArguments
    node_id: int
    neighbor_ids: list
    # 混合精度配置
    storage_dtype: torch.dtype = torch.bfloat16  # 存储精度 bf16/fp16
    compute_dtype: torch.dtype = torch.float32   # 计算精度 fp32


class BaseAgent:

    def __init__(
        self, 
        config: DecentralizedConfig,
        rank: int,
        world_size: int,
        graph: Graph,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        glb_eval_dataset: Dataset = None
    ):
        # 参数配置
        self.config = config
        self.rank = rank
        self.graph = graph
        self.size = world_size
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.glb_eval_dataset = glb_eval_dataset
        self.tokenizer = tokenizer
        
        # 混合精度配置
        self.storage_dtype = getattr(config, 'storage_dtype', torch.bfloat16)  # 存储精度 bf16/fp16
        self.compute_dtype = getattr(config, 'compute_dtype', torch.float32)   # 计算精度 fp32
        
        # 数据管理
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training_args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True
        ) if self.train_dataset is not None else None
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config.training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False
        ) if self.eval_dataset is not None else None
        
        self._init_topo()

    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p
            
    def _setup_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


    def _init_model(self, from_ckpt=False):
        # 模型初始化
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_args.base_model_path,
            torch_dtype=self.storage_dtype,
            use_flash_attention_2=False,
            trust_remote_code=True
        )
        state_dict_path = getattr(self.config.model_args, "ckpt_path", None)
        if not from_ckpt or not os.path.exists(state_dict_path):
            # lora tuning
            if "lora" in self.config.model_args.method_type or "caf" in self.config.model_args.method_type:
                lora_args = self.config.lora_args
                lora_config = LoraConfig(
                    r=getattr(lora_args, 'lora_rank', 16),
                    lora_alpha=getattr(lora_args, 'lora_alpha', 32),
                    target_modules=getattr(lora_args, 'target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
                    lora_dropout=getattr(lora_args, 'dropout', 0.1),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.model = get_peft_model(self.model, lora_config)
                for name, param in self.model.named_parameters():
                    if 'lora_' in name: 
                        param.data = param.data.to(self.compute_dtype)
            self.named_parameters_list = list(self.model.named_parameters()) 
        else:
            config = AutoConfig.from_pretrained(self.config.model_args.base_model_path, trust_remote_code=True)
            if "lora" in self.config.model_args.method_type:
                self.model = PeftModel.from_pretrained(self.model, state_dict_path)
                self.model.merge_and_unload()
            else:
                self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                state_dict = torch.load(state_dict_path, map_location="cpu")
                self.model.load_state_dict(state_dict)


    def _init_topo(self):
        # 拓扑管理
        self.groups, self.dist_groups, self.p2p_groups = self.graph.get_comm_group()
        self.neighbor_rank, self.neighbor_weight, self.neighbor, self.dsm = self.graph.get_neighbor_info(self.rank)
        self.neighbor_rank_ns = [r for r in self.neighbor_rank if r != self.rank]
        

    def _init_trainer(self, config):
        # 轮次配置
        self.current_epoch = 0
        self.total_epochs = getattr(config.ds_args, 'num_rounds', 128)
        self.local_epochs = getattr(config.ds_args, 'local_epochs', 4)
        # optimizer配置
        self.lr = config.training_args.learning_rate
        self.weight_decay = config.training_args.weight_decay
        self.block_target_modules = config.lora_args.target_modules
        self.use_tsm = True if "tsm" in self.config.model_args.method_type else False
        self.use_mm = True if "moment" in self.config.model_args.method_type else False
        if config.training_args.optim == "mezo":
            self.optimizer = MeZOBlockAdamW(
                self.model.parameters(),
                self.model,
                lr=config.training_args.learning_rate,
                weight_decay=config.training_args.weight_decay,
                block_switch_frequency=getattr(config.model_args, 'blkcg_frq', 16),
                block_strategy=getattr(config.training_args, 'block_strategy', 'fixed_size'),
                block_size=getattr(config.training_args, 'block_size', 1),
                block_sequence=getattr(config.model_args, 'blk_seq', 'ascending'),
                target_modules=self.block_target_modules,
            )
        elif config.training_args.optim == "blockwise":
            self.optimizer = BlockAdamW(
                self.model.parameters(),
                self.model,
                lr=config.training_args.learning_rate,
                weight_decay=config.training_args.weight_decay,
                block_switch_frequency=getattr(config.model_args, 'blkcg_frq', 16),
                block_strategy=getattr(config.training_args, 'block_strategy', 'fixed_size'),
                block_size=getattr(config.training_args, 'block_size', 1),
                block_sequence=getattr(config.model_args, 'blk_seq', 'ascending'),
                target_modules=self.block_target_modules,
                use_tsm=self.use_tsm,
                use_mm=self.use_mm,
                mu=(config.training_args.mu1,config.training_args.mu2)
            )
        elif config.training_args.optim == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.training_args.learning_rate,
                weight_decay=config.training_args.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.training_args.learning_rate,
                weight_decay=config.training_args.weight_decay
            )
        self.scaler = torch.cuda.amp.GradScaler() if self.storage_dtype == torch.float16 else None
        self.training_steps = 1.5 * self.total_epochs * self.config.ds_args.local_epochs
        self.lr_scheduler = transformers.get_scheduler(
            name=self.config.training_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=getattr(self.config.training_args, 'warmup_ratio', 0.0) * self.training_steps,
            num_training_steps=self.training_steps,
        )
        accelerator = Accelerator()
        self.device = accelerator.device
        self.model.to(device=self.device)
    
    def _init_util(self, config):
        # self._initialize_wandb()
        self.logger = self._setup_logger(f"node_{self.rank+1}")
        self.timer = Timer() 
        # self.monitor = SystemMonitor(self.device) # 使用 SystemMonitor 内存监控
        self.plotter = LossPlotter(self.rank, self.size, config) 
        self.ckpt_manager = CheckpointManager(config, self.rank)   # 使用 CheckpointManager 保存

    def update_gradients(self):
        for name, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        if self.gm:
            for p, p_prev, buf in zip(self.model.parameters(), self.prev_params, self.momentum_buff):
                buf.mul_(self.momentum).add_(p_prev.data-p.data, alpha=(1.0-self.momentum)/self.lr) #m_hat
                mom_buff = copy.deepcopy(buf)
                mom_buff.mul_(self.momentum).add_(p.grad.data) # m
                if self.nesterov:
                    p.grad.data.add_(mom_buff, alpha=self.momentum) # nestrove momentum
                else:
                    p.grad.data.copy_(mom_buff) 
            for p, p_prev in zip(self.model.parameters(), self.prev_params):
                p_prev.data.copy_(p.data)
        else:
            for p, buf in zip(self.model.parameters(), self.momentum_buff):
                buf.mul_(self.momentum).add_(p.grad.data)
                if self.nesterov:
                    p.grad.data.add_(buf, alpha=self.momentum) # nestrove momentum
                else:
                    p.grad.data.copy_(buf)

    def _exchange_params(self):
        received_params = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}      
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            param_cpu = param.data
            param_buf = torch.zeros_like(param_cpu)
            for src_rank in range(self.size):
                if self.rank in self.groups[src_rank]:
                    if src_rank == self.rank:
                        param_buf.copy_(param_cpu)
                    else:
                        param_buf.zero_()
                    dist.broadcast(tensor=param_buf, src=src_rank, group=self.dist_groups[src_rank])
                    if src_rank in self.neighbor_rank_ns:
                        received_params[src_rank].append(param_buf.clone().to(self.device))
        return received_params
    
    def _aggregate_params(self, received_params):
        param_idx = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            param.data.mul_(self.neighbor[self.rank])
            for src_id in self.neighbor_rank_ns:
                if param_idx < len(received_params[src_id]):
                    param.data.add_(received_params[src_id][param_idx], alpha=self.neighbor[src_id])
            param_idx += 1

    def _aggregate_var(self, var, keys=None):
        def _avg(tensor):
            t = tensor.detach().float().flatten().to(self.device)
            g = [torch.zeros_like(t).to(self.device) for _ in range(self.size)]
            dist.all_gather(g, t)
            return torch.stack(g).mean(0).view(t.shape).to(t.device, t.dtype)
        def _avg_list(list):
            t = torch.tensor(list, dtype=self.compute_dtype).flatten().to(self.device)
            g = [torch.zeros_like(t).to(self.device) for _ in range(self.size)]
            dist.all_gather(g, t)
            avg_tensor = torch.stack(g).mean(0)
            return avg_tensor.tolist()
        if isinstance(var, (torch.Tensor, nn.Parameter)):
            return _avg(var)
        elif isinstance(var, dict):
            return {k: _avg(v) if (keys is None or k in keys) else v for k, v in var.items()}
        elif isinstance(var, nn.Module):
            averaged_state = {name: _avg(param) for name, param in var.state_dict().items()}
            var.load_state_dict(averaged_state)
            return var
        elif isinstance(var, list):
            return _avg_list(var)
        else:
            raise TypeError(f"Unsupported type for distributed_average: {type(var)}")

    def _aggregate_var_cpu(self, var, keys=None):
        def _avg(tensor):
            t = tensor.detach().float().cpu().flatten()
            g = [torch.zeros_like(t) for _ in range(self.size)]
            dist.all_gather(g, t)
            return torch.stack(g).mean(0).view(t.shape).to(t.device, t.dtype)
        def _avg_list(list):
            t = torch.tensor(list, dtype=self.compute_dtype, device='cpu').flatten()
            g = [torch.zeros_like(t) for _ in range(self.size)]
            dist.all_gather(g, t)
            avg_tensor = torch.stack(g).mean(0)
            return avg_tensor.tolist()
        if isinstance(var, (torch.Tensor, nn.Parameter)):
            return _avg(var)
        elif isinstance(var, dict):
            return {k: _avg(v) if (keys is None or k in keys) else v for k, v in var.items()}
        elif isinstance(var, nn.Module):
            averaged_state = {name: _avg(param) for name, param in var.state_dict().items()}
            var.load_state_dict(averaged_state)
            return var
        elif isinstance(var, list):
            return _avg_list(var)
        else:
            raise TypeError(f"Unsupported type for distributed_average: {type(var)}")

    def _initialize_wandb(self):
        self.wandb = wandb.init(
            project=f"decentralized-{self.config.model_args.method_type}-{self.config.model_args.model_type}",
            name=f"C{self.size}_{self.config.data_args.dataset_name}_{self.rank}",
            config={
                "rank": self.rank,
                "world_size": self.size,
                "alpha": self.config.data_args.dirichlet_alpha,
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.config.training_args.per_device_train_batch_size,
                "total_epochs": self.total_epochs,
                "monitor_gpus": True
            },
            group=f"{self.config.data_args.dataset_name}_{self.size}"
        )

    def _log_step_metrics(self, tmp_loss):
        step_metrics = {
            'batch_loss': tmp_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'compute_time_sec': self.timer.records.get("compute", 0.0),
            'comm_time_sec': self.timer.records.get("comm", 0.0)
        }
        step_metrics.update(self.monitor.get_gpu_memory_usage())
        step_metrics.update(self.monitor.get_cpu_memory_usage())
        self.wandb.log(step_metrics)
        # 额外记录时间统计
        self.wandb.log({
            "compute_time": self.timer.records.get("compute", 0.0),
            "comm_time": self.timer.records.get("comm", 0.0),
            "total_time": self.timer.records.get("compute", 0.0) + self.timer.records.get("comm", 0.0),
        })