import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import functools
import os

@dataclass
class ModelArguments:
    method_type: str = field(
        default="bqgm",
        metadata={"help": "bqgm/dlora/fedprompt"}
    )
    model_type: str = field(
        default="llama3",
        metadata={"help": "Type of the pretrained model: 'llama3' or 'qwen2.5'."}
    )
    base_model_path: str = field(
        default="/home/yuanyunsheng/hf-models/Llama-3.2-3B",
        metadata={"help": "Path of the pretrained model: /home/yuanyunsheng/hf-models/Qwen2.5-3B-Instruct, Llama-3.2-3B"}
    )
    ckpt_path: str = field(
        default="/home/yuanyunsheng/bqgm/checkpoints/mnli/qwen2.5_global_model_20.pt",
        metadata={"help": "Path of the finetuned oracle model."}
    )
    load_from_ckpt: bool = field(
        default=False, 
        metadata={"help": "false for train and true for eval."}
    )
    load_from_lora: bool = field(
        default=True, 
        metadata={"help": "Whether to load from LoRA weights or merged full model."}
    )
    layer_module: str = field(
        default="MetaLlamaDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    generation_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    blkcg_frq:int = field(
        default=32,
        metadata={"help": "Frequency of block changing"}
    )
    blk_seq: str = field(
        default="descending",
        metadata={"help": "random, descending, ascending, fixed"}
    )

@dataclass
class LoRAArguments:
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA target modules."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "LoRA bias."}
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "LoRA task type."}
    )
    data_type: str = field(
        default="bfloat16",
        metadata={"help": "Floating point precision (bfloat16 or float16)."}    
    )
    modified_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to use (0 for full model)."}
    )

@dataclass
class DSArguments:
    num_clients: int = field(
        default=4,
        metadata={"help": "Number of clients."}
    )
    topology: str = field(
        default="Ring",
        metadata={"help": "Graph topology: 'Ring', 'Bipar', 'FC', 'Grid', 'Random'."}
    )
    avg: bool = field(
        default=False,
        metadata={"help": "Whether to use average consensus or not."}
    )
    reg: float = field(
        default=1e-3,
        metadata={"help": "Regularization parameter for doubly stochastic matrix computation."}
    )
    num_rounds: int = field(
        default=100,
        metadata={"help": "Number of federated rounds."}
    )
    local_epochs: int = field(
        default=2,
        metadata={"help": "Number of local epochs per round."}
    )
    local_batch_size: int = field(
        default=1,
        metadata={"help": "Local batch size per client."}
    )
    sample_rate: float = field(
        default=0.8,
        metadata={"help": "Sample rate for clients in each round."}
    )
    master_addr: str = field(
        default="localhost",
        metadata={"help": "Address of the master node."}    
    )
    master_port: int = field(
        default=12345,
        metadata={"help": "Port of the master node."}
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum factor for SGD."}
    )
    mu: float = field(
        default=0.6,
        metadata={"help": "Mixing parameter for QGM."}
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Whether to use Nesterov momentum."}
    )
    gm: bool = field(
        default=False,
        metadata={"help": "Whether to use GM."}
    )
    device_index: int = field(
        default=0,
        metadata={"help": "GPU device number to use."}
    )
    comm_frq: int = field(
        default=1,
        metadata={"help": "communication frequence"}
    )
    eval_frq: int = field(
        default=50,
        metadata={"help": "evaluate during training"}
    )
    prt_frq: int = field(
        default=4,
        metadata={"help": "print and record frequence"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="nwgi",
        metadata={"help": "Name of the dataset."}
    )
    data_path: str = field(
        default="/home/yuanyunsheng/bqgm/data/nwgi",
        metadata={"help": "Directory containing the dataset."}  
    )
    data_split: str = field(
        default="all",
        metadata={"help": "Split of the dataset to use."}
    )
    test_rate: float = field(
        default=0.2,
        metadata={"help": "Proportion of the dataset to use for testing."}
    )
    div_test_data: bool = field(
        default=False,
        metadata={"help": "Whether to divide test data among clients."}
    )
    div_test_data_rate: float = field(
        default=1.0,
        metadata={"help": "Proportion of test data to divide among clients if div_test_data"}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    max_len: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    partition_method: str = field(
        default="noniid",
        metadata={"help": "Method to partition data among clients: 'uniform' or 'dirichlet'."}
    )
    sample_method: str = field(
        default="random",
        metadata={"help": "Method to sample data for each client: 'random' or 'sequential'."}   
    )
    dirichlet_alpha: float = field(
        default=2.0,
        metadata={"help": "Alpha parameter for Dirichlet distribution (used with 'dirichlet' partition_method)."}
    )

@dataclass
class TrainingArguments:

    # --- training ---
    classify: bool = field(
        default=False,
        metadata={"help": "Classify dataset."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the pre-trained models downloaded from huggingface.co."}
    )
    optim: str = field(
        default="blockwise",
        metadata={"help": "Optimizer to use."}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bf16:  bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision (Ampere+ GPUs)."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size per GPU."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size per GPU."}
    )
    eval_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "Evaluation strategy."}
    )
    eval_steps: Optional[int] = field(
        default=8,
        metadata={"help": "Number of update steps between two evaluations."}
    )
    eval_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={"help": "Number of steps to accumulate before performing evaluation."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={"help": "Effective batch size multiplier."}
    )
    logging_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "Logging strategy."}
    )
    logging_steps: Optional[int] = field(
        default=20,
        metadata={"help": "Number of update steps between two logs."}
    )
    eval_logging_steps: Optional[int] = field(
        default=50,
        metadata={"help": "Number of eval steps between two logs."}
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Total training epochs."}
    )
    max_steps: Optional[int] = field(
        default=200,
        metadata={"help": "Max steps."}
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Learning rate."}
    )
    warmup_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "LR warmup proportion."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "Learning rate schedule."}
    )
    weight_decay: Optional[float] = field(
        default=0.02,
        metadata={"help": "L2 regularization strength."}
    )
    save_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Checkpoint save interval."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Max checkpoints to keep."}
    )
    mu1: Optional[float] = field(
        default=0.9,
        metadata={"help": "mu1."}
    )
    mu2: Optional[float] = field(
        default=0.999,
        metadata={"help": "mu2"}
    )