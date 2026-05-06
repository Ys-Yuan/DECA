import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import time
import torch
import psutil
import pynvml
import Levenshtein
import json
import random
import pandas as pd
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from sklearn.metrics import classification_report
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import torch.distributed as dist


class Timer:
    def __init__(self):
        self.records = {}

    def __call__(self, name):
        return self._TimerContext(self, name)

    class _TimerContext:
        def __init__(self, outer, name):
            self.outer = outer
            self.name = name

        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.time()

        def __exit__(self, exc_type, exc_value, traceback):
            torch.cuda.synchronize()
            duration = time.time() - self.start
            self.outer.records[self.name] = duration


class DirectoryManager:
    def __init__(self, base, *parts, filename=None):
        self.base = base
        if filename is not None:
            # 文件路径
            self.root = os.path.join(base, *parts)
            self.filename = filename
        else:
            # 目录路径
            self.root = os.path.join(base, *parts)
            self.filename = None
        os.makedirs(self.root, exist_ok=True)

    def sub(self, *paths):
        path = os.path.join(self.root, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def dir(self, *paths):
        d = os.path.join(self.root, *paths)
        os.makedirs(d, exist_ok=True)
        return d
    
    def write_text(self, *paths, content):
        path = self.sub(*paths)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def write_json(self, *paths, obj):
        path = self.sub(*paths)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def write_bytes(self, *paths, data):
        path = self.sub(*paths)
        with open(path, "wb") as f:
            f.write(data)


class CheckpointManager:
    def __init__(self, config, rank=-1):
        self.config = config
        self.rank = rank
        self.lora_rank = config.lora_args.lora_rank
        dataset = config.data_args.dataset_name
        method = config.model_args.method_type
        model = config.model_args.model_type
        self.alpha = f"{config.data_args.dirichlet_alpha}".replace('.', '')
        # Base output directories
        self.ckpt = DirectoryManager("./ckpt", f"{dataset}_{method}_{model}")
        self.output = DirectoryManager("./output/results", dataset, method, model)
        self.csv_path = DirectoryManager("./output/results")
        self.loss_path = DirectoryManager("./output/results")

    def _model(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, epoch: int, rank: int = -1):
        suffix = f"_c{rank}" if rank != -1 else ""
        if isinstance(model, PeftModel):
            save_dir = self.ckpt.dir(f"r{self.lora_rank}_m{epoch}_a{self.alpha}{suffix}")
            model.save_pretrained(save_dir)
            if tokenizer:
                tokenizer.save_pretrained(save_dir)
        elif hasattr(model, "adapters_config"):
            adapter_name = getattr(self.config.model_args, "adapter_name", "task_adapter")
            save_dir = self.ckpt.dir(f"adapter_m{epoch}_a{self.alpha}{suffix}")
            model.save_adapter(save_dir, adapter_name)
        elif hasattr(model, "prompt"):
            torch.save(model.prompt.detach().cpu(), self.ckpt.sub(f"m{epoch}_a{self.alpha}{suffix}.pt"))
        else:
            torch.save(model.state_dict(), self.ckpt.sub(f"m{epoch}_a{self.alpha}{suffix}.pt"))

    def _generation(self, reference_texts, generated_texts, glb=True, rank=-1):
        if rank == -1:
            paired = [{'reference': r, 'generated': g} for r, g in zip(reference_texts, generated_texts)]
            self.output.write_json(f"r_{self.alpha}.json" if glb or self.rank==-1 else f"r{self.rank}_{self.alpha}.json", obj=paired)
            report = classification_report(reference_texts, generated_texts, digits=4)
            self.output.write_text(f"q_{self.alpha}.txt" if glb or self.rank==-1 else f"q{self.rank}_{self.alpha}.txt", content=report)
        else:
            paired = [{'reference': r, 'generated': g} for r, g in zip(reference_texts, generated_texts)]
            self.output.write_json(f"r_{self.alpha}_c{rank}.json" if glb or self.rank==-1 else f"r{self.rank}_{self.alpha}_c{rank}.json", obj=paired)
            report = classification_report(reference_texts, generated_texts, digits=4)
            self.output.write_text(f"q_{self.alpha}_c{rank}.txt" if glb or self.rank==-1 else f"q{self.rank}_{self.alpha}_c{rank}.txt", content=report)

    def _metrics(self, result_dict, epoch, rank=-1):
        if rank == -1:
            csv_path = self.csv_path.sub(f"{self.config.model_args.model_type}.csv")
        else:
            csv_path = self.csv_path.sub(f"{self.config.model_args.model_type}_c{rank}.csv")
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=result_dict.keys())
        row_idf = (
            (df["dataset_name"] == result_dict["dataset_name"]) &
            (df["method"] == result_dict["method"]) &
            (df["alpha"] == result_dict["alpha"]) &
            (df["seed"] == result_dict["seed"]) & 
            (df["topo"] == result_dict["topo"]) &
            (df["epoch"] == result_dict["epoch"])
        )
        if not df[row_idf].empty:
            if result_dict["F1"] > df.loc[row_idf, "F1"].values[0]:
                df.loc[row_idf, ["Accuracy", "Precision", "Recall", "F1"]] = [
                    result_dict["Accuracy"], result_dict["Precision"], result_dict["Recall"], result_dict["F1"]
                ]
        else:
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)   
        df.to_csv(csv_path, index=False)

    def _metrics_gen(self, result_dict):
        csv_path = self.csv_path.sub(f"{self.config.model_args.model_type}_gen.csv")
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=result_dict.keys())
        row_idf = (
            (df["dataset_name"] == result_dict["dataset_name"]) &
            (df["method"] == result_dict["method"]) &
            (df["alpha"] == result_dict["alpha"])
        )
        if not df[row_idf].empty:
            if result_dict["METEOR"] > df.loc[row_idf, "METEOR"].values[0]:
                df.loc[row_idf, ["BLEU","ROUGEL","METEOR","CIDEr"]] = [
                    result_dict["BLEU"], result_dict["ROUGEL"], result_dict["METEOR"], result_dict["CIDEr"]
                ]
        else:
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)   
        df.to_csv(csv_path, index=False)

    def _loss(self, global_loss):
        loss_path = self.loss_path.sub("loss.csv")
        df = pd.read_csv(loss_path) if os.path.exists(loss_path) else pd.DataFrame(columns=["dataset_name", "model", "method", "alpha", "seed", "topo", "loss"])
        loss_str = ",".join([str(round(l,3)) for l in global_loss])
        row_idf = (
            (df["dataset_name"] == self.config.data_args.dataset_name) &
            (df["model"] == self.config.model_args.model_type) &
            (df["method"] == self.config.model_args.method_type) &
            (df["alpha"] == self.config.data_args.dirichlet_alpha) &
            (df["seed"] == self.config.ds_args.seed) &
            (df["topo"] == self.config.ds_args.topology) 
        )
        if not df[row_idf].empty:
            df.loc[row_idf, "loss"] = loss_str
        else:
            new_row = {
                "dataset_name": self.config.data_args.dataset_name,
                "model": self.config.model_args.model_type,
                "method": self.config.model_args.method_type,
                "alpha": self.config.data_args.dirichlet_alpha,
                "seed": self.config.ds_args.seed,
                "topo": self.config.ds_args.topology,
                "loss": loss_str
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(loss_path, index=False)



class SystemMonitor:

    def __init__(self, device: torch.device):

        self.device = device
        self.process = psutil.Process(os.getpid())
        # GPU内存监控相关
        self.gpu_memory_stats: Dict[str, Any] = {
            'peak_allocated': 0,  # 峰值已分配内存
            'peak_reserved': 0,   # 峰值已保留内存
            'allocation_history': [],  # 内存分配历史记录
            'peak_contexts': []  # 记录峰值出现的上下文
        }
        # 初始化 NVIDIA 管理接口
        self.gpu_handle = None
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                pynvml.nvmlInit()
                gpu_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except pynvml.NVMLError as e:
                print(f"Warning: Could not initialize pynvml. GPU monitoring disabled. Error: {e}")
                self.gpu_handle = None
        
    def _format_memory(self, bytes_value: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024 or unit == 'GB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.2f} GB"

    # --- 内存和利用率获取 ---

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """ 获取当前 GPU 内存使用情况 """
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            return {
                'gpu_memory_allocated_gb': allocated,
                'gpu_memory_reserved_gb': reserved,
                'gpu_memory_max_allocated_gb': max_allocated
            }
        return {}
        
    def get_cpu_memory_usage(self) -> Dict[str, float]:
        """ 获取当前进程的 CPU 内存使用情况 """
        mem_info = self.process.memory_info()
        return {
            'cpu_memory_rss_gb': mem_info.rss / (1024**3),
            'cpu_memory_vms_gb': mem_info.vms / (1024**3)
        }
    
    def get_cpu_gpu_utilization(self) -> Dict[str, float]:
        """ 获取 CPU 和 GPU 的利用率百分比。"""
        cpu_percent = self.process.cpu_percent(interval=None) 
        metrics = {'cpu_utilization_percent': cpu_percent}
        if self.gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics['gpu_utilization_percent'] = util.gpu
            except pynvml.NVMLError as e:
                print(f"Warning: Failed to get GPU utilization. Error: {e}")
        return metrics

    # --- 内存峰值跟踪 ---

    def get_gpu_memory_stats(self) -> Dict[str, int]:
        """ 获取 PyTorch CUDA 内存的实时统计数据 """
        if self.device.type != 'cuda':
             return {'allocated': 0, 'reserved': 0, 'max_allocated': 0, 'max_reserved': 0}
        torch.cuda.set_device(self.device)
        return {
            'allocated': torch.cuda.memory_allocated(self.device),
            'reserved': torch.cuda.memory_reserved(self.device),
            'max_allocated': torch.cuda.max_memory_allocated(self.device),
            'max_reserved': torch.cuda.max_memory_reserved(self.device)
        }

    def record_gpu_memory(self, context: Optional[str] = None) -> Dict[str, int]:
        """ 记录当前的 GPU 内存统计数据，并更新峰值信息 """
        if self.device.type != 'cuda':
            return {}
        stats = self.get_gpu_memory_stats()
        timestamp = time.time()
        # 更新峰值分配内存
        if stats['max_allocated'] > self.gpu_memory_stats['peak_allocated']:
            self.gpu_memory_stats['peak_allocated'] = stats['max_allocated']
            if context:
                self.gpu_memory_stats['peak_contexts'].append({
                    'type': 'allocated_max',
                    'value': stats['max_allocated'],
                    'context': context,
                    'timestamp': timestamp
                })
        # 更新峰值保留内存
        if stats['max_reserved'] > self.gpu_memory_stats['peak_reserved']:
            self.gpu_memory_stats['peak_reserved'] = stats['max_reserved']
            if context:
                self.gpu_memory_stats['peak_contexts'].append({
                    'type': 'reserved_max',
                    'value': stats['max_reserved'],
                    'context': context,
                    'timestamp': timestamp
                })
        # 记录历史分配
        self.gpu_memory_stats['allocation_history'].append({
            'timestamp': timestamp,
            'allocated': stats['allocated'],
            'reserved': stats['reserved'],
            'context': context
        })
        return stats

    def reset_gpu_memory_tracking(self):
        """重置 PyTorch CUDA 内存的峰值统计和 Monitor 内部的峰值记录 """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.gpu_memory_stats['peak_allocated'] = 0
        self.gpu_memory_stats['peak_reserved'] = 0
        self.gpu_memory_stats['peak_contexts'] = []

    def save_gpu_memory_stats(self, filepath: str):
        """将 GPU 内存统计数据保存到 JSON 文件中"""
        serializable_stats = {
            'peak_allocated': self._format_memory(self.gpu_memory_stats['peak_allocated']),
            'peak_reserved': self._format_memory(self.gpu_memory_stats['peak_reserved']),
            'peak_contexts': self.gpu_memory_stats['peak_contexts'],
            'recent_allocation_history': self.gpu_memory_stats['allocation_history'][-1000:]
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_stats, f, indent=2, default=str)


class LossPlotter:
    def __init__(self, rank: int, world_size: int, config: Any):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        dataset_name = config.data_args.dataset_name
        method = config.model_args.method_type
        model = config.model_args.model_type
        self.colors = [
            "#C00000", "#004C99", "#E6A100", "#2F8F2F", "#6B2FB3"
        ]
        self.markers = ["o", "s", "D", "P", "X"]
        self.plot_dir = DirectoryManager("./output/plot/", dataset_name, method, model)

    def _set_style(self, ax):
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color("#333333")
        ax.tick_params(axis='both', which='major', labelsize=36)
        ax.grid(
            axis='both',
            linewidth=1.2,
            color="#CCCCCC",
            linestyle="--",
            zorder=0
        )

    def _plot(self, losses):

        length_tensor = torch.tensor([len(losses)], dtype=torch.long, device=self.rank)
        gathered_len = [torch.zeros(1, dtype=torch.long, device=self.rank) for _ in range(self.world_size)]
        dist.all_gather(gathered_len, length_tensor)
        expected_len = gathered_len[0].item()
        loss_tensor = torch.tensor(losses, dtype=torch.float32, device=self.rank)
        gathered_losses = [
            torch.empty(expected_len, dtype=torch.float32, device=self.rank)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_losses, loss_tensor)

        if self.rank == 0:
            plt.figure(figsize=(16, 9))
            ax = plt.gca()
            self._set_style(ax)
            for r in range(self.world_size):
                curve = gathered_losses[r].cpu().tolist()
                ax.plot(
                    range(1, expected_len + 1),
                    curve,
                    label=f"Rank {r}",
                    linewidth=3.5,
                    marker=self.markers[r % len(self.markers)],
                    markersize=9,
                    markevery=max(1, expected_len // 10),
                    color=self.colors[r % len(self.colors)],
                )
            ax.set_xlabel("Global Update Step", fontsize=40)
            ax.set_ylabel("Loss", fontsize=40)
            ax.legend(
                loc='upper right',
                bbox_to_anchor=(1.01, 1.02),
                prop={'size': 32},
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="#333333",
                borderpad=0.3,
                ncol=2,
                handletextpad=0.2,
                columnspacing=0.4,
                labelspacing=0.2,
                handlelength=1.2, 
                markerscale=1.2,
            )
            plt.tight_layout()
            plt.savefig(self.plot_dir.sub("loss.png"), dpi=600)
            plt.close()
        dist.barrier()


class ResponseHandler:
    def __init__(self, labels):
        self.labels = labels

    def closest_label(self, response):
        return min(self.labels, key=lambda l: Levenshtein.distance(response, l))

    def handle_response(self, response):
        for label in self.labels:
            if label in response:
                return label
        return self.labels[-1]

    def handle_batch_responses(self, responses, window_size=20):
        results = []
        for resp in responses:
            window = resp[:window_size]
            # 优先精确匹配开头 window
            exact = next((label for label in self.labels if label in window), None)
            if exact:
                results.append(exact)
                continue
            # 使用最接近标签
            results.append(self.closest_label(resp))
        return results

def print_rank_0(s, force=True):
    if not torch.distributed.is_initialized():
        print(s)
    elif torch.distributed.get_rank() == 0 and force:
        print(s)

def _group_random_max_k(blocks, k):
    groups = []
    idx = 0
    n = len(blocks)
    group_id = 0
    while idx < n:
        max_size = min(k, n - idx)
        group_size = random.randint(1, max_size)
        group_blocks = blocks[idx : idx + group_size]
        groups.append({
            "group_id": group_id,
            "blocks": group_blocks,
            "start": idx,
            "end": idx + group_size - 1,
        })
        idx += group_size
        group_id += 1
    return groups

def _group_fixed_k(blocks, k):
    groups = []
    n = len(blocks)
    group_id = 0
    for start in range(0, n, k):
        end = min(start + k, n)
        group_blocks = blocks[start:end]
        groups.append({
            "group_id": group_id,
            "blocks": group_blocks,
            "start": start,
            "end": end - 1,
        })
        group_id += 1
    return groups
