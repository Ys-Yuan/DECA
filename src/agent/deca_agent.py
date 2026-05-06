import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import copy
from typing import Dict, Optional, Sequence, List
import torch
import torch.distributed as dist
from itertools import islice, cycle
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.util import *
from utils.topology import Graph
from src.agent.base_agent import BaseAgent
from src.optimizer.deca_optimizer import BlockAdamW
from agent.evaluator import Evaluator
import warnings
warnings.filterwarnings("ignore")


class DECAAgent(BaseAgent):

    def __init__(
        self, 
        config,
        rank: int,
        world_size: int,
        graph: Graph,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        glb_eval_dataset: Dataset = None
    ):
        super().__init__(
            config, rank, world_size, graph, tokenizer, train_dataset, eval_dataset, glb_eval_dataset
        )
        self._init_model(from_ckpt=False)
        self._init_trainer(self.config)
        self._init_util(self.config)
        self.logger.info(f"Data volume: {len(self.train_dataset)}")
        self.ex_mb_history: List[float] = []

    def run_training(self, num_epochs=10):
        self.model.train()
        self.glb_loss = []
        for epoch in range(num_epochs):
            self.model.train()
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            sampler = RandomSampler(self.train_dataset)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training_args.per_device_train_batch_size,
                sampler=sampler,
                drop_last=True,
            )
            self._train_epoch()
            self.current_epoch = epoch + 1
            if self.config.data_args.dataset_name == "alpaca" and self.current_epoch in [244, 248, 252]:
                self.ckpt_manager._model(self.model, self.tokenizer, epoch+1, self.rank)
        if self.config.data_args.dataset_name == "alpaca":
            self.ckpt_manager._model(self.model, self.tokenizer, epoch+1, self.rank)
        self.plotter._plot(self.glb_loss)
        avg_glb_loss = self._aggregate_var(self.glb_loss)
        if self.rank == 0:
            self.ckpt_manager._loss(avg_glb_loss)
        if len(self.ex_mb_history) > 0:
            self.ex_mb = sum(self.ex_mb_history) / len(self.ex_mb_history)
            # self.logger.info(f"Average exchange sent payload bytes: {self.ex_mb:.2f} MB")
            log_str = f"Average exchange sent payload bytes: {self.ex_mb:.2f} MB"
            with open("./output/exchange_log.txt", "a") as f:
                if self.rank == 0:
                    f.write(f"\n-------- {self.config.model_args.method_type} --------\n")
                dist.barrier()
                f.write(log_str + "\n")
        self.optimizer.state.clear()
        self.evaluator = Evaluator(
            self.config,
            self.tokenizer,
            self.glb_eval_dataset,
            model=self.model
        )
        if self.config.data_args.dataset_name != "alpaca":
            self.evaluator._evaluate(self.current_epoch, self.rank)

    def _tensor_bytes(self, t):
        return t.numel() * t.element_size()

    def _train_epoch(self):
        tmp_loss, tmp_step = 0.0, 0
        for batch in cycle(self.train_loader):
            with self.timer("compute"):
                loss = self._train_step(batch)
            tmp_loss += loss
            tmp_step += 1
            self.glb_loss.append(loss)
            dist.barrier()
            if self.optimizer.global_step % self.config.ds_args.comm_frq == 0:
                with self.timer("comm"):
                    if "cmpr" in self.config.model_args.method_type:
                        received_params = self._block_exchange_cmpr()
                    else:
                        received_params = self._block_exchange()
                dist.barrier()
                self.optimizer._agg_blk_params(
                    received_params,
                    self.neighbor,
                    self.neighbor_rank_ns,
                    True if "cmpr" in self.config.model_args.method_type else False
                )
                if not any(k in self.config.model_args.method_type for k in ("avg", "tsm", "moment")):
                    self.optimizer._qg_update()
                elif "tsm" in self.config.model_args.method_type:
                    self.optimizer._qg_update_tsm()
                elif "moment" in self.config.model_args.method_type:
                    self.optimizer._qg_update_momentum()
            self.optimizer._fix_shadow()
            if self.optimizer.global_step % self.optimizer.block_switch_frequency == 0:
                if self.optimizer.block_sequence in ['importance','weighted', 'ucb']:
                    self.optimizer._blk_switch(self.bi)
                else:
                    self.optimizer._blk_switch()
            if tmp_step >= self.local_epochs:
                break
        tmp_loss /= tmp_step
        self.logger.info(f"Loss: {tmp_loss:.4f}")
        
    
    def _train_step(self, batch):
        # 单轮训练
        self.model.train()
        self.optimizer.zero_grad()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if self.config.training_args.optim != "mezo":
            with torch.cuda.amp.autocast(dtype=self.storage_dtype):
                outputs = self.model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                    ignore_index=-100
                )
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            del outputs, logits, labels, shift_logits, shift_labels, batch
            loss = loss.item()
        else:
            logits, loss = self.optimizer.step(batch)
            del batch
        if self.lr_scheduler:
            self.lr_scheduler.step()
            self.lr = self.optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        return loss
    
    # dense
    def _block_exchange(self): 
        received_params = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        comm_bytes = 0
        for _, param in self.optimizer.block_params:
            if not param.requires_grad:
                continue
            param_tmp = param.data
            param_buf = torch.zeros_like(param.data)
            tensor_bytes = param_tmp.numel() * param_tmp.element_size() 
            for src_rank in range(self.size):
                if self.rank in self.groups[src_rank]:
                    if src_rank == self.rank:
                        param_buf.copy_(param_tmp)
                    else:
                        param_buf.zero_()
                    dist.broadcast(tensor=param_buf, src=src_rank, group=self.dist_groups[src_rank])
                    if src_rank != self.rank:
                        comm_bytes += self._tensor_bytes(param_buf)
                    if src_rank in self.neighbor_rank_ns:
                        received_params[src_rank].append(param_buf.clone().to(self.device))
        comm_bytes /= 1024 ** 2
        self.ex_mb_history.append(comm_bytes)
        # self.logger.info(f"Exchange: {comm_bytes:.2f} MB")
        return received_params


    def _block_exchange_cmpr(self):
        received_params = {nb: [] for nb in self.neighbor_rank_ns}
        comm_bytes = 0
        for _, param in self.optimizer._get_comm_param():
            compressed = self.compressor.compress(param, self.optimizer.cur_blk_idx)
            val_buf = torch.zeros_like(compressed[0]); idx_buf = torch.zeros_like(compressed[1])
            for src_rank in range(self.size):
                if self.rank in self.groups[src_rank]:
                    if src_rank == self.rank:
                        val_buf.copy_(compressed[0])
                        idx_buf.copy_(compressed[1])
                    else:
                        val_buf.zero_()
                        idx_buf.zero_()
                    dist.broadcast(tensor=val_buf, src=src_rank, group=self.dist_groups[src_rank])
                    dist.broadcast(tensor=idx_buf, src=src_rank, group=self.dist_groups[src_rank])
                    if src_rank != self.rank:
                        comm_bytes += self._tensor_bytes(val_buf) + self._tensor_bytes(idx_buf)
                    if src_rank in self.neighbor_rank_ns:
                        rec = self.compressor.decompress(
                            (val_buf.clone().to(self.device), idx_buf.clone().to(self.device)),
                            param.shape,
                            self.device,
                        )
                        received_params[src_rank].append(rec)
        comm_bytes /= 1024 ** 2
        self.ex_mb_history.append(comm_bytes)
        # self.logger.info(f"Exchange: {comm_bytes:.2f} MB")
        return received_params

    def _agg_global(self, is_gpu=False):
        local_state = {
            k: v.clone().detach() if is_gpu else v.clone().detach().cpu() 
            for k, v in self.model.state_dict().items()
        }
        flat_params = {k: v.flatten() for k, v in local_state.items()}
        gathered_params = {
            k: [torch.zeros_like(flat_params[k]) for _ in range(self.size)]
            for k in flat_params
        }
        for k, tensor in flat_params.items():
            dist.all_gather(gathered_params[k], tensor)
        avg_state = {}
        for k, tensors in gathered_params.items():
            stacked = torch.stack(tensors, dim=0)
            avg_tensor = stacked.mean(dim=0)
            avg_state[k] = avg_tensor.view(local_state[k].shape)
            del stacked, avg_tensor, tensors
        self.model.load_state_dict(avg_state, strict=False)