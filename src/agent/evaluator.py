import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from data_loader.manager import *
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, HfArgumentParser, Trainer, AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from src.utils.util import *
from src.utils.topology import Graph
from src.utils.metrics import TextQualityEvaluator
from src.utils.argument import ModelArguments, DataArguments, TrainingArguments, DSArguments, LoRAArguments
from agent.agent_util import Timer, DirectoryManager, CheckpointManager, SystemMonitor, LossPlotter, ResponseHandler
from sklearn.metrics import classification_report
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score
import sacrebleu
import nltk
from pycocoevalcap.cider.cider import Cider
import warnings
warnings.filterwarnings("ignore")


@dataclass 
class AgentConfig:
    model_args: ModelArguments
    lora_args: LoRAArguments
    ds_args: DSArguments
    data_args: DataArguments
    training_args: TrainingArguments

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

    
class Evaluator:
    def __init__(
        self, 
        config: DecentralizedConfig,
        tokenizer: AutoTokenizer,
        eval_dataset: Dataset,
        ckpt_path: str = None,
        model: nn.Module = None
    ):
        self.config = config
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.ckpt_path = ckpt_path

        # 数据管理
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config.training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False
        ) if self.eval_dataset is not None else None

        self.storage_dtype = getattr(config, 'storage_dtype', torch.bfloat16)  # 存储精度 bf16/fp16
        self.compute_dtype = getattr(config, 'compute_dtype', torch.float32)   # 计算精度 fp32
        
        # 模型初始化
        self.initialize_model() if model is None else setattr(self, 'model', model)
        self.ckpt_manager = CheckpointManager(config)   # 使用 CheckpointManager 保存
        self.text_evaluator = TextQualityEvaluator(self.tokenizer)
        self.handler = ResponseHandler(self.eval_dataset.label_map)


    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_args.base_model_path,
            torch_dtype=self.storage_dtype,
            use_flash_attention_2=False,
            trust_remote_code=True
        )
        state_dict_path = (
            self.ckpt_path
            if hasattr(self, "ckpt_path") and self.ckpt_path is not None
            else getattr(self.config.model_args, "ckpt_path", None)
        )
        if "lora" in self.config.model_args.method_type or "caf" in self.config.model_args.method_type:
            self.model = PeftModel.from_pretrained(self.model, state_dict_path)
            self.model.merge_and_unload()
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def _generate(self):
        self.model.eval()
        reference_texts, generated_texts = [], []
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model.generate(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.config.model_args.max_new_tokens,
                    do_sample=False,
                    top_p=0.9,
                    temperature=0.3,
                    use_cache=True
                )
                if "prompt" in self.config.model_args.method_type:
                    generated_batch = self.tokenizer.batch_decode(
                        outputs[:, :],
                        skip_special_tokens=True
                    )
                else:
                    generated_batch = self.tokenizer.batch_decode(
                        outputs[:, batch["input_ids"].shape[-1]:],
                        skip_special_tokens=True
                    )
                generated_texts.extend(self.handler.handle_batch_responses(generated_batch, 20))
                if isinstance(batch["answer"], list):
                    reference_texts.extend(batch["answer"])
                else:
                    reference_texts.extend(batch["answer"].tolist() if isinstance(batch["answer"], torch.Tensor) else [batch["answer"]])
                del batch, outputs
                torch.cuda.empty_cache()
        return reference_texts, generated_texts
    
    def _evaluate(self, epoch, rank=-1):
        self.device = torch.device(f'cuda:{self.config.ds_args.device_index}') if rank == -1 else torch.device(f'cuda:{rank}')
        self.model.to(device=self.device, dtype=self.compute_dtype)
        reference_texts, generated_texts = self._generate()
        report_dict = classification_report(reference_texts, generated_texts, output_dict=True, digits=4)
        result_dict = {
            "dataset_name": self.config.data_args.dataset_name,
            "method": self.config.model_args.method_type,
            "alpha": self.config.data_args.dirichlet_alpha,
            "seed": self.config.ds_args.seed,
            "topo": self.config.ds_args.topology,
            "epoch": epoch,
            "Accuracy": round(report_dict["accuracy"], 4),
            "Precision": round(report_dict["macro avg"]["precision"], 4),
            "Recall": round(report_dict["macro avg"]["recall"], 4),
            "F1": round(report_dict["macro avg"]["f1-score"], 4)
        }
        self.ckpt_manager._generation(reference_texts, generated_texts, rank=rank)
        self.ckpt_manager._metrics(result_dict, epoch, rank=rank)

    def _compute_rouge(self, refs, gens):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        total_r1, total_r2, total_l = 0, 0, 0
        n = len(refs)
        for ref, gen in zip(refs, gens):
            score = scorer.score(ref, gen)
            total_r1 += score["rouge1"].fmeasure
            total_r2 += score["rouge2"].fmeasure
            total_l += score["rougeL"].fmeasure
        return {
            "rouge1": total_r1 / n,
            "rouge2": total_r2 / n,
            "rougeL": total_l / n,
        }

    def _compute_bleu(self, refs, gens):
        bleu = sacrebleu.corpus_bleu(gens, [refs])
        return bleu.score / 100.0

    def _compute_meteor(self, refs, gens):
        total = 0
        for r, g in zip(refs, gens):
            # pre-tokenized tokens
            r_tokens = nltk.word_tokenize(r)   # 或 r.split()
            g_tokens = nltk.word_tokenize(g)
            total += meteor_score([r_tokens], g_tokens)
        return total / len(refs)

    def _compute_cider(self, refs, gens):
        refs_dict = {i: [refs[i]] for i in range(len(refs))}
        hyps_dict = {i: [gens[i]] for i in range(len(gens))}
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(refs_dict, hyps_dict)
        return score

    def _evaluate_gen(self, rank=-1):
        self.device = torch.device(f'cuda:{self.config.ds_args.device_index}') if rank == -1 else torch.device(f'cuda:{rank}')
        self.model.to(device=self.device, dtype=self.compute_dtype)
        reference_texts, generated_texts = self._generate()
        bleu_score   = self._compute_bleu(reference_texts, generated_texts)
        rouge_scores = self._compute_rouge(reference_texts, generated_texts)
        meteor_avg   = self._compute_meteor(reference_texts, generated_texts)
        cider_avg    = self._compute_cider(reference_texts, generated_texts)
        result_dict = {
            "dataset_name": self.config.data_args.dataset_name,
            "method": self.config.model_args.method_type,
            "alpha": self.config.data_args.dirichlet_alpha,
            "BLEU": round(bleu_score, 4),
            "ROUGE1": round(rouge_scores["rouge1"], 4),
            "ROUGE2": round(rouge_scores["rouge2"], 4),
            "ROUGEL": round(rouge_scores["rougeL"], 4),
            "METEOR": round(meteor_avg, 4),
            "CIDEr": round(cider_avg, 4),
        }
        self.ckpt_manager._metrics_gen(result_dict)
        return result_dict
