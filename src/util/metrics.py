import numpy as np
import deepspeed
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextStreamer,
    HfArgumentParser, 
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import evaluate
from Levenshtein import distance
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score
import sacrebleu
from pycocoevalcap.cider.cider import Cider
import warnings
warnings.filterwarnings("ignore")


class TextQualityEvaluator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval(self, preds, labels, metrics):
        results = {}
        if 'bleu' in metrics:
            results.update(self.bleu_score(preds, labels))
        if 'rouge1' in metrics or 'rouge2' in metrics or 'rougeL' in metrics:
            results.update(self.rouge_score(preds, labels))
        if 'levenshtein' in metrics:
            results.update(self.lst_distance(preds, labels))
        if 'meteor' in metrics:
            results.update(self.meteor_score(preds, labels))
        if 'truth_ratio' in metrics:
            results.update(self.truth_ratio(preds, labels))
        if 'entailment' in metrics:
            results.update(self.entailment_score(preds, labels))
        if 'similarity' in metrics:
            results.update(self.similarity_score(preds, labels))
        return results

    def _preprocess_text(self, text):
        if isinstance(text, list):
            return [str(t).strip() for t in text]
        else:
            return str(text).strip()
            
    def _tokenize_text(self, text):
        if isinstance(text, str):
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(text.lower())
                return tokens
            else:
                return nltk.word_tokenize(text.lower())
        return text

    def bleu_score(self, preds, labels, n_gram=4, max_gram=4):
        """BLEU"""
        bleu_scores = {}
        smooth = SmoothingFunction().method1
        
        preds = self._preprocess_text(preds)
        labels = self._preprocess_text(labels)
        if len(labels) > 0 and not isinstance(labels[0], list):
            labels = [[label] for label in labels]
        tokenized_preds = [self._tokenize_text(pred) for pred in preds]
        tokenized_labels = [[self._tokenize_text(ref) for ref in refs] for refs in labels]
        for n in range(1, n_gram + 1):
            weights = tuple([1.0 / n] * n + [0] * (max_gram - n))
            scores = []
            for pred, refs in zip(tokenized_preds, tokenized_labels):
                try:
                    score = sentence_bleu(refs, pred, weights=weights, smoothing_function=smooth)
                    scores.append(score)
                except:
                    scores.append(0.0)
            bleu_scores[f'BLEU-{n}'] = round(np.mean(scores).item(), 3)
        try:
            overall_bleu = corpus_bleu(tokenized_labels, tokenized_preds, smoothing_function=smooth)
            bleu_scores['bleu'] = round(overall_bleu, 3)
        except:
            bleu_scores['bleu'] = 0.0
        return bleu_scores

    
    def rouge_score(self, preds, labels):
        """rouge"""
        gnrt_txt = self._preprocess_text(preds)
        orgn_txt = self._preprocess_text(labels)
        if not hasattr(self, 'rouge_metric'):
            self.rouge_metric = evaluate.load("rouge")
        rouge_index = ['rouge1', 'rouge2', 'rougeL']
        try:
            if isinstance(gnrt_txt, list):
                results = {ri: [] for ri in rouge_index}
                for ri in rouge_index:
                    scorer = rouge_scorer.RougeScorer([ri], use_stemmer=True, tokenizer=self.tokenizer)
                    for ot, gt in zip(orgn_txt, gnrt_txt):
                        results[ri].append(scorer.score(ot, gt)[ri].fmeasure)
                results = {ri: np.mean(results[ri]) for ri in rouge_index}
                return results
            else:
                results = {}
                for ri in rouge_index:
                    scorer = rouge_scorer.RougeScorer([ri], use_stemmer=True, tokenizer=self.tokenizer)
                    results[ri] = scorer.score(orgn_txt, gnrt_txt)[ri].fmeasure
                return results
        except Exception as e:
            return {}

            
    def lst_distance(self, preds, labels):
        preds = self._preprocess_text(preds)
        labels = self._preprocess_text(labels)
        if len(labels) > 0 and isinstance(labels[0], list):
            labels = [label[0] for label in labels]
        distances = []
        for pred, label in zip(preds, labels):
            dist = distance(pred, label)
            distances.append(dist)
        return {
            "levenshtein": round(np.mean(distances).item(), 3)
        }

    
    def meteor_score(self, preds, labels):
        preds = self._preprocess_text(preds)
        labels = self._preprocess_text(labels)
        if len(labels) > 0 and isinstance(labels[0], list):
            labels = [label[0] for label in labels]
        meteor_scores = []
        for pred, label in zip(preds, labels):
            try:
                pred_tokens = self._tokenize_text(pred)
                label_tokens = self._tokenize_text(label)
                score = meteor_score([label_tokens], pred_tokens)
                meteor_scores.append(score)
            except Exception as e:
                meteor_scores.append(0.0)
        
        return {
            "meteor": round(np.mean(meteor_scores).item(), 3)
        }
    
    def truth_ratio(self, preds, labels):
        gnrt_txt = self._preprocess_text(preds)
        orgn_txt = self._preprocess_text(labels)
        if not hasattr(self, 'gnrtq_model'):
            self.gnrtq_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self.gnrtq_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        def sentence_likelihood(sentence):
            with torch.no_grad():
                inputs = self.gnrtq_tokenizer(sentence, return_tensors="pt").to(self.device)
                outputs = self.gnrtq_model(**inputs, labels=inputs["input_ids"])
            return np.exp(-outputs.loss.item())
        if isinstance(gnrt_txt, list):
            p_correct = [sentence_likelihood(pred) for pred in gnrt_txt]
            p_incorrect = [sentence_likelihood(label) for label in orgn_txt]
            truth_ratios = [pc / (pi + 1e-12) for pc, pi in zip(p_correct, p_incorrect)]
            return {"truth_ratio": np.mean(truth_ratios)}
        else:
            p_correct = sentence_likelihood(preds)
            p_incorrect = sentence_likelihood(labels)
            truth_ratio = p_correct / (p_incorrect + 1e-12)
            return {"truth_ratio": truth_ratio}

    def entailment_score(self, preds, labels):
        gnrt_txt = self._preprocess_text(preds)
        orgn_txt = self._preprocess_text(labels)
        if not hasattr(self, 'entailment_model'):
            self._entailment_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli").to(self.device)
            self._entailment_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli") 
        if isinstance(gnrt_txt, list):
            entailment_probs = []
            for gt, ot in zip(gnrt_txt, orgn_txt):
                with torch.no_grad():
                    inputs = self._entailment_tokenizer(gt, ot, return_tensors="pt", truncation=True).to(self.device)
                    logits = self._entailment_model(**inputs).logits
                    entailment_prob = torch.softmax(logits, dim=-1)[0][2].item()
                    entailment_probs.append(entailment_prob)
            return {"entailment": round(np.mean(entailment_probs), 3)}
        else:
            with torch.no_grad():
                inputs = self._entailment_tokenizer(gnrt_txt, orgn_txt, return_tensors="pt", truncation=True).to(self.device)
                logits = self._entailment_model(**inputs).logits
                entailment_prob = torch.softmax(logits, dim=-1)[0][2].item()
            return {"entailment": round(entailment_prob, 3)}

    def similarity_score(self, preds, labels):
        gnrt_txt = self._preprocess_text(preds)
        orgn_txt = self._preprocess_text(labels)
        if not hasattr(self, '_sentence_embedder'):
            self._sentence_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        if isinstance(gnrt_txt, list):
            similarities = []
            for gt, ot in zip(gnrt_txt, orgn_txt):
                with torch.no_grad():
                    embeddings = self._sentence_embedder.encode([gt, ot], convert_to_tensor=True, device=self.device)
                    similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
                    similarities.append(similarity)
            return {"similarity": round(np.mean(similarities), 3)}
        else:
            with torch.no_grad():
                embeddings = self._sentence_embedder.encode([gnrt_txt, orgn_txt], convert_to_tensor=True, device=self.device)
                similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
            return {"similarity": round(similarity, 3)}