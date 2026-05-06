from abc import ABC, abstractmethod
from datasets import load_dataset
import numpy as np
from typing import Tuple, Optional, Any
import re

ALPACA_TASK_PRIORITY = (
    "Code",
    "Math",
    "Classification",
    "Editing",
    "Writing",
    "Reasoning",
    "Knowledge",
)

ALPACA_TASK_PATTERNS = {
    "Code": (
        r"\b(python|java(script)?|typescript|c\+\+|c#|java|go|rust|sql|html|css|bash|shell|linux|docker|git|regex|api|json|xml|yaml|database)\b",
        r"\b(code|coding|program|script|function|class|method|debug|bug|compile|refactor|implement|algorithm|pseudocode|query|web page|frontend|backend)\b",
        r"\b(unit test|test case|complexity|data structure|software|programming)\b",
    ),
    "Math": (
        r"\b(scientific notation|equation|algebra|geometry|calculus|integral|derivative|matrix|vector|probability|statistics|combinatorics|trigonometry)\b",
        r"\b(calculate|compute|solve|evaluate|simplify|differentiate|integrate|factor)\b",
        r"\b(fraction|decimal|percentage|percent|ratio|mean|median|mode|variance|standard deviation|perimeter|area)\b",
    ),
    "Classification": (
        r"\b(classify|classification|categorize|category|label|tag)\b",
        r"\b(sentiment|emotion|intent|topic|genre|language|spam|toxicity)\b",
        r"\b(determine whether|which category|what category)\b",
    ),
    "Editing": (
        r"\b(rewrite|edit|revise|paraphrase|proofread|summarize|translate|simplify|shorten|expand|improve|polish)\b",
        r"\b(fix|correct)\b.*\b(grammar|spelling|punctuation|sentence|wording|tone)\b",
        r"\b(clearer language|highlight any errors|improve the following|rewrite the following)\b",
    ),
    "Writing": (
        r"\b(write|draft|compose|generate|create)\b.*\b(story|poem|essay|article|blog|headline|title|bio|caption|slogan|tagline|email|letter|speech|script|dialogue|advertisement|review|joke|tweet|post)\b",
        r"\b(story|poem|essay|article|blog|headline|title|bio|caption|slogan|tagline|email|letter|speech|script|dialogue|advertisement|review|joke|tweet|post)\b",
        r"\b(creative writing|marketing copy|cover letter|product description)\b",
    ),
    "Reasoning": (
        r"\b(explain why|why does|why is|analyze|analysis|reasoning|justify|compare|contrast|infer|deduce|evaluate|assess|argue)\b",
        r"\b(step by step|pros and cons|trade[- ]?off|best approach|plan for|strategy for)\b",
    ),
    "Knowledge": (
        r"\b(what is|who is|when did|where is|define|describe|give an example|examples of|benefits of|causes of|difference between|how to)\b",
        r"\b(history|science|biology|physics|chemistry|economics|finance|marketing|business|health|medicine|law|education|geography|politics|cybersecurity|blockchain|cuisine)\b",
    ),
}


def _alpaca_text(example):
    parts = [
        example.get("instruction", ""),
        example.get("input", ""),
        example.get("question", ""),
    ]
    text = " ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", text).strip().lower()


def _count_pattern_matches(text, patterns):
    return sum(1 for pattern in patterns if re.search(pattern, text))


def classify_alpaca(example):
    text = _alpaca_text(example)
    scores = {
        task_type: _count_pattern_matches(text, patterns)
        for task_type, patterns in ALPACA_TASK_PATTERNS.items()
    }

    # Favor domain-specific buckets when a request clearly asks for an artifact.
    if re.search(r"\b(generate|create|write|draft|compose)\b", text) and re.search(
        r"\b(code|program|script|function|query|sql|html|css|javascript|python)\b", text
    ):
        scores["Code"] += 2
    if re.search(r"\b(generate|create|write|draft|compose)\b", text) and re.search(
        r"\b(story|poem|essay|headline|title|email|letter|speech|script|dialogue|slogan|tagline|caption)\b", text
    ):
        scores["Writing"] += 2
    if re.search(r"\b(solve|calculate|compute|evaluate)\b", text):
        scores["Math"] += 1
    if re.search(r"\b(classify|categorize|label|tag)\b", text):
        scores["Classification"] += 1

    best_score = max(scores.values())
    if best_score <= 0:
        task = "Knowledge"
    else:
        task = max(
            ALPACA_TASK_PRIORITY,
            key=lambda task_type: (scores[task_type], -ALPACA_TASK_PRIORITY.index(task_type)),
        )

    return {"task_type": task}


class BaseDataProcessor(ABC):

    def __init__(self, data_path, test_rate: float = 0.2, div_test_data_rate: float = 1.0):
        self.data_path = data_path
        self.test_rate = test_rate
        self.div_test_data_rate = div_test_data_rate

    @abstractmethod
    def load_and_process(self):
        pass
    
    def _split(self, train_dataset, test_dataset=None) -> Tuple[Any, Any]:
        has_test_set = test_dataset is not None and len(test_dataset) > 0
        if not has_test_set:
            split = train_dataset.train_test_split(test_size=self.test_rate, seed=42)
            return split["train"], split["test"]
        if self.div_test_data_rate < 1.0:
            split = test_dataset.train_test_split(
                test_size=self.div_test_data_rate, 
                seed=42
            )
            return train_dataset, split["test"]
        return train_dataset, test_dataset
    
# -----------------------------------
# ------  classify data
# --------------------------------

class NWGIProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['test']
        train = train.rename_columns({'question': 'prompt', 'context': 'question'})
        test = test.rename_columns({'question': 'prompt', 'context': 'question'})
        return self._split(train, test), list(set(test["answer"]))

class FIQAProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['test']
        train = train.rename_columns({'question': 'prompt', 'context': 'question'})
        test = test.rename_columns({'question': 'prompt', 'context': 'question'})
        return self._split(train, test), list(set(test["answer"]))

class EMOTIONProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['test']
        train = train.rename_columns({'question': 'prompt', 'context': 'question'})
        test = test.rename_columns({'question': 'prompt', 'context': 'question'})
        return self._split(train, test), list(set(test["answer"]))
    
class MMLUProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path, 'all')
        train, test = ds['auxiliary_train'], ds['test']
        def format_mmlu(ex):
            ex["prompt"] = "Answer the following multiple choice question."
            letters = [chr(ord('A') + i) for i in range(len(ex["choices"]))]
            options = "\n".join(f"{letters[i]}. {ex['choices'][i]}" for i in range(len(ex["choices"])))
            ex["question"] = f"{ex['question']}\nOptions:\n{options}"
            ex["label"] = ex["answer"]
            ex["answer"] = letters[ex["answer"]]
            return ex
        train = train.map(format_mmlu)
        test = test.map(format_mmlu)
        return self._split(train, test), list(set(test["answer"]))

class MNLIProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['validation']
        def format_mnli(ex):
            ex["prompt"] = "Please determine whether the hypothesis is entailment, neutral, or contradiction to the premise."
            ex["question"] = "The hypothesis is: "+ ex["text1"] + "\n" + "The premise is: " + ex["text2"]
            ex["answer"] = ex["label_text"]
            return ex
        train = train.map(format_mnli)
        test = test.map(format_mnli)
        return self._split(train, test), list(set(test["answer"]))
    
class TFNSProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['validation']
        label_dict = {"positive": 0, "neutral": 1, "negative": 2}
        def format_tfns(ex):
            ex["prompt"] = ex["question"]
            ex["question"] = re.sub(r" http\S+|www\.\S+", "", ex["context"])
            ex["label"] = label_dict[ex["answer"]]
            return ex
        train = train.map(format_tfns)
        test = test.map(format_tfns)
        return self._split(train, test), list(set(test["answer"]))

class AquaProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['validation']
        def format_aqua(ex):
            ex["prompt"] = "Complete the following multiple-choice questions and provide the correct answer."
            options = "\n".join(ex["options"])
            ex["question"] = f"{ex['question']}\nOptions:\n{options}"
            ex["answer"] = ex["correct"]
            ex["label"] =  ord(ex["answer"]) - ord('A')
            return ex
        train = train.map(format_aqua)
        test = test.map(format_aqua)
        return self._split(train, test), list(set(test["answer"]))

class FPBProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['test']
        def format_fpb(ex):
            ex["prompt"] = "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral."
            ex["question"] = ex["text"]
            return ex
        train = train.rename_columns({'gold': 'label'})
        test = test.rename_columns({'gold': 'label'})
        train = train.map(format_fpb)
        test = test.map(format_fpb)
        return self._split(train, test), list(set(test["answer"]))
    
class AGNewsProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train, test = ds['train'], ds['test']
        label_dict = {0: "World",1: "Sports",2: "Business", 3: "Sci/Tech"}
        def format_ag(ex):
            ex["prompt"] = "Please classify the following news into categories. Provide your answer as either World, Sports, Business, or Sci/Tech."
            ex["question"] = ex["text"]
            ex["answer"] = label_dict[ex["label"]]
            return ex
        train = train.map(format_ag)
        test = test.map(format_ag)
        return self._split(train, test), list(label_dict.values())
    
# class ARCEProcessor(BaseDataProcessor):
#     def load_and_process(self):
#         ds = load_dataset(self.data_path, "ARC-Easy")
#         train, test = ds['train'], ds['test']
#         label_dict = {'1': "A", '2': "B",'3': "C", '4': "D"}
#         def format_arc(ex):
#             ex["prompt"] = "Complete the following multiple-choice questions and provide the correct answer."
#             options = "\n".join([f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])])
#             ex["question"] = f"{ex['question']}\nOptions:\n{options}"
#             ex["answer"] = label_dict.get(ex["answerKey"], ex["answerKey"])
#             ex["label"] =  ord(ex["answerKey"]) - ord('A')
#             return ex
#         train = train.map(format_arc)
#         test = test.map(format_arc)
#         return self._split(train, test), list(label_dict.values())
    
# class ARCCProcessor(BaseDataProcessor):
#     def load_and_process(self):
#         ds = load_dataset(self.data_path, "ARC-Challenge")
#         train, test = ds['train'], ds['test']
#         label_dict = {'1': "A", '2': "B",'3': "C", '4': "D"}
#         def format_arc(ex):
#             ex["prompt"] = "Complete the following multiple-choice questions and provide the correct answer."
#             options = "\n".join([f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])])
#             ex["question"] = f"{ex['question']}\nOptions:\n{options}"
#             ex["answer"] = label_dict.get(ex["answerKey"], ex["answerKey"])
#             ex["label"] =  ord(ex["answerKey"]) - ord('A')
#             return ex
#         train = train.map(format_arc)
#         test = test.map(format_arc)
#         return self._split(train, test), list(label_dict.values())
  

# -----------------------------------
# ------  generate data
# --------------------------------

# class AlpacaProcessor(BaseDataProcessor):
#     def load_and_process(self):
#         ds = load_dataset(self.data_path)
#         train, test = ds['train'], ds['test']
#         def format_alpaca(ex):
#             prompt_base = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#             prompt_complex = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
#             if ex.get("input"):
#                 ex["prompt"] = prompt_complex
#                 ex["question"] = f"Instruction: {ex['instruction']}\nInput: {ex['input']}"
#             else:
#                 ex["prompt"] = prompt_base
#                 ex["question"] = f"Instruction: {ex['instruction']}"
#             ex["answer"] = ex["output"]
#             return ex
#         train = train.map(format_alpaca)
#         train = train.map(classify_alpaca)
#         test = test.map(format_alpaca)
#         return self._split(train, test), None

# class AlpacaProcessor(BaseDataProcessor):
#     def load_and_process(self):
#         ds = load_dataset(self.data_path)
#         train, test = ds['train'], ds['test']
#         def format_alpaca(ex):
#             prompt_base = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#             prompt_complex = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
#             if ex.get("input"):
#                 ex["prompt"] = prompt_complex
#                 ex["question"] = f"Instruction: {ex['instruction']}\nInput: {ex['input']}"
#             else:
#                 ex["prompt"] = prompt_base
#                 ex["question"] = f"Instruction: {ex['instruction']}"
#             ex["answer"] = ex["output"]
#             return ex
#         train = train.map(format_alpaca)
#         train = train.map(classify_alpaca)
#         test = test.map(format_alpaca)
#         test = test.map(classify_alpaca)
#         return self._split(train, test), None

class AlpacaProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path)
        train = ds['train']
        def format_alpaca(ex):
            prompt_base = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            prompt_complex = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
            instruction = str(ex.get("instruction", "")).strip()
            input_text = str(ex.get("input", "")).strip()
            if input_text:
                ex["prompt"] = prompt_complex
                ex["question"] = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
            else:
                ex["prompt"] = prompt_base
                ex["question"] = f"### Instruction:\n{instruction}"
            ex["answer"] = str(ex.get("output", ""))
            return ex
        train = train.map(format_alpaca)
        train = train.map(classify_alpaca)
        return self._split(train, train), None


class ARCEProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path, "ARC-Easy")
        train, test = ds['train'], ds['test']
        label_dict = {'1': "A", '2': "B",'3': "C", '4': "D"}
        def format_arc(ex):
            ex["prompt"] = "Complete the following multiple-choice questions and provide the correct answer."
            options = "\n".join([f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])])
            ex["question"] = f"{ex['question']}\nOptions:\n{options}"
            idx = ex["choices"]["label"].index(ex["answerKey"])
            ex["answer"] = ex["choices"]["text"][idx]
            return ex
        train = train.map(format_arc)
        test = test.map(format_arc)
        return self._split(train, test), None
    
class ARCCProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path, "ARC-Challenge")
        train, test = ds['train'], ds['test']
        label_dict = {'1': "A", '2': "B",'3': "C", '4': "D"}
        def format_arc(ex):
            ex["prompt"] = "Complete the following multiple-choice questions and provide the correct answer."
            options = "\n".join([f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])])
            ex["question"] = f"{ex['question']}\nOptions:\n{options}"
            idx = ex["choices"]["label"].index(ex["answerKey"])
            ex["answer"] = ex["choices"]["text"][idx]
            return ex
        train = train.map(format_arc)
        test = test.map(format_arc)
        return self._split(train, test), None

class GSM8KProcessor(BaseDataProcessor):
    def load_and_process(self):
        ds = load_dataset(self.data_path, "main")
        train, test = ds['train'], ds['test']
        def format_gsm8k(ex):
            ex["prompt"] = "Solve the following math problem, providing the derivation process and the final answer."
            return ex
        train = train.map(format_gsm8k)
        test = test.map(format_gsm8k)
        return self._split(train, test), None
    
# 注册表
PROCESSOR_MAP = {
    "nwgi": NWGIProcessor,
    "fiqa": FIQAProcessor,
    "emotion": EMOTIONProcessor,
    "alpaca": AlpacaProcessor,
    "mmlu": MMLUProcessor,
    "mnli": MNLIProcessor,
    "tfns": TFNSProcessor,
    "aqua": AquaProcessor,
    "fpb": FPBProcessor,
    "ag_news": AGNewsProcessor,
    "arc-easy": ARCEProcessor,
    "arc-challenge": ARCCProcessor,
    "gsm8k": GSM8KProcessor
}

def get_processor(data_name: str, path: str, **kwargs) -> BaseDataProcessor:
    if data_name not in PROCESSOR_MAP:
        raise ValueError(f"Processor for {data_name} not found.")
    return PROCESSOR_MAP[data_name](path, **kwargs)