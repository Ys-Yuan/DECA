from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch.utils.data import Dataset
from typing import List


class ChatTemplate(ABC):
    @abstractmethod
    def format(self, prompt: str, question: str, answer: str) -> Tuple[str, str]:
        pass

    @property
    @abstractmethod
    def ignore_token_id(self) -> int:
        return -100

class Llama3Template(ChatTemplate):
    def format(self, prompt, question, answer):
        # Llama3
        system = f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. {prompt}<|eot_id|>\n"
        user = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n"
        assistant_header = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        context = f"<|begin_of_text|>{system}{user}{assistant_header}"
        response = f"{answer}<|eot_id|>"
        return context, response

    @property
    def ignore_token_id(self):
        return -100

class Llama2Template(ChatTemplate):
    def format(self, prompt, question, answer):
        # Llama2 chat format ([INST] ... [/INST])
        system_prompt = "You are a helpful assistant."
        prompt = (prompt or "").strip()
        question = (question or "").strip()
        answer = "" if answer is None else str(answer)
        if prompt:
            system_prompt = f"{system_prompt} {prompt}"

        context = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question} [/INST] "
        response = answer
        return context, response

    @property
    def ignore_token_id(self):
        return -100

class Qwen2Template(ChatTemplate):
    def format(self, prompt, question, answer):
        # Qwen2 instruct/chat uses the same ChatML-style format.
        system = f"<|im_start|>system\nYou are a helpful assistant. {prompt}<|im_end|>\n"
        user = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_header = f"<|im_start|>assistant\n"

        context = f"{system}{user}{assistant_header}"
        response = f"{answer}<|im_end|>"
        return context, response

    @property
    def ignore_token_id(self):
        return -100
        
class Qwen25Template(ChatTemplate):
    def format(self, prompt, question, answer):
        # Qwen2.5
        system = f"<|im_start|>system\nYou are a helpful assistant. {prompt}<|im_end|>\n"
        user = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_header = f"<|im_start|>assistant\n"
        
        context = f"{system}{user}{assistant_header}"
        response = f"{answer}<|im_end|>"
        return context, response

    @property
    def ignore_token_id(self):
        return -100


class Qwen3Template(ChatTemplate):
    def format(self, prompt, question, answer):
        # Qwen3 instruct/chat style is compatible with the <|im_start|> chat format.
        system = f"<|im_start|>system\nYou are a helpful assistant. {prompt}<|im_end|>\n"
        user = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_header = f"<|im_start|>assistant\n"

        context = f"{system}{user}{assistant_header}"
        response = f"{answer}<|im_end|>"
        return context, response

    @property
    def ignore_token_id(self):
        return -100


class QwenVLTemplateBase(ChatTemplate):
    IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
    VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"

    def _normalize_visual_placeholders(self, text: str) -> str:
        if not text:
            return text
        normalized = text
        image_markers = ["<image>", "[IMAGE]", "<img>", "{image}"]
        video_markers = ["<video>", "[VIDEO]", "{video}"]
        for marker in image_markers:
            normalized = normalized.replace(marker, self.IMAGE_TOKEN)
        for marker in video_markers:
            normalized = normalized.replace(marker, self.VIDEO_TOKEN)
        return normalized

    def _system_text(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if prompt:
            return f"You are a helpful assistant. {prompt}"
        return "You are a helpful assistant."

    @property
    def ignore_token_id(self):
        return -100


class Qwen25VLTemplate(QwenVLTemplateBase):
    def format(self, prompt, question, answer):
        system = f"<|im_start|>system\n{self._system_text(prompt)}<|im_end|>\n"
        user = f"<|im_start|>user\n{self._normalize_visual_placeholders(question)}<|im_end|>\n"
        assistant_header = "<|im_start|>assistant\n"

        context = f"{system}{user}{assistant_header}"
        response = f"{answer}<|im_end|>"
        return context, response


class Qwen3VLTemplate(QwenVLTemplateBase):
    def format(self, prompt, question, answer):
        system = f"<|im_start|>system\n{self._system_text(prompt)}<|im_end|>\n"
        user = f"<|im_start|>user\n{self._normalize_visual_placeholders(question)}<|im_end|>\n"
        assistant_header = "<|im_start|>assistant\n"

        context = f"{system}{user}{assistant_header}"
        response = f"{answer}<|im_end|>"
        return context, response

class HaiRuoTemplate(ChatTemplate):
    """
    HaiRuo-13B chat template matching tokenizer_config.json:

        {system}<reserved_106>{question}<reserved_107>{answer}

    Requires a BaichuanTokenizer loaded from the model's config directory.
    EOS token (</s>) is appended by TrainDataset via tokenizer.eos_token_id.
    """
    USER_TOKEN = "<reserved_106>"
    ASST_TOKEN = "<reserved_107>"

    def format(self, prompt, question, answer):
        system = prompt.strip() if prompt else ""
        context = f"{system}{self.USER_TOKEN}{question}{self.ASST_TOKEN}"
        response = answer
        return context, response

    @property
    def ignore_token_id(self):
        return -100


TEMPLATE_MAP = {
    "llama2-7B": Llama2Template,
    "llama2-13B": Llama2Template,
    "llama3-8B": Llama3Template,
    "qwen2": Qwen2Template,
    "qwen2.5": Qwen25Template,
    "qwen2.5-14B": Qwen25Template,
}

def get_template(model_type: str) -> ChatTemplate:
    if model_type not in TEMPLATE_MAP:
        raise ValueError(f"Template for {model_type} not found.")
    return TEMPLATE_MAP[model_type]()


class LLMBaseDataset(Dataset):
    def __init__(self, data, tokenizer, template, max_len, indices=None, label_map=None):
        self.data = data
        self.tokenizer = tokenizer
        self.template = template
        self.max_len = max_len
        self.label_map = label_map
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.pad_token_id = self.tokenizer.eos_token_id
        if indices is not None:
            self.data = self.data.select(indices)

    def __len__(self):
        return len(self.data)
    
    def _pad_crop(self, input_ids, labels=None, padding_side=None):
        side = padding_side or getattr(self.tokenizer, "padding_side", "right")
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            pad_ids = [self.pad_token_id] * pad_len
            pad_lbl = ([self.template.ignore_token_id] * pad_len) if labels is not None else None
            if side == "right":
                attention_mask = [1] * len(input_ids) + [0] * pad_len
                input_ids = input_ids + pad_ids
                if labels is not None:
                    labels = labels + pad_lbl
            else:  # left pad
                attention_mask = [0] * pad_len + [1] * len(input_ids)
                input_ids = pad_ids + input_ids
                if labels is not None:
                    labels = pad_lbl + labels
        else:
            if side == "right":
                input_ids = input_ids[:self.max_len]
                if labels is not None:
                    labels = labels[:self.max_len]
            else:
                input_ids = input_ids[-self.max_len:]
                if labels is not None:
                    labels = labels[-self.max_len:]
            attention_mask = [1] * len(input_ids)
            
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            torch.tensor(attention_mask, dtype=torch.long),
        )

class TrainDataset(LLMBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        # Template
        context_str, response_str = self.template.format(item['prompt'], item['question'], item['answer'])
        # Tokenization
        context_ids = self.tokenizer(context_str, add_special_tokens=False)['input_ids']
        response_ids = self.tokenizer(response_str, add_special_tokens=False)['input_ids']
        # Input and Labels
        input_ids = context_ids + response_ids
        labels = [self.template.ignore_token_id] * len(context_ids) + response_ids
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (len(input_ids) == 0 or input_ids[-1] != eos_id):
            input_ids.append(eos_id)
            labels.append(eos_id)
        # For causal LM training, keep right padding so labels align naturally.
        input_tensor, label_tensor, attention_mask = self._pad_crop(input_ids, labels, padding_side="right")
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }

class TestDataset(LLMBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        context_str, response_str = self.template.format(item['prompt'], item['question'], item['answer'])
        context_ids = self.tokenizer(context_str, add_special_tokens=False)['input_ids']
        # For batched generation on decoder-only LMs, enforce left padding.
        input_ids, _, attention_mask = self._pad_crop(context_ids, padding_side="left")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer": item['answer']
        }
