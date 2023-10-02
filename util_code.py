import transformers
import trl
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
import torch.nn as nn
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from peft import PeftModel

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from huggingface_hub import login

from transformers import trainer
from trl import trainer as trl_trainer
from peft import PeftConfig, PeftModel, get_peft_model




def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name_, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    
def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    return example['text']

def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        num_proc=args.num_workers
    )
    dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset



def compute_label_loss(model_output, labels, shift_labels=False):
    ignore_index: int = -100
    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    log_probs = -nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32

    nll_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    return nll_loss


