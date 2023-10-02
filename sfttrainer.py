import transformers
import trl
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
import torch.nn as nn

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraModel, PeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from peft import PeftModel
from peft import tuners

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from huggingface_hub import login

from transformers import trainer
from trl import trainer as trl_trainer
from peft import PeftConfig, PeftModel, get_peft_model

from util_code import *
from custom_code import *
from custom_model_defs.opt import CustomOPTForCausalLM

"""
Fine-Tune Llama-7b on SE paired dataset
"""


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b")
    parser.add_argument("--model_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="/var/local/pbansal/dumps/earlyexit/checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()



# class CustomModel(PeftModel):

def run_training(args, train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform = list(range(12)),
    )

    train_data.start_iteration = 0

    print("Starting main loop")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        run_name="llama-7b-finetuned",
        report_to="wandb",
    )

    model = OPTForCausalLM.from_pretrained(args.model_path)
    # model = CustomOPTForCausalLM.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = PeftModelForCausalLM(model, lora_config, "default")

    # model.base_model.lm_head.requires_grad_(True)
    # model.base_model.early_exit_heads.requires_grad_(True)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks = [trl_trainer.utils.PeftSavingCallback],
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    # wandb login
    wandb.login(key="e45d2f6c4df62f742cc5974e9865de8bfeaacc00")
    # wandb.init(project=params.wandb_name, entity="pbansal")
    # wandb.run.name = '%s%0.3frs_weight_%s_%s_%0.2fcer_target_%s_model_%sseed_%0.1fgamma'%(params.prepend,this_weight,dataset,method,cer_target,model_name,params.seed,params.gamma)

    # huggingface login
    login(token='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')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)