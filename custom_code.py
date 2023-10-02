import transformers
import trl
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict
from torch.utils.data import Dataset

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
from peft import PeftModelForCausalLM
from util_code import *
from transformers import AutoTokenizer, OPTForCausalLM,AutoModelForCausalLM
from transformers.trainer_utils import speed_metrics
import time, math

# class CustomModel(PeftModelForCausalLM):
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         task_ids=None,
#         **kwargs,
#     ):
#         peft_config = self.active_peft_config
#         if not peft_config.is_prompt_learning:
#             if self.base_model.config.model_type == "mpt":
#                 if inputs_embeds is not None:
#                     raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
#                 return self.base_model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels,
#                     output_attentions=output_attentions,
#                     output_hidden_states=output_hidden_states,
#                     return_dict=return_dict,
#                     **kwargs,
#                 )

#             return self.base_model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs,
#             )

#         batch_size = _get_batch_size(input_ids, inputs_embeds)
#         if attention_mask is not None:
#             # concat prompt attention mask
#             prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
#             attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

#         if kwargs.get("position_ids", None) is not None:
#             warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
#             kwargs["position_ids"] = None
#         if kwargs.get("token_type_ids", None) is not None:
#             warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
#             kwargs["token_type_ids"] = None
#         kwargs.update(
#             {
#                 "attention_mask": attention_mask,
#                 "labels": labels,
#                 "output_attentions": output_attentions,
#                 "output_hidden_states": output_hidden_states,
#                 "return_dict": return_dict,
#             }
#         )

#         if peft_config.peft_type == PeftType.PREFIX_TUNING:
#             past_key_values = self.get_prompt(batch_size)
#             return self.base_model(
#                 input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
#             )
#         else:
#             if inputs_embeds is None:
#                 inputs_embeds = self.word_embeddings(input_ids)
#             # concat prompt labels
#             if labels is not None:
#                 prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
#                 kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
#             prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
#             prompts = prompts.to(inputs_embeds.dtype)
#             inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
#             return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

#     def generate(self, **kwargs):
#         self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
#         if hasattr(self.base_model, "model"):
#             self.base_model.model.generation_config = self.generation_config
#         else:
#             self.base_model.generation_config = self.generation_config
#         try:
#             outputs = self.base_model.generate(**kwargs)
#         except:
#             self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
#             raise
#         else:
#             self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
#             return outputs



class CustomTrainer(SFTTrainer):
    def unwrap_model(self,model: nn.Module) -> nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).

        Args:
            model (`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        layer_for_pred = np.arange(-12,0)
        
        outputs = model(**inputs,output_hidden_states = True)


        total_loss = 0        
        for layer_ in layer_for_pred:

            outputs['logits'] = model.lm_head(outputs['hidden_states'][layer_])

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                assert isinstance(model, PeftModel)
                model_name = self.unwrap_model(model.base_model)._get_name()
                # if model_name in trainer.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                #     loss = compute_label_loss(outputs, labels, shift_labels=True)
                # else:
                #     loss = compute_label_loss(outputs, labels)

                loss = compute_label_loss(outputs, labels, shift_labels=True)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            total_loss = total_loss + loss

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)


        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics