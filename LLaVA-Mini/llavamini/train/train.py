# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import pickle
import torch
import random
import yaml
import transformers
import tokenizers

from llavamini.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,DEFAULT_VIDEO_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llavamini.train.llava_trainer import LLaVATrainer

from llavamini import conversation as conversation_lib
from llavamini.model import *
from llavamini.mm_utils import tokenizer_image_token

from PIL import Image

import numpy as np
import decord
from decord import VideoReader, cpu
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_prefusion: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    compressor_size: Optional[int] = field(default=2)
    resolution_ratio:int = field(default=2)
    prefusion_layer_num: Optional[int] = field(default=4)


@dataclass
class DataArguments:
    # data_path: str = field(default=None,
    #                        metadata={"help": "Path to the training data."})
    data_paths: List[str] = field(default_factory=list,
                                  metadata={"help": "Paths to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    # image_folder: Optional[str] = field(default=None)
    image_folders: List[str] = field(default_factory=list,
                                     metadata={"help": "Paths to the image folders."})
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_compressor: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'compressor', 'prefusion_layers']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'compressor', 'prefusion_layers']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_param_info(model,output_dir):
    parameters_info = []
    for name, param in model.named_parameters():
        parameters_info.append((name, param.size(), param.requires_grad))

    # 将信息保存到文件中
    output_file = f"{output_dir}/model_parameters_info.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for name, size, requires_grad in parameters_info:
            f.write(f"{name}: Size={size}, {'Trainable' if requires_grad else 'Frozen'}\n")



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        idx=0
        for sentence in source:

            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        idx+=1
    return sources

def preprocess_multimodal_video(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        idx=0
        for sentence in source:
            if DEFAULT_VIDEO_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
                sentence['value'] = DEFAULT_VIDEO_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)
        idx=1
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            
            if i > 0:
                # round_len -= 1
                instruction_len -= 1
            
            

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(input_ids.size(),targets.size())
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
        sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets

    sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
    #sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds= [rounds[0]] + [rounds[idx] + rounds[idx+1] for idx in range(1, len(rounds)-1, 2)]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(tokenizer(rou, add_special_tokens=False).input_ids)

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3_1(
        sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
    #sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds= [rounds[0]] + [rounds[idx] + rounds[idx+1] for idx in range(1, len(rounds)-1, 2)]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(tokenizer(rou, add_special_tokens=False).input_ids)

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len= cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1:
        return preprocess_llama_3_1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_paths: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))

        # rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = tokenizer
        # self.list_data_dict = list_data_dict
        # self.data_args = data_args

        # self.resolution_ratio=data_args.resolution_ratio
        # print(f"****** Resolution_ratio = {self.resolution_ratio} ******")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = []

        # Ensure data_paths and image_folders are of the same length
        if len(data_paths) != len(data_args.image_folders):
            raise ValueError("The number of data paths must match the number of image folders.")

        for data_path, image_folder in zip(data_paths, data_args.image_folders):
            if data_path.endswith(".yaml"):
                with open(data_path, "r") as file:
                    yaml_data = yaml.safe_load(file)
                    datasets = yaml_data.get("datasets")
                    # file should be in the format of:
                    # datasets:
                    #   - json_path: xxxx1.json
                    #     sampling_strategy: first:1000
                    #   - json_path: xxxx2.json
                    #     sampling_strategy: end:3000
                    #   - json_path: xxxx3.json
                    #     sampling_strategy: random:999
                    data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                    for dataset in datasets:
                        json_path = dataset.get("json_path")
                        sampling_strategy = dataset.get("sampling_strategy", "all")
                        sampling_number = None

                        rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                        if json_path.endswith(".jsonl"):
                            cur_data_dict = []
                            with open(json_path, "r") as json_file:
                                for line in json_file:
                                    cur_data_dict.append(json.loads(line.strip()))
                        elif json_path.endswith(".json"):
                            with open(json_path, "r") as json_file:
                                cur_data_dict = json.load(json_file)
                        else:
                            raise ValueError(f"Unsupported file type: {json_path}")

                        if ":" in sampling_strategy:
                            sampling_strategy, sampling_number = sampling_strategy.split(":")
                            if "%" in sampling_number:
                                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                            else:
                                sampling_number = int(sampling_number)

                        # Apply the sampling strategy
                        if sampling_strategy == "first" and sampling_number is not None:
                            cur_data_dict = cur_data_dict[:sampling_number]
                        elif sampling_strategy == "end" and sampling_number is not None:
                            cur_data_dict = cur_data_dict[-sampling_number:]
                        elif sampling_strategy == "random" and sampling_number is not None:
                            state = random.getstate()  # 保存当前随机状态
                            random.seed(1)
                            random.shuffle(cur_data_dict)
                            random.setstate(state)  # 恢复之前的随机状态
                            cur_data_dict = cur_data_dict[:sampling_number]
                        for item in cur_data_dict:
                            # Associate each item with the corresponding image folder
                            item['folder'] = image_folder

                        rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                        self.list_data_dict.extend(cur_data_dict)

            else:
                with open(data_path, "r") as f:
                    data_dict = json.load(f)
                    # if "llava-onevision" in image_folder: data_dict=data_dict[::5]
                    for item in data_dict:
                        # Associate each item with the corresponding image folder
                        item['folder'] = image_folder
                    self.list_data_dict.extend(data_dict)  # Append data from each file to the list
        
        
        print("DATA SIZE:",len(self.list_data_dict))
        self.resolution_ratio=data_args.resolution_ratio
        print(f"****** Resolution_ratio = {self.resolution_ratio} ******")


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample:
                cur_len = cur_len
            elif 'video' in sample:
                cur_len = -cur_len
            else:
                cur_len = -1*cur_len
            # cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def split_image(self, image, n=2):
        if n==1: return [image]
        width, height = image.size
        block_width = width // n
        block_height = height // n

        blocks = []

        for i in range(n):
            for j in range(n):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height
                block = image.crop((left, upper, right, lower))
                blocks.append(block)
        blocks.append(image)

        return blocks

    def load_video_org(self,path, n_clips=1, num_frm=None):
        """
        Load video frames from a video file.

        Parameters:
        vis_path (str): Path to the video file.
        n_clips (int): Number of clips to extract from the video. Defaults to 1.
        num_frm (int): Number of frames to extract from each clip. Defaults to 100.

        Returns:
        list: List of PIL.Image.Image objects representing video frames.
        """

        # Load video with VideoReader
        vr = VideoReader(path, ctx=cpu(0))
        total_frame_num = len(vr)

        # Currently, this function supports only 1 clip
        assert n_clips == 1
        fps=vr.get_avg_fps()
        if num_frm==None :
            num_frm=int(total_frame_num//fps)
            num_frm=min(num_frm,8)


        # Calculate total number of frames to extract
        total_num_frm = min(total_frame_num, num_frm)
        # Get indices of frames to extract
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        # Extract frames as numpy array
        img_array = vr.get_batch(frame_idx).asnumpy()
        # Set target image height and width
        target_h, target_w = 336, 336   
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        # Reshape array to match number of clips and frames
        img_array = img_array.reshape(
            (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

        return clip_imgs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # text_only=False
        # if random.random()<0.5:
        #     text_only=True


        # if 'image' in sources[0]:
        #     try:
        #         image_file = self.list_data_dict[i]['image']
        #         image_folder = self.list_data_dict[i]['folder']
        #         processor = self.data_args.image_processor
        #         image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        #         if self.data_args.image_aspect_ratio == 'pad':
        #             def expand2square(pil_img, background_color):
        #                 width, height = pil_img.size
        #                 if width == height:
        #                     return pil_img
        #                 elif width > height:
        #                     result = Image.new(pil_img.mode, (width, width), background_color)
        #                     result.paste(pil_img, (0, (width - height) // 2))
        #                     return result
        #                 else:
        #                     result = Image.new(pil_img.mode, (height, height), background_color)
        #                     result.paste(pil_img, ((height - width) // 2, 0))
        #                     return result
        #             image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        #             # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #         # else:
        #             # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #         N=self.resolution_ratio
        #         image=self.split_image(image,n=N)
        #         image = processor.preprocess(image, return_tensors='pt')['pixel_values']
        #     except:
        #         print(self.list_data_dict[i])
        #         crop_size = self.data_args.image_processor.crop_size
        #         N=self.resolution_ratio
        #         if N==1:
        #             image = torch.zeros(1,3, crop_size['height'], crop_size['width'])
        #         else:
        #             image = torch.zeros(N**2+1,3, crop_size['height'], crop_size['width'])



        #     sources = preprocess_multimodal(
        #         copy.deepcopy([e["conversations"] for e in sources]),
        #         self.data_args)
        # elif 'video' in sources[0]:
        #     try:
        #         image_file = self.list_data_dict[i]['video']
        #         image_folder = self.list_data_dict[i]['folder']
        #         video_frames = self.load_video_org(os.path.join(image_folder, image_file),num_frm=8)
        #         temporal_len=len(video_frames)
        #         processor = self.data_args.image_processor
        #         N=self.resolution_ratio
        #         images=[]
        #         for video_frame in video_frames:
        #             images.extend(self.split_image(video_frame,n=N))

        #         images = processor.preprocess(images, return_tensors='pt')['pixel_values']
        #         N2_x_temporal,rgb,height,width=images.size()
        #         image=images.view(temporal_len,-1,rgb,height,width)

        #     except:
        #         print(self.list_data_dict[i])
        #         crop_size = self.data_args.image_processor.crop_size
        #         image = torch.zeros(8,1,3, crop_size['height'], crop_size['width'])

        #     sources = preprocess_multimodal_video(
        #         copy.deepcopy([e["conversations"] for e in sources]),
        #         self.data_args)
        # else:
        #     print("NO IMAGE")
        #     sources = copy.deepcopy([e["conversations"] for e in sources])
        valid=False
        data_dict=None
        while not valid:
            try:
            # if True:
                if 'image' in sources[0]:
                    # print(f"proccess\n{self.list_data_dict[i]}")
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.list_data_dict[i]['folder']
                    processor = self.data_args.image_processor
                    # if os.path.getsize(os.path.join(image_folder, image_file)) > 10*1024*1024:
                    #     assert False
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    # else:
                        # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    N=self.resolution_ratio
                    image=self.split_image(image,n=N)
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values']

                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                    # print(f"finish proccess")
                elif 'video' in sources[0]:
                    try:
                        image_file = self.list_data_dict[i]['video']
                        image_folder = self.list_data_dict[i]['folder']
                        video_frames = self.load_video_org(os.path.join(image_folder, image_file),num_frm=8)
                        temporal_len=len(video_frames)
                        processor = self.data_args.image_processor
                        N=self.resolution_ratio
                        images=[]
                        for video_frame in video_frames:
                            images.extend(self.split_image(video_frame,n=N))

                        images = processor.preprocess(images, return_tensors='pt')['pixel_values']
                        N2_x_temporal,rgb,height,width=images.size()
                        image=images.view(temporal_len,-1,rgb,height,width)
                    except:
                        
                        print(self.list_data_dict[i])
                        crop_size = self.data_args.image_processor.crop_size
                        image = torch.zeros(8,1,3, crop_size['height'], crop_size['width'])

        
                    sources = preprocess_multimodal_video(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                    
                else:
                    # print("NO IMAGE")
                    # print(sources[0])
                    # if isinstance(sources[0], list):
                    #     sources=[sources[0][0]]
                    # sources = copy.deepcopy([e["conversations"] for e in sources])
                    # sources = copy.deepcopy([e["conversations"] for e in sources])
                    sources = copy.deepcopy([self.list_data_dict[i]["conversations"]])
                    # assert False
                    # assert False
                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=('image' in self.list_data_dict[i]))
                valid=True
                
            except Exception as e:
                print(f"Error: {e}")
                print(self.list_data_dict[i])
                # assert False
                i=random.randint(0, len(self.list_data_dict))


        '''
        valid=False
        while not valid:
            try:
                if 'image' in sources[0]:
                    
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.list_data_dict[i]['folder']
                    processor = self.data_args.image_processor
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    # else:
                        # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    N=self.resolution_ratio
                    image=self.split_image(image,n=N)
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                    valid=True

                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                elif 'video' in sources[0]:
                    image_file = self.list_data_dict[i]['video']
                    image_folder = self.list_data_dict[i]['folder']
                    video_frames = self.load_video_org(os.path.join(image_folder, image_file),num_frm=8)
                    processor = self.data_args.image_processor



                    # processor = self.data_args.image_processor
                    # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    # if self.data_args.image_aspect_ratio == 'pad':
                    #     def expand2square(pil_img, background_color):
                    #         width, height = pil_img.size
                    #         if width == height:
                    #             return pil_img
                    #         elif width > height:
                    #             result = Image.new(pil_img.mode, (width, width), background_color)
                    #             result.paste(pil_img, (0, (width - height) // 2))
                    #             return result
                    #         else:
                    #             result = Image.new(pil_img.mode, (height, height), background_color)
                    #             result.paste(pil_img, ((height - width) // 2, 0))
                    #             return result
                    #     image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    #     # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    # # else:
                    #     # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    N=self.resolution_ratio
                    images=[]
                    for video_frame in video_frames:
                        images.extend(self.split_image(video_frame,n=N))

                    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
                    N2_x_temporal,rgb,height,width=images.size()
                    image=images.view(N2_x_temporal,-1,rgb,height,width)


                    
                    valid=True
                    

                    sources = preprocess_multimodal_video(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                else:
                    print(sources)
                    sources = copy.deepcopy([e["conversations"] for e in sources])

            except:
                print(self.list_data_dict[i])
                i=random.randint(0, len(self.list_data_dict))
        '''
        if data_dict is None:
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
            data_dict['image'] = image
        
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(1,3, crop_size['height'], crop_size['width'])
            N=self.resolution_ratio
            if N==1:
                image = torch.zeros(1,3, crop_size['height'], crop_size['width'])
            else:
                image = torch.zeros(N**2+1,3, crop_size['height'], crop_size['width'])
            data_dict['image'] = image
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_paths=data_args.data_paths,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaMiniLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.is_quantized=False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        # from copy import deepcopy
        # org_model=deepcopy(model)
        model = get_peft_model(model, lora_config)
        # model.model.base_model.prefusion_layers=org_model.model.base_model.prefusion_layers
        # del org_model

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if model_args.version=="llava_llama_3_1":
            # tokenizer.unk_token="<|reserved_special_token_5|>"
            tokenizer.pad_token_id = 128004
            tokenizer.unk_token=tokenizer.pad_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # print("conv:",conversation_lib.default_conversation )


    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

            for p in model.get_model().prefusion_layers.parameters():
                p.requires_grad = True

            for p in model.get_model().compressor.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.freeze_compressor:
            for p in model.get_model().prefusion_layers.parameters():
                p.requires_grad = False

            for p in model.get_model().compressor.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.compressor_size = model_args.compressor_size
        model.config.prefusion_layer_num = model_args.prefusion_layer_num
        model.config.resolution_ratio = data_args.resolution_ratio = model_args.resolution_ratio
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    save_param_info(model,training_args.output_dir)
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
