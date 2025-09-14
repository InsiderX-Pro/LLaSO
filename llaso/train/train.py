import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import numpy as np
from typing import Dict, Optional, Sequence, List, Any
import librosa
import soundfile
import torch

import transformers
import tokenizers

from llaso.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN
from torch.utils.data import Dataset
from llaso.train.llava_trainer import LLaVATrainer

from llaso import conversation as conversation_lib
from llaso.model import *
from llaso.mm_utils import tokenizer_image_token, tokenizer_image_audio_token

from PIL import Image

os.environ["WANDB_MODE"]="offline"

AUDIOSTART = "/data/"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None) 
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None) 
    audio_tower: Optional[str] = field(default=None) 
    mm_vision_select_layer: Optional[int] = field(default=-2)    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    audio_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)  
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
    mm_use_audio_start_end: bool = field(default=False)
    tune_mm_audio_aligner: bool = field(default=False)
    tune_mm_audio_projector: bool = field(default=False)
    mm_audio_select_layer: Optional[int] = field(default=-1)
    mm_audio_select_feature: Optional[str] = field(default="patch")
    pretrain_audio_aligner: Optional[str] = field(default=None)
    language: Optional[str] = field(default="English")
    task: Optional[str] = field(default="transcribe")
    local_files_only: Optional[str] = field(default=False)
    #query_tokens_size: Optional[int] = field(default=50)

     
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    pad_audio: bool = True
    mix_va: bool = False
    eval_data_type: str = field(default=None, metadata={"help": "Eval only. Data type of the dataset."})
    eval_output: Optional[str] = field(default=None, metadata={"help": "Eval only. Output path of the eval result."})
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    max_grad_norm:  Optional[float] = field(default=1.0)
    dataloader_drop_last: bool = field(default=True)
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
    bf16: bool = field(default=False)
     
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
    multimodal_keywords = ['audio_tower','mm_audio_aligner','mm_projector', 'vision_tower', 'vision_resampler'] 
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_all_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
     
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    # visual projector
        keys_to_match = ['mm_projector', 'vision_resampler']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        if trainer.args.local_rank == 0 :#or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
         
    # audio projector/aligner
    if getattr(trainer.args, "tune_mm_audio_projector", False):
        #keys_to_match = ['mm_audio_aligner',"query_tokens"]
        keys_to_match = ['mm_audio_aligner' ]
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = {(k[11:] if k.startswith('base_model.') else k): v for k, v in weight_to_save.items()}
        if any(k.startswith('model.model.') for k in weight_to_save):
            weight_to_save = {(k[6:] if k.startswith('model.') else k): v for k, v in weight_to_save.items()}

        if trainer.args.local_rank == 0 : #or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_audio_aligner.bin'))
         

     
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
        trainer._save(output_dir, state_dict=cpu_state_dict)



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

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
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

    if has_image or has_audio:
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer)) - 2
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


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
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
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image or has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer)) - 2
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


def move_audio_to_user_prompt_start(text):
     
    user_start = text.find("<|start_header_id|>user<|end_header_id|>")
    audio_start = text.find("<audio>")
    
    if user_start == -1 or audio_start == -1:
        return text   
    
     
    sys_prompt = text[:user_start].strip()  
    user_prompt = text[user_start + len("<|start_header_id|>user<|end_header_id|>"):audio_start].strip()  
    audio_tag = "<audio>"   
    assistant_part = text[audio_start + len("<audio>"):].strip()    
    
    
    result = f"{sys_prompt}<|start_header_id|>user<|end_header_id|>\n\n{audio_tag}\n{user_prompt}{assistant_part}"
    result.strip()
    return result
    
   
def move_audio_to_user_prompt_start_mmtag(text,mm_use_audio_start_end):
    user_start = text.find("<|start_header_id|>user<|end_header_id|>")
    audio_start = text.find("<audio>")
    
    if user_start == -1 or audio_start == -1:
        return text  
    
    sys_prompt = text[:user_start].strip() 
    user_prompt = text[user_start + len("<|start_header_id|>user<|end_header_id|>"):audio_start].strip()  
    audio_tag = "<audio>"   
    assistant_part = text[audio_start + len("<audio>"):].strip()    
    
    
    if mm_use_audio_start_end:
        # result = f"{sys_prompt}<|start_header_id|>user<|end_header_id|>\n\n{audio_tag}\n{user_prompt}{assistant_part}" 
        result = f"{sys_prompt}<|start_header_id|>user<|end_header_id|>\n\n<Audio>{DEFAULT_AUDIO_START_TOKEN}{audio_tag}{DEFAULT_AUDIO_END_TOKEN}</Audio>\n{user_prompt}{assistant_part}"
    else:
        result = f"{sys_prompt}<|start_header_id|>user<|end_header_id|>\n\n<Audio>{audio_tag}</Audio>\n{user_prompt}{assistant_part}"
    result.strip()
    return result
         
def move_audio_to_user_prompt_start_mmtag_InstructionOrInputAudio(text,mm_use_audio_start_end):
    user_start = text.find("<|start_header_id|>user<|end_header_id|>")
    audio_start = text.find("<audio>")
    
    if user_start == -1 or audio_start == -1:
        return text   
    
    #sys_prompt = text[:user_start].strip() #+ temp_sep   
    user_prompt = text[user_start + len("<|start_header_id|>user<|end_header_id|>"):audio_start].strip()  
     
    if user_prompt != '':
        if mm_use_audio_start_end:
            result = text.replace('<audio>',f'\n<Audio>{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}</Audio>\n\n')
        else:
            result = text.replace('<audio>',f'\n<Audio><audio></Audio>\n\n')
    elif user_prompt == '' :        
        if mm_use_audio_start_end:
            result = text.replace('<audio>',f'<Audio>{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}</Audio>\n')
        else:
            result = text.replace('<audio>',f'<Audio><audio></Audio>\n')
    else :
        print('The instruction may wrong. Printing..')
        print(text)
            
    result.strip()
    return result    
    
 
def preprocess_audio_v1(sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
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
            
         
        #conversations.append(conv.get_prompt())
        conversations.append(move_audio_to_user_prompt_start(conv.get_prompt()))

    
    
    # Tokenize conversations
    if has_image or has_audio:
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX  
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou = rou + conv.sep2 
            
            parts = rou.split(sep) 
            if len(parts) != 2:
                break
            parts[0] += sep 

            if has_image or has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer)) 
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer))    
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += (round_len) 
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
    
def preprocess_llama32_audio_v1(sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        assert len(source) == 2, "only support single turn conversation at present"
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]] 
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"]) 
            
        #conversations.append(conv.get_prompt())
        conversations.append(move_audio_to_user_prompt_start(conv.get_prompt()))

    
    
    # Tokenize conversations
    if has_image or has_audio:
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

     # Mask targets 
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) 

        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep)
        parts[0] += sep
        
        
        if has_image or has_audio:
            conversation_len = len(tokenizer_image_audio_token(conversation, tokenizer))  
            instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer))     
        else:
            conversation_len = len(tokenizer(conversation).input_ids) 
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += conversation_len
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

def preprocess_llama32_audio_v1_mmtag(sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    mm_use_audio_start_end: bool = False
) -> Dict:
    
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates  
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        assert len(source) == 2, "only support single turn conversation at present"
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]] 
            assert role == conv.roles[j % 2], f"{i}"  
            conv.append_message(role, sentence["value"]) 
            
        #conv.get_prompt()
        #conversations.append(conv.get_prompt())
        #conversations.append(move_audio_to_user_prompt_start_mmtag(text = conv.get_prompt(),mm_use_audio_start_end = mm_use_audio_start_end))
        conversations.append(move_audio_to_user_prompt_start_mmtag_InstructionOrInputAudio(text = conv.get_prompt(),mm_use_audio_start_end = mm_use_audio_start_end))
    
    
    # Tokenize conversations

    if has_image or has_audio: 
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

     # Mask targets  
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) 

        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep)
        parts[0] += sep
        
        
        if has_image or has_audio:
            conversation_len = len(tokenizer_image_audio_token(conversation, tokenizer))  
            instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer))     
        else:
            conversation_len = len(tokenizer(conversation).input_ids) 
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += conversation_len
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
        assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_AUDIO_TOKEN in source[0]['value']
        if DEFAULT_IMAGE_TOKEN in source[0]['value']:
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
            conversations.append(conversation)
        elif DEFAULT_AUDIO_TOKEN in source[0]['value']:
            
            source[0]['value'] = source[0]['value'].replace(DEFAULT_AUDIO_TOKEN,"").strip()
            source[0]['value'] = DEFAULT_AUDIO_TOKEN + '\n' + source[0]['value']
            conversation = source[0]['value'] + source[1]['value'] + '</s>'
            conversation = conversation.strip()
            
            conversations.append(conversation)    
                
    # tokenize conversations
    input_ids = [tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_audio_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    mm_use_audio_start_end: bool = False
) -> Dict:
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version.startswith("audio_v1"):
        return preprocess_audio_v1(sources, tokenizer, has_image=has_image,has_audio=has_audio)
    
    # if conversation_lib.default_conversation.version.startswith("llama32_audio_v1"):
    #     return preprocess_llama32_audio_v1(sources, tokenizer, has_image=has_image,has_audio=has_audio)
    
    if conversation_lib.default_conversation.version.startswith("llama32_audio_v1_mmtag"):
        return preprocess_llama32_audio_v1_mmtag(sources, tokenizer, has_image=has_image,has_audio=has_audio,mm_use_audio_start_end= mm_use_audio_start_end)
    
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image,has_audio=has_audio)
    
# =============================================================================
# dataset and Collator
# =============================================================================
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, mm_use_audio_start_end: bool = False):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.data_path = data_path
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.mono = True
        self.sample_rate = 16000
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        self.mm_use_audio_start_end = mm_use_audio_start_end

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths_type(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value']) for conv in sample['conversations'])
            input = sample['conversations'][0]['value']
            cur_type = 0
            if "<audio>" in input:
                cur_type += 2
            if "<image>" in input:
                cur_type += 1
            length_list.append((cur_len, cur_type))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # ============ img process ============ 
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_processor = self.data_args.image_processor
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
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # ============ audio process ============ 
        # ----------------------------------------------------
        if 'voice' in sources[0]:
            audio_processor = self.data_args.audio_processor  
            audio_files = sources[0]["voice"]
            language = self.data_args.language
            # to split audio by chunk, so set max chunk num.
            max_chunks = 20  
            features_list = []
            features_mask = []
            if isinstance(audio_files, str):
                audio_files = [audio_files]  
            
            target_sr = self._decide_sample_rate(audio_processor)
            
            for j, audio_file in enumerate(audio_files):
                try:
                    sample, sample_rate = soundfile.read(audio_file, dtype='float32')
                except Exception as e:
                    print(f"Failed to load audio file {audio_file}: {e}")
                    continue
                
                sample = sample.T 
                if self.mono:
                    sample = librosa.to_mono(sample)
                if target_sr != sample_rate :
                    sample = self.resample(sample, orig_sr=sample_rate, target_sr=target_sr) 
                        
                
                tmp_sample = sample.copy()
                
                start_len = len(features_list)
                # different audio processor branch 
                # ------------------------ WavLM / HuBERT / Wav2Vec2 ------------------------
                if getattr(audio_processor, "feature_extractor_type", None) == 'Wav2Vec2FeatureExtractor' or audio_processor.__class__.__name__ == 'Wav2Vec2FeatureExtractor':
                    chunk_threshold = 480000  
                    while len(tmp_sample) > 0:
                        if len(tmp_sample) > chunk_threshold:
                            chunk = tmp_sample[:chunk_threshold+1]
                            tmp_sample = tmp_sample[chunk_threshold+1:]
                            features = audio_processor(raw_speech=chunk,sampling_rate=target_sr)
                            f = features["input_values"][0] #f.shape(480001,)
                            features_list.append(f)
                            features_mask.append(j+1)
                        else:
                            features = audio_processor(raw_speech=tmp_sample,sampling_rate=target_sr)
                            f = features["input_values"][0]
                            features_list.append(f)
                            features_mask.append(j+1)
                            tmp_sample = []       
                # ------------------------ Whisper ------------------------                                
                elif getattr(audio_processor, "feature_extractor_class", None) == 'WhisperFeatureExtractor':
                    audio_processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.data_args.language)
                    while len(tmp_sample) > 0:
                        if len(tmp_sample) > 480000:
                            chunk = tmp_sample[:480001]
                            tmp_sample = tmp_sample[480001:]
                            features = audio_processor(audio=chunk, sampling_rate=target_sr).input_features
                            features_list.append(features)
                            features_mask.append(j+1)
                        else:
                            data = audio_processor(audio=tmp_sample, sampling_rate=target_sr)
                            features_list.append(data["input_features"])
                            features_mask.append(j+1)
                            tmp_sample = []
                # ------------------------ CLAP ------------------------
                elif getattr(audio_processor, "feature_extractor_class", None) == 'ClapFeatureExtractor':
                    
                    chunk_threshold = 1440000 
                    while len(tmp_sample) > 0:
                        if len(tmp_sample) > chunk_threshold:
                            chunk = tmp_sample[:chunk_threshold+1]
                            tmp_sample = tmp_sample[chunk_threshold+1:]
                        else:
                            chunk = tmp_sample
                            tmp_sample = []

                        features = audio_processor(audios=chunk, sampling_rate=target_sr, return_tensors="pt") 
                        f = features["input_features"][0].squeeze(0) #f.shape(1001, 64)
                        features_list.append(f)
                        features_mask.append(j+1)
                else:
                    print(f"[Dataset] unrecognized audio processor: {audio_processor.__class__.__name__}")
                if len(features_list) - start_len > max_chunks:
                    features_list = features_list[:start_len + max_chunks]
                    features_mask = features_mask[:start_len + max_chunks]
        # ---------------------- ----------------------
        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=('image' in self.list_data_dict[i]), has_audio=('voice' in self.list_data_dict[i]), mm_use_audio_start_end=self.mm_use_audio_start_end)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        if 'voice' in self.list_data_dict[i]:
            data_dict["input_features"] = features_list  
            data_dict["features_mask"] = features_mask
        elif self.data_args.is_multimodal or self.data_args.mix_va:
            data_dict["input_features"] = [[np.ones((80, 3000))]]
            data_dict["features_mask"] = [0]
        return data_dict

    @staticmethod
    def resample(sample, orig_sr, target_sr):
        return librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
 
    def _decide_sample_rate(self, audio_processor):
        processor_class_name = audio_processor.__class__.__name__
        feature_extractor_class = getattr(audio_processor, "feature_extractor_class", "")
        
        feature_extractor_name = ""
        if hasattr(audio_processor, "feature_extractor"):
            feature_extractor_name = audio_processor.feature_extractor.__class__.__name__
        
        if (
            feature_extractor_class == "ClapFeatureExtractor"
            or processor_class_name == "ClapProcessor"
            or feature_extractor_name == "ClapFeatureExtractor"
        ):
            return 48000
        else:
            return self.sample_rate
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    processor: Any  

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # ----------------- text input ids Padding -----------------
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
        
        batch = {}
        
        # ----------------- audio feature process -----------------
        if "input_features" in instances[0]:
            input_features = []
            padlen = max([len(ins["input_features"]) for ins in instances])
            pad_features = []
            audio_attn_masks = []
            
            is_wav = (hasattr(self.processor, "feature_extractor_type") and 
            self.processor.feature_extractor_type == "Wav2Vec2FeatureExtractor") \
                or self.processor.__class__.__name__ == 'Wav2Vec2FeatureExtractor' \
                or (hasattr(self.processor, "feature_extractor_class") and self.processor.feature_extractor_class == "Wav2Vec2FeatureExtractor")
            
            is_whisper = ((hasattr(self.processor, "feature_extractor_type") and self.processor.feature_extractor_type == "WhisperFeatureExtractor")
                or self.processor.__class__.__name__ == "WhisperFeatureExtractor") or (hasattr(self.processor, "feature_extractor_class") and self.processor.feature_extractor_class == "WhisperFeatureExtractor")
            
            is_clap = ((hasattr(self.processor, "feature_extractor_class") and self.processor.feature_extractor_class == "ClapFeatureExtractor")
                or self.processor.__class__.__name__ == "ClapFeatureExtractor")
            
            
            # -------------------- audio processor branch --------------------
            if is_wav:
                
                for ins in instances:
                    flist = [t for t in ins["input_features"]]
                    pad_value = flist[0][0]
                    while len(flist) < padlen:
                        flist.append(np.ones(flist[0].shape, dtype=flist[0].dtype) * pad_value)
                    pad_features.append(flist)
            elif is_whisper:
                for ins in instances:
                    flist = [t[0] for t in ins["input_features"]]
                    pad_value = flist[0][0, 0]
                    while len(flist) < padlen:
                        flist.append(np.ones(flist[0].shape, dtype=flist[0].dtype) * pad_value)
                    pad_features.append(flist)
            elif is_clap:
                for ins in instances:
                    flist = [t for t in ins["input_features"]]  

                    #pad_value = flist[0].flatten()[0] 
                    # while len(flist) < padlen:
                    #     flist.append(
                    #         np.ones(flist[0].shape, dtype=flist[0].dtype) * pad_value
                    #     )

                    pad_value = float(flist[0].flatten()[0].item())
                    np_dtype = flist[0].detach().cpu().numpy().dtype 
                    while len(flist) < padlen:
                        pad_block = np.ones(flist[0].shape, dtype=np_dtype) * pad_value  # shape: [H, W]
                        flist.append(pad_block)
                        
                    pad_features.append(flist)
                
            else:
                print(f"[DataCollator] unrecognized audio processor: {self.processor.__class__.__name__}")
                
            
            # -------------------- features and attention_mask --------------------
            for i in range(padlen):
                if is_whisper:
                    i_features = [{"input_features": feature[i]} for feature in pad_features]
                    batch_features = self.processor.feature_extractor.pad(i_features, return_tensors="pt")
                    input_features.append(batch_features["input_features"].unsqueeze(dim=0).clone())
                    audio_attn_masks.append(None)  
                elif is_wav:
                    i_features = [{"input_values": feature[i]} for feature in pad_features]
                    batch_features = self.processor.pad(i_features, return_tensors="pt")
                    input_features.append(batch_features["input_values"].unsqueeze(dim=0).clone())
                    audio_attn_masks.append(batch_features["attention_mask"].unsqueeze(dim=0).clone() if "attention_mask" in batch_features else None)
                elif is_clap:
                    i_features = [{"input_features": feature[i]} for feature in pad_features]
                    batch_features = self.processor.feature_extractor.pad(i_features, return_tensors="pt")
                    input_features.append(batch_features["input_features"].unsqueeze(dim=0).clone())
                    audio_attn_masks.append(None)    
                else:
                    print(f"[DataCollator] unrecognized audio processor: {self.processor.__class__.__name__}")
                 
                 
 
            #================= feature post process =================
            max_feat_len = max(t.shape[-1] for t in input_features) 
            padded_input_features = []
            for t in input_features:
                if t.shape[-1] < max_feat_len:
                    pad_len = max_feat_len - t.shape[-1]
                    pad_tensor = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad_tensor], dim=-1)
                padded_input_features.append(t)

            input_feature = torch.cat(padded_input_features, dim=0).transpose(0, 1)
            batch["input_features"] = input_feature
             
 
            # -------------------- mask post process --------------------
            if any(m is not None for m in audio_attn_masks):
                padded_audio_masks = []
                for i, mask in enumerate(audio_attn_masks):
                    if mask is None:
                        full_mask = torch.ones_like(input_features[i], dtype=torch.long)
                    else:
                        cur_len = mask.shape[-1]
                        if cur_len < max_feat_len:
                            pad_len = max_feat_len - cur_len
                            pad_tensor = torch.zeros(*mask.shape[:-1], pad_len, dtype=mask.dtype, device=mask.device)
                            full_mask = torch.cat([mask, pad_tensor], dim=-1)
                        else:
                            full_mask = mask
                    padded_audio_masks.append(full_mask)
                audio_attn_mask_cat = torch.cat(padded_audio_masks, dim=0).transpose(0, 1)
                batch["audio_attention_mask"] = audio_attn_mask_cat
 
            # -------------------- mask index alignment -------------------- 
            batch_fmask = []
            for ins in instances:
                fmask = ins["features_mask"]
                while len(fmask) < padlen:
                    fmask.append(0)
                batch_fmask.append(torch.tensor(fmask))
            batch["features_mask"] = torch.stack(batch_fmask, dim=0)
                     

        # ----------------- text input ids into batch -----------------
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] =input_ids.ne(self.tokenizer.pad_token_id)
        
        # ----------------- img (if used) -----------------
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_mask'] = torch.tensor([instance['has_image'] for instance in instances])

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                mm_use_audio_start_end = model_args.mm_use_audio_start_end)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,processor=data_args.audio_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    if training_args.bf16:
        target_dtype = torch.bfloat16
    elif training_args.fp16:
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32  
    print(f"compute_dtype: {target_dtype}")

    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    
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
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=target_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
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
        
    # model = transformers.LlamaForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         attn_implementation=attn_implementation,
    #         torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    #         **bnb_model_from_pretrained_args
    # )
    model.config.use_cache = False
    
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
        model = get_peft_model(model, lora_config)

    #load tokenizer
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

    #to set padding token
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )  
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version in ["llama32_audio_v1", "llama32_audio_v1_mmtag"]:
        
        special_tokens_dict = {}
        if tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = '<unk>'
            special_tokens_dict['unk_token'] = '<unk>'
        if 'mmtag' in model_args.version:
            # Add tokens <Audio> and </Audio> to tokenizer and embedding
            special_tokens_dict.setdefault('additional_special_tokens', [])  
            special_tokens_dict['additional_special_tokens'].extend(['<Audio>', '</Audio>'])
            
        if model_args.mm_use_audio_start_end:
            # Add tokens <audio_start> and <audio_end> to tokenizer and embedding
            special_tokens_dict.setdefault('additional_special_tokens', [])
            special_tokens_dict['additional_special_tokens'].extend([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
        if special_tokens_dict:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=model
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

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

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=target_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
    if model_args.audio_tower is not None:
        
        if training_args.bf16:
            target_dtype = torch.bfloat16
        elif training_args.fp16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        print(f"target_dtype for audio_tower: {target_dtype}")
        
        model.get_model().initialize_audio_modules(model_args=model_args, fsdp=training_args.fsdp, target_dtype=target_dtype)
        audio_tower = model.get_audio_tower()
        # keep target_dtype consistency
        audio_tower.to(dtype=(torch.bfloat16 if training_args.bf16 else torch.float16), device=training_args.device)
        data_args.audio_processor = audio_tower.audio_processor
        data_args.language = model_args.language
        data_args.is_multimodal = True
        if model_args.tune_mm_audio_aligner:
            model.requires_grad_(False)
            for p in model.get_model().mm_audio_aligner.parameters():
                p.requires_grad = True
            #model.get_model().query_tokens.requires_grad = True
        training_args.tune_mm_audio_aligner = model_args.tune_mm_audio_aligner
        training_args.tune_mm_audio_projector=model_args.tune_mm_audio_projector
        if model_args.tune_mm_audio_projector:
            model.requires_grad_(False)
            for p in model.get_model().mm_audio_aligner.projector.parameters():
                p.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_audio_aligner.to(dtype=target_dtype, device=training_args.device)

    
    if model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    if model_args.tune_mm_audio_projector:
        for p in model.get_model().mm_audio_aligner.projector.parameters():
            p.requires_grad = True
            
    
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
                                              data_args=data_args,model_args=model_args)
    from transformers import TrainerCallback

    
    
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
        if training_args.local_rank == 0 :#or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_all_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
