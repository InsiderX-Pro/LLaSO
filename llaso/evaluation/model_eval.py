import argparse
#import evaluate
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llaso.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN ,IGNORE_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_AUDIO_END_TOKEN,DEFAULT_AUDIO_START_TOKEN
from llaso.conversation import conv_templates, SeparatorStyle
from llaso.model.builder import load_pretrained_model

from llaso.train.train import DataArguments

from torch.utils.data import Dataset, DataLoader



import os
import sys
root_path = os.path.abspath("/code/LLaSO") 
sys.path.append(root_path) 
os.environ['CUDA_VISIBLE_DEVICES']='0'
 

from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Union
import io
import random
import librosa
import numpy as np
import soundfile
from tqdm import tqdm


import torch
import transformers
from torch.utils.data import Dataset
import llaso.conversation as conversation_lib

from llaso.mm_utils import tokenizer_image_audio_token

import zipfile
from PIL import Image

AUDIOSTART = "/data/"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def extract_image_from_zip(zip_path, image_to_extract):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(image_to_extract) as image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
    return image

import io
import base64
def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

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

  
def preprocess_va(
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
            if j == len(source)-1:
                conv.append_message(role, "")
            else:
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

    return dict(
        input_ids=input_ids,
        labels=None,
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
            # source[0]['value'] = DEFAULT_AUDIO_TOKEN+source[0]['value'].split('<audio>')[0]
            # conversation = source[0]['value'] 
            source[0]['value'] = source[0]['value'].replace(DEFAULT_AUDIO_TOKEN,"").strip()
            source[0]['value'] = DEFAULT_AUDIO_TOKEN + '\n' + source[0]['value']
            conversation = source[0]['value'] + source[1]['value'] + '</s>'
            conversation = conversation.strip()
            conversations.append(conversation)    
                
    # tokenize conversations
    input_ids = [tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

    return dict(input_ids=input_ids, labels=None)

     
def move_audio_to_user_prompt_start_mmtag_InstructionOrInputAudio(text,mm_use_audio_start_end):
    user_start = text.find("<|start_header_id|>user<|end_header_id|>")
    audio_start = text.find("<audio>")
    
    if user_start == -1 or audio_start == -1:
        return text  
    
    user_prompt = text[user_start + len("<|start_header_id|>user<|end_header_id|>"):audio_start].strip() #+ temp_sep # 提取 'prompt'
    
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
           
    
def preprocess_audio_v1(sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    
    def find_sequence_and_get_preceding(tensor, sequence):
        
        seq_len = len(sequence)
        
       
        for i in range(tensor.shape[1] - seq_len + 1):
            if torch.equal(tensor[:,i:i+seq_len], torch.tensor([sequence])):
                
                return tensor[:,:i+seq_len]
        
        
        print('didnt find match tesnor, now return an empty tensor')
        return torch.tensor([[]])
    
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
        #conversations.append(move_audio_to_user_prompt_start(conv.get_prompt()))

    
    
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
     
    sequence = [319, 1799, 9047, 13566, 29901]
    input_ids = find_sequence_and_get_preceding(input_ids, sequence)
    
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
                
        input_ids = input_ids[:,:instruction_len]  
        
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

         
        input_ids = input_ids[:,:instruction_len]  
        
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    mm_use_audio_start_end: bool = False
) -> Dict:
    # return preprocess_va(sources, tokenizer, has_image=has_image, has_audio=has_audio)
    #return preprocess_audio_v1(sources, tokenizer, has_image=has_image,has_audio=has_audio)
    #return preprocess_llama32_audio_v1(sources, tokenizer, has_image=has_image,has_audio=has_audio)
    return preprocess_llama32_audio_v1_mmtag(sources, tokenizer, has_image=has_image,has_audio=has_audio,mm_use_audio_start_end= mm_use_audio_start_end)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args=None,
                 mm_use_audio_start_end: bool = False
                 ):
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
         
        if 'voice' in sources[0]:
            audio_processor = self.data_args.audio_processor  
            audio_files = sources[0]["voice"]
            language = self.data_args.language
            # audio to be split chunk so set max audio chunk  
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
                # ------------------------ WavLM / HuBERT / Wav2Vec2  ------------------------
                if getattr(audio_processor, "feature_extractor_type", None) == 'Wav2Vec2FeatureExtractor' or audio_processor.__class__.__name__ == 'Wav2Vec2FeatureExtractor':
                    
                    chunk_threshold = 480000  
                    while len(tmp_sample) > 0:
                        if len(tmp_sample) > chunk_threshold:
                            chunk = tmp_sample[:chunk_threshold+1]
                            tmp_sample = tmp_sample[chunk_threshold+1:]
                            features = audio_processor(raw_speech=chunk,sampling_rate=target_sr)
                             
                            f = features["input_values"][0]  
                            features_list.append(f)
                            features_mask.append(j+1)
                        else:
                            features = audio_processor(raw_speech=tmp_sample,sampling_rate=target_sr)
                            f = features["input_values"][0]
                            features_list.append(f)
                            features_mask.append(j+1)
                            tmp_sample = []       
                # ------------------------ Whisper  ------------------------                                
                elif getattr(audio_processor, "feature_extractor_class", None) == 'WhisperFeatureExtractor':
                    # (1, 128, 3000)
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
                # ------------------------ CLAP  ------------------------
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
         
        # deal text
        text_len = 200
        #bos_token="<s>"
        #eos_token="</s>"            
        data_dict = preprocess(
            [sources[0]["conversations"]],
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            has_audio=('voice' in self.list_data_dict[i]),
            mm_use_audio_start_end = self.mm_use_audio_start_end
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             data_ori= sources[0]
                             )

        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        #     data_dict['has_image'] = True
        # elif self.data_args.is_multimodal or self.data_args.mix_va:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = self.data_args.image_processor.crop_size
        #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        #     data_dict['has_image'] = False
        if 'voice' in self.list_data_dict[i]:
            data_dict["input_features"] = features_list
            data_dict["features_mask"] = features_mask
        elif self.data_args.is_multimodal or self.data_args.mix_va:
            # audio does not exist in the data, but the model is multimodal
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
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = [ids if ids is not None else torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long) for ids in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        if labels[0] is not None:
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
            labels = labels[:, :self.tokenizer.model_max_length]
        else:
            labels = None 
        
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
            
            
            # -------------------- different processor branch --------------------
            if is_wav:
                # ----- WavLM or wav2vec2 or hubert-----
                for ins in instances:
                     
                    flist = [t for t in ins["input_features"]]
                     
                    pad_value = flist[0][0]
                     
                    while len(flist) < padlen:
                        flist.append(np.ones(flist[0].shape, dtype=flist[0].dtype) * pad_value)
                    pad_features.append(flist)
            elif is_whisper:
                # ----- Whisper -----
                 
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
                
            
            # -------------------- audio feature and attention_mask --------------------
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
                 
            #================= feature post process  =================
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
             
 
            # -------------------- mask post process--------------------
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
 
            # -------------------- mask index align -------------------- 
            batch_fmask = []
            for ins in instances:
                fmask = ins["features_mask"]
                while len(fmask) < padlen:
                    fmask.append(0)
                batch_fmask.append(torch.tensor(fmask))
            batch["features_mask"] = torch.stack(batch_fmask, dim=0)
                     
        batch["input_ids"] = input_ids 
        batch["labels"] = labels   
        batch["attention_mask"] =batch["input_ids"].ne(self.tokenizer.pad_token_id)
        batch["position_ids"] = torch.arange(batch["input_ids"].size(1), device=batch["input_ids"].device).unsqueeze(0).expand_as(batch["input_ids"])
        

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_mask'] = torch.tensor([instance['has_image'] for instance in instances])
        batch['data_ori'] = [instance['data_ori'] for instance in instances]
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, evaluate_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                mm_use_audio_start_end = evaluate_args.mm_use_audio_start_end)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,processor=data_args.audio_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


@dataclass
class args:
    data_path: str = None
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    pad_audio: bool = True
    mix_va: bool = True
    def __init__(self):
        self.language = "English"
        self.task = "transcribe"

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_tower", type=str, default=None) 
    parser.add_argument("--audio_tower", type=str, default="./whisper-large-v3")   # your local whisper-large-v3 path
    parser.add_argument("--model_base", type=str, default= None)  #set as None or the Base LLM such as the path of your local Llama-3.2-3B-Instruct
    parser.add_argument("--model_path", type=str, default="./LLaSO-Base") #LLaSO-Base ckpts path
    parser.add_argument("--version", type=str, default="llama32_audio_v1_mmtag") #keep it 
    parser.add_argument("--data_path", type=str, default="./LLaSO-Eval/xxx.json")  
    parser.add_argument("--data_type", type=str, default="asr")  #just keep it
    parser.add_argument("--output_dir", type=str, default="./your output dir")
    parser.add_argument("--mm_use_audio_start_end", type=bool, default=True) #just keep it

    evaluate_args = parser.parse_args()

    model_name = "llava"
    
    tokenizer, model, image_processor,audio_processor, context_len = load_pretrained_model(evaluate_args.model_path, evaluate_args.model_base,model_name)
    model.bfloat16()
    model.cuda()
    model.eval()

    data_args = args()
    data_args.data_path = evaluate_args.data_path
    #data_args.image_processor = image_processor
    data_args.audio_processor = audio_processor

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              evaluate_args=evaluate_args)

    # test data
    from torch.utils.data import DataLoader
    data = data_module["train_dataset"]
    collator = data_module["data_collator"]
    train_loader = DataLoader(data, batch_size = 1, collate_fn = collator, num_workers = 2)
    outlist = []
    print(data_args.data_path)
    
    
    if "vqa" in evaluate_args.data_type or "mmbench" in evaluate_args.data_type :
        kwargs = dict(do_sample = False,
                    num_beams = 1,
                    temperature = 0,
                    max_new_tokens=512)
    elif "asr" in evaluate_args.data_type or "librispeech" in evaluate_args.data_type :
        kwargs = dict(do_sample=False,
                num_beams = 1,
                temperature=0.2,
                max_new_tokens=256)
    else:
        kwargs = dict(do_sample=True,
                temperature=0.2,
                max_new_tokens=1024)
    
    for step, batch in tqdm(enumerate(train_loader)):
        
        if batch["input_ids"] is None:
            print("Warning: input_ids is None! Skipping this batch.")
            continue   
        
        output_ids = model.generate(
                **kwargs,
                inputs = batch["input_ids"].to(device=model.device),   
                attention_mask = batch["attention_mask"].to(device=model.device), 
                position_ids = batch["position_ids"].to(device=model.device),
                input_features = batch["input_features"].to(device=model.device).bfloat16(),  
                features_mask = batch["features_mask"].to(device=model.device),  
                use_cache = True,
                audio_attention_mask=batch.get("audio_attention_mask", None),   
            )
        
        
        outputs = tokenizer.batch_decode(output_ids[:,:], skip_special_tokens=True)[0] 
        outputs = outputs.strip()
        dic = batch["data_ori"][0]
        dic["answer"] = str(outputs)
        print(dic)
        outlist.append(dic)

    file_path =  evaluate_args.output_dir
    directory = os.path.dirname(file_path)
    
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, "w",encoding='utf-8') as f:
        json.dump(outlist, f, indent=4)


if __name__ == "__main__":
    eval()
