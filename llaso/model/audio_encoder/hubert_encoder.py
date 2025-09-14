import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
from typing import Optional

class HubertAudioTower(nn.Module):
    def __init__(self, audio_tower: str, args, delay_load: bool = False):
        super().__init__()
        config = HubertConfig.from_pretrained(audio_tower)
        self.is_loaded = False
        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.audio_split_type_dim = 2
        self.language = getattr(args, 'language', None)
        self.task = getattr(args, 'task', None)
        self.local_files_only = getattr(args, 'local_files_only', False)
         
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = HubertConfig.from_pretrained(self.audio_tower_name)

    def load_model(self, target_dtype: Optional[torch.dtype] = None):
        if self.is_loaded:
            print(f'{self.audio_tower_name} already loaded. now skip loading')
            return
        print("loading Hubert path/name:", self.audio_tower_name)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.audio_tower_name,
            local_files_only=self.local_files_only
        )
 
        self.audio_tower = HubertModel.from_pretrained(
            self.audio_tower_name,
            local_files_only=self.local_files_only
        )
         
        self.audio_tower.requires_grad_(False)
        self.is_loaded = True
        
    def feature_select(self, audio_forward_outs):

        if hasattr(audio_forward_outs, "hidden_states"):
            audio_features = audio_forward_outs.hidden_states[self.select_layer]
            #if (audio_forward_outs.hidden_states[self.select_layer] == audio_forward_outs.last_hidden_state).all():
                #print("audio_forward_outs.hidden_states[self.select_layer] == audio_forward_outs.last_hidden_state")
        else:
            raise NotImplementedError("hidden_states is not enabled, so it is not possible to extract the specified layer output based on mm_audio_select_layer.")
        return audio_features
    
    @torch.no_grad()
    def forward(self, audios: torch.Tensor,attention_mask=None) -> torch.Tensor:
        # Hubert expect 2D tensor [batch_size, sequence_length]
        if audios.ndim != 2:
            raise ValueError(f"HubertAudioTower expect 2D tensor, now dim is {audios.ndim}")
        
        if self.audio_processor.return_attention_mask:
            audio_forward_out = self.audio_tower(
                audios.to(device=self.device, dtype=self.dtype),
                attention_mask=(
                    attention_mask.to(device=self.device)
                    if attention_mask is not None else None
                ),
                output_hidden_states=True,
                return_dict=True
            )
        else:
            audio_forward_out = self.audio_tower(
                audios.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True)
            
        # select state from the returned hidden_states 
        audio_features = self.feature_select(audio_forward_out).to(audios.dtype)
        return audio_features
    

    @property
    def dummy_feature(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.audio_tower.dtype

    @property
    def device(self) -> torch.device:
        return next(self.audio_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size
