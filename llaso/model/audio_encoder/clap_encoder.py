import torch
import torch.nn as nn
from typing import Optional
from transformers import  ClapModel, ClapProcessor, ClapConfig


class ClapAudioTower(nn.Module):

    def __init__(self, audio_tower: str, args, delay_load: bool = False):
        
        super().__init__()
        #config = ClapAudioConfig.from_pretrained(audio_tower)
        config = ClapConfig.from_pretrained(audio_tower)
        self.is_loaded = False
        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.language = getattr(args, 'language', None)
        self.task = getattr(args, 'task', None)
        self.local_files_only = getattr(args, 'local_files_only', False)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ClapConfig.from_pretrained(self.audio_tower_name)

    def load_model(self, target_dtype: Optional[torch.dtype] = None):
        
        if self.is_loaded:
            print(f'{self.audio_tower_name} already loaded. now skip loading')
            return
        print("loading Clap path/name:", self.audio_tower_name)

        # load Clap audio processor，
        self.audio_processor = ClapProcessor.from_pretrained(
            self.audio_tower_name,
            local_files_only=self.local_files_only
        )

        # load ClapAudioModel
        self.audio_tower =  ClapModel.from_pretrained(
            self.audio_tower_name,
            local_files_only=self.local_files_only
        ).audio_model

        self.audio_tower.requires_grad_(False)
        self.is_loaded = True


        
    def feature_select(self, clap_forward_out):
        if hasattr(clap_forward_out, "hidden_states") and clap_forward_out.hidden_states is not None:
            audio_features = clap_forward_out.hidden_states[self.select_layer]       
        else:
            raise NotImplementedError(
                    "hidden_states is not enabled, so it is not possible to extract the specified layer output based on mm_audio_select_layer."
            )

        
        B, hidden_dim, height, width = audio_features.shape  # for example [1, 1024, 8, 8]
        audio_features = audio_features.permute(0, 2, 3, 1).contiguous()
        seq_len = height*width 
        audio_features = audio_features.view(B, seq_len, hidden_dim)

        return audio_features

    @torch.no_grad()
    def forward(self, audios: torch.Tensor, attention_mask=None) -> torch.Tensor:
        
        if audios.ndim != 4:
            raise ValueError(f"ClapAudioTower receives 4D dims [B, C, H, W], now dim is {audios.ndim}")

        
        if getattr(self.audio_processor.feature_extractor, "return_attention_mask", False) or getattr(self.audio_processor, "return_attention_mask", False):
            audio_forward_out =  self.audio_tower(
                input_features = audios.to(device=self.device, dtype=self.dtype),
                attention_mask=(
                    attention_mask.to(device=self.device)
                    if attention_mask is not None else None
                ),
                output_hidden_states=True,
                return_dict=True
            )
        else:
            audio_forward_out =  self.audio_tower(
                input_features = audios.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True)
            
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
