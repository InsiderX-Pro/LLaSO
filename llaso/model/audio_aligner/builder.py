import os
import torch
import torch.nn as nn
import re

class AudioAligner(nn.Module):
    
    def __init__(self, audio_tower, config, **kwargs):
        super().__init__()
        
        projector_type = getattr(config, 'audio_projector_type', 'linear')
        if projector_type == 'linear':
            projector = nn.Linear(config.mm_audio_hidden_size, config.hidden_size)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = []
                modules.append(nn.Linear(config.mm_audio_hidden_size, config.hidden_size))
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                projector = nn.Sequential(*modules)
            else:
                raise ValueError(f'Unsupported audio_projector_type: {projector_type}')
        self.projector = projector

    def forward(self, encoder_output,  **kwargs):
        
        projector_output = self.projector(encoder_output)
        return projector_output

def build_audio_aligner(config, **kwargs):
    audio_tower = getattr(config, 'mm_audio_tower', getattr(config, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists and 'whisper' in audio_tower.lower():
        return AudioAligner(audio_tower=audio_tower, config=config, **kwargs)
    elif is_absolute_path_exists and 'wavlm' in audio_tower.lower():
        return AudioAligner(audio_tower=audio_tower, config=config, **kwargs)
    elif is_absolute_path_exists and 'hubert' in audio_tower.lower():
        return AudioAligner(audio_tower=audio_tower, config=config, **kwargs)
    elif is_absolute_path_exists and 'wav2vec2' in audio_tower.lower():
        return AudioAligner(audio_tower=audio_tower, config=config, **kwargs)
    elif is_absolute_path_exists and 'clap' in audio_tower.lower():
        return AudioAligner(audio_tower=audio_tower, config=config, **kwargs)
    else:
        raise ValueError(f'Unknown audio tower: {audio_tower}')
