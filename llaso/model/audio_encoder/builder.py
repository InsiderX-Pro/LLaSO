
import os
from .whisper_encoder import WhisperAudioTower
from .wavlm_encoder import WavLMAudioTower
from .hubert_encoder import HubertAudioTower
from .wav2vec2_encoder import Wav2Vec2AudioTower
from .clap_encoder import ClapAudioTower
def build_audio_tower(audio_tower_cfg, **kwargs):
    
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    
    if is_absolute_path_exists and 'whisper' in audio_tower.lower():
        return WhisperAudioTower(audio_tower=audio_tower, args=audio_tower_cfg, **kwargs)
    elif is_absolute_path_exists and 'wavlm' in audio_tower.lower():
        return WavLMAudioTower(audio_tower=audio_tower, args=audio_tower_cfg, **kwargs)
    elif is_absolute_path_exists and 'hubert' in audio_tower.lower():
        return HubertAudioTower(audio_tower=audio_tower, args=audio_tower_cfg, **kwargs)
    elif is_absolute_path_exists and 'wav2vec2' in audio_tower.lower():
        return Wav2Vec2AudioTower(audio_tower=audio_tower, args=audio_tower_cfg, **kwargs)
    elif is_absolute_path_exists and 'clap' in audio_tower.lower():
        return ClapAudioTower(audio_tower=audio_tower, args=audio_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown audio tower: {audio_tower}')
