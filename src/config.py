import torch
from dataclasses import dataclass

@dataclass
class vit_config:
    num_channels: int = 3
    batch_size:int = 32
    image_size: int = 224
    patch_size: int = 16
    num_heads:int = 8
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    num_encoder_layers: int = 12
    random_seed: int = 42
    epochs: int = 30
    num_classes: int = 10          
    learning_rate: float = 1e-5
    adam_weight_decay: int = 0
    adam_betas: tuple = (0.9, 0.999)
    embd_dim: int = (patch_size ** 2) * num_channels           # 768
    num_patches: int = (image_size // patch_size) ** 2         # 196
    device: str = "cuda" if torch.cuda.is_available() else "cpu"