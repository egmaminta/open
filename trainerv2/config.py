from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class WandBConfig:
    wandb_project: str
    wandb_entity: str
    wandb_name: str
    log_every_n_steps: int = 100

@dataclass
class OptimizerConfig:
    optim_type: str = 'adamw'
    lr: float = 3e-4
    adamw_weight_decay: float = 0.1
    adamw_betas: Tuple[float] = field(default_factory=lambda: (0.9, 0.999))
    adamw_eps: float = 1e-8
    adamw_fused: bool = True
    warmup_iters: int = 512
    cosine_scheduler_cycles: float = 0.5

@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    enable_wandb_logging: bool = True
    out_dir: str = 'output'
    device: str = 'cuda'
    num_epochs: int = 2
    max_seq_length: int = 4000
    padding_side: Literal['left', 'right'] = 'right'
    optim_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb_config: WandBConfig = field(default_factory=WandBConfig)
    eval_iters: int = 100