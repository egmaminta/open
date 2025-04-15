from abc import ABC, abstractmethod
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import os
import loguru
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from typing import Dict

from .optimizers import Muon


@dataclass
class WandbConfig:
    project: str
    entity: str
    name: str


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    num_warmup_steps: int = 64
    num_cycles: int = 1.0


@dataclass
class TrainerConfig:
    batch_size: int
    gradient_accumulation_steps: int
    grad_clip: float
    optimizer_config: OptimizerConfig
    enable_wandb_logging: bool


class Trainer(ABC):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        device: str,
        config: TrainerConfig,
        wandb_config: WandbConfig
    ) -> None:
        self.policy_model = policy_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.wandb_config = wandb_config
        
        self.logger = loguru.logger
        
        if hasattr(self.processor, 'image_processor'):
            self.logger.info('Setting image processor size to 512x512...')
            self.processor.image_processor.size = {'height': 512, 'width': 512}

        self.logger.info('Initializing Trainer...')
        self.logger.info(f'Using policy model: {self.policy_model.config._name_or_path}')

    def add_tokens_for_loc_and_seg_tasks(self, precision: float=0.001):
        float_numbers = np.arange(0., 1, precision)
        tokens_for_loc_and_seg_tasks = [f"{num:.{len(str(precision).split('.')[-1])}f}" for num in float_numbers]
        tokens_for_loc_and_seg_tasks = ["<seg" + num.split('.')[-1] + ">" for num in tokens_for_loc_and_seg_tasks]
        tokens_for_loc_and_seg_tasks += ['<seg1000>']
        tokens_for_loc_and_seg_tasks += ['<task:segmentation>', '<task:localization>', '<seg_r>', '</seg_r>']
        self.logger.info(f'Adding {tokens_for_loc_and_seg_tasks} tokens for localization and segmentation tasks.')
        self.logger.info(f'Adding {len(tokens_for_loc_and_seg_tasks)} tokens for localization and segmentation tasks.')
        self.logger.info(f'Previous tokenizer size: {len(self.processor.tokenizer):,}')
        self.logger.info(f'Previous Embedding size: {self.policy_model.get_input_embeddings().weight.shape}')
        self.logger.info(f'Previous LM head size: {self.policy_model.lm_head.weight.shape}')
        
        self.processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.policy_model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)

        self.logger.info(f'New tokenizer size: {len(self.processor.tokenizer):,} ({len(tokens_for_loc_and_seg_tasks):,} new tokens)')
        self.logger.info(f'New Embedding size: {self.policy_model.get_input_embeddings().weight.shape}')
        self.logger.info(f'New LM head size: {self.policy_model.lm_head.weight.shape}')

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.logger.info(f'Using optimizer: {self.optimizer.__class__.__name__}')

    @staticmethod
    def prepare_optimizer(model, optim_config: OptimizerConfig, type: str="adamw"):
        learning_rate = optim_config.lr
        weight_decay = optim_config.weight_decay
        betas = optim_config.betas
        eps = optim_config.eps

        if type == 'adamw':
            param_dict = {pn: p for pn, p in model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            loguru.logger.info(f'Using AdamW optimizer with {num_decay_params:,} decay parameters and {num_nodecay_params:,} no-decay parameters.')
            return torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=True
            )

    # @staticmethod
    # def prepare_optimizers(model, optim_config: Dict, type: str="adamw"):
    #     learning_rate = optim_config.get('learning_rate', 2e-5)
    #     weight_decay = optim_config.get('weight_decay', 0.1)
    #     momentum = optim_config.get('momentum', 0.95)
    #     nesterov = optim_config.get('nesterov', True)
    #     ns_steps = optim_config.get('ns_steps', 5)
    #     adamw_betas = optim_config.get('adamw_betas', (0.9, 0.95))
    #     adamw_eps = optim_config.get('adamw_eps', 1e-8)

    #     if type == 'adamw':
    #         loguru.logger.info('Using AdamW optimizer...')
            
    #         return torch.optim.AdamW(
    #             model.parameters(),
    #             lr=learning_rate,
    #             betas=adamw_betas,
    #             eps=adamw_eps,
    #             weight_decay=weight_decay,
    #             fused=True
    #         )

    #     elif type == 'muon':
    #         models = []
            
    #         if hasattr(model.model, 'text_model'):
    #             loguru.logger.info('Will be performing gradient optimization on the language model (text_model)...')
    #             models.append(model.model.text_model)
    #         if hasattr(model.model, 'connector'):
    #             loguru.logger.info('Will be performing gradient optimization on the projector module (connector)...')
    #             models.append(model.model.connector)

    #         muon_params = []
    #         adamw_params = []
    #         for model_ in models:
    #             for name, p in model_.named_parameters():
    #                 if p.ndim >= 2 and 'embed' not in name and 'lm_head' not in name:
    #                     muon_params.append(p)
    #                 if not (p.ndim >= 2 and 'embed' not in name and 'lm_head' not in name):
    #                     adamw_params.append(p)


    #         loguru.logger.info(f'Using Muon for {len(muon_params):,} parameters and AdamW for {len(adamw_params):,} parameters.')

    #         return Muon(
    #             lr=learning_rate,
    #             wd=weight_decay,
    #             muon_params=muon_params,
    #             momentum=momentum,
    #             nesterov=nesterov,
    #             ns_steps=ns_steps,
    #             adamw_params=adamw_params,
    #             adamw_betas=adamw_betas,
    #             adamw_eps=adamw_eps
    #         )

    def set_lr_scheduler(self, num_iterations: int, optim_config: OptimizerConfig):
        num_warmup_steps = optim_config.num_warmup_steps
        num_training_steps = num_iterations
        num_cycles = optim_config.num_cycles

        loguru.logger.info(f'Using cosine scheduler with {num_warmup_steps:,} warmup steps and {num_training_steps:,} training steps.')

        self.scheduler =  get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )

    @abstractmethod
    def prepare_dataset(self, messages) -> Dataset:
        ...

    @abstractmethod
    def build_dataloader(self, dataset, batch_size: int) -> DataLoader:
        ...

    @abstractmethod
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def train_step(self, batch, step: int, num_iterations: int) -> float:
        ...

    @abstractmethod
    def valid_step(self, batch) -> float:
        ...

    @abstractmethod
    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, num_iterations: int):
        ...

    def save_checkpoint(self, step: int, save_dir: str):
        checkpoint_path = os.path.join(save_dir, f'checkpoint-{step}')
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        self.logger.info(f'Model {self.policy_model.config._name_or_path}')