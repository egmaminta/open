from abc import ABC, abstractmethod
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
import torch
import os
import loguru
import numpy as np
from datasets import Dataset
from typing import Dict
import wandb


from .training_config import TrainingConfig, OptimizerConfig


class Trainer(ABC):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        do_add_tokens_for_loc_and_seg_tasks: bool = False,
    ) -> None:
        self.policy_model = policy_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = config.device
        self.config = config

        self.logger = loguru.logger

        self.logger.info(f'Initializing trainer with device: {self.device} and config: {self.config}')
        self.logger.info(f'Using Policy Model: {self.policy_model.config._name_or_path}')

        if hasattr(self.processor, 'image_processor'):
            self.logger.info('Setting image processor size to height=512, width=512')
            self.processor.image_processor.size = {'height': 512, 'width': 512}

        self.special_viz_tokens_added_flag = False
        if do_add_tokens_for_loc_and_seg_tasks:
            self.special_viz_tokens_added_flag = True
            self.add_tokens_for_loc_and_seg_tasks()

        if self.config.enable_wandb_logging:
            wandb.init(
                project=self.config.wandb_config.wandb_project,
                entity=self.config.wandb_config.wandb_entity,
                name=self.config.wandb_config.wandb_name,
            )
            self.logger.info(f'WandB logging enabled with project: {self.config.wandb_config.wandb_project}, entity: {self.config.wandb_config.wandb_entity}, name: {self.config.wandb_config.wandb_name}')
        else:
            self.logger.info('WandB logging disabled.')


    def add_tokens_for_loc_and_seg_tasks(self, precision: float = 0.001) -> None:
        """Add special tokens for location and segmentation tasks."""
        float_numbers = np.arange(0., 1., precision)
        tokens_for_loc_and_seg_tasks_original = [f"{num:.{len(str(precision).split('.')[-1])}f}" for num in float_numbers]
        tokens_for_loc_and_seg_tasks = ['<seg' + num.split('.')[-1] + '>' for num in tokens_for_loc_and_seg_tasks_original]
        tokens_for_loc_and_seg_tasks += ['<seg1000>']
        tokens_for_loc_and_seg_tasks += ['<task:segmentation>', '<seg_r>', '</seg_r>']
        tokens_for_loc_and_seg_tasks += ['<loc' + num.split('.')[-1] + '>' for num in tokens_for_loc_and_seg_tasks_original]
        tokens_for_loc_and_seg_tasks += ['<loc1000>']
        tokens_for_loc_and_seg_tasks += ['<task:localization>, <loc_r>', '</loc_r>']

        self.logger.info(f'Adding {tokens_for_loc_and_seg_tasks} tokens for localization and segmentation tasks.')
        self.logger.info(f'Total tokens added: {len(tokens_for_loc_and_seg_tasks):,} tokens')
        self.logger.info('Previous sizes:')
        self.logger.info(f' Tokenizer size: {len(self.processor.tokenizer):,}')
        self.logger.info(f' Embedding size: {self.policy_model.get_input_embeddings().weight.shape[0]:,}')
        self.logger.info(f' LM head size: {self.policy_model.lm_head.weight.shape[0]:,}')

        self.processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.policy_model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)

        self.logger.info(f'New sizes:')
        self.logger.info(f' Tokenizer size: {len(self.processor.tokenizer):,} ({len(tokens_for_loc_and_seg_tasks):,} new tokens)')
        self.logger.info(f' Embedding size: {self.policy_model.get_input_embeddings().weight.shape[0]:,}')
        self.logger.info(f' LM head size: {self.policy_model.lm_head.weight.shape[0]:,}')

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.logger.info(f'Setting optimizer: {self.optimizer.__class__.__name__}')

    def set_lr_scheduler(self, len_dataloader: int) -> None:
        """Set the learning rate scheduler."""
        warmup_iters = self.config.optim_config.warmup_iters
        num_cycles = self.config.optim_config.cosine_scheduler_cycles
        num_iterations = self.config.num_epochs * len_dataloader

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_iters,
            num_training_steps=num_iterations,
            num_cycles=num_cycles,
        )
        self.logger.info(f'Setting learning rate scheduler with warmup iterations: {warmup_iters}, num cycles: {num_cycles}, total iterations: {num_iterations}')


    @staticmethod
    def configure_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        """Configure the optimizer for the model."""
        optim_type = config.optim_type
        lr = config.lr
        if optim_type == 'adamw':
            adamw_weight_decay = config.adamw_weight_decay
            adamw_betas = config.adamw_betas
            adamw_eps = config.adamw_eps
            adamw_fused = config.adamw_fused

            param_dict = {pn: p for pn, p in model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': adamw_weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            loguru.logger.info(f'Using AdamW optimizer with {num_decay_params:,} decay parameters and {num_nodecay_params:,} no-decay parameters.')

            return torch.optim.AdamW(
                optim_groups, lr=lr, betas=adamw_betas, eps=adamw_eps, fused=adamw_fused
            )

    @abstractmethod
    def prepare_dataset(self, **kwargs) -> Dataset:
        ...

    @abstractmethod
    def build_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        ...

    @abstractmethod
    def collate_fn(self, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def train_step(self, **kwargs) -> float:
        ...

    @abstractmethod
    def valid_step(self, **kwargs) -> float:
        ...

    @abstractmethod
    def train(self, **kwargs) -> None:
        ...

    def save_checkpoint(self, ckpt_name: str) -> None:
        """Save the model checkpoint."""
        save_dir = self.config.out_dir
        checkpoint_path = os.path.join(save_dir, ckpt_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))

        self.logger.info(f'Saved checkpoint at {checkpoint_path}')
        self.logger.info(f'Saved optimizer state at {os.path.join(checkpoint_path, "optimizer.pt")}')
