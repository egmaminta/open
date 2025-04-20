from abc import ABC, abstractmethod
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
import torch
import os
import loguru
import numpy as np
from datasets import Dataset
from typing import Dict, Literal
import wandb

from .config import TrainingConfig
from .utils import configure_finetune_mode


class Trainer(ABC):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        finetune_mode: Literal['connector', 'text', 'connector_text', 'embeds'] = 'connector_text',
        do_add_tokens_for_loc_and_seg_tasks: bool = False,
    ) -> None:
        self.policy_model: torch.nn.Module | AutoModelForVision2Seq = policy_model
        self.processor: AutoProcessor = processor
        self.tokenizer: AutoTokenizer = tokenizer
        self.config: TrainingConfig = config

        self.logger = loguru.logger

        self.logger.info(f'Initializing trainer with device: {self.config.device} and config: {self.config}')
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

        configure_finetune_mode(finetune_mode=finetune_mode, model=self.policy_model)

    def add_tokens_for_loc_and_seg_tasks(self, precision: float = 0.001) -> None:
        """Add special tokens for location and segmentation tasks."""
        float_numbers = np.arange(0., 1., precision)
        tokens_for_loc_and_seg_tasks_original = [f"{num:.{len(str(precision).split('.')[-1])}f}" for num in float_numbers]

        ## uncomment if you want to add segmentation tokens
        # tokens_for_loc_and_seg_tasks = ['<seg' + num.split('.')[-1] + '>' for num in tokens_for_loc_and_seg_tasks_original]
        # tokens_for_loc_and_seg_tasks += ['<seg1000>', '<task:segmentation>', '<seg_r>', '</seg_r>']

        tokens_for_loc_and_seg_tasks = ['<loc' + num.split('.')[-1] + '>' for num in tokens_for_loc_and_seg_tasks_original]
        tokens_for_loc_and_seg_tasks += ['<loc1000>', '<task:localization>', '<loc_r>', '</loc_r>']

        self.logger.info(f'Special tokens for location and segmentation tasks added: {tokens_for_loc_and_seg_tasks}')
        self.logger.info(f'Total tokens added: {len(tokens_for_loc_and_seg_tasks):,} tokens')
        self.logger.info(f'Previous sizes:\n\tTokenizer size: {len(self.processor.tokenizer):,}\n\tEmbedding size: {self.policy_model.get_input_embeddings().weight.shape[0]:,}\n\tLM head size: {self.policy_model.lm_head.weight.shape[0]:,}')

        self.processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks, special_tokens=False)
        self.policy_model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=True)

        assert len(self.processor.tokenizer) == len(self.tokenizer), "Base tokenizer and Processor tokenizer should have the same size."
        self.logger.info(f'New sizes:\n\tTokenizer size: {len(self.processor.tokenizer):,}\n\tEmbedding size: {self.policy_model.get_input_embeddings().weight.shape[0]:,}\n\tLM head size: {self.policy_model.lm_head.weight.shape[0]:,}')
    
    def set_scheduler(self) -> None:
        warmup_iters = self.config.optim_config.warmup_iters
        num_cycles = self.config.optim_config.cosine_scheduler_cycles
        num_iterations = self.config.num_iterations

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_iters,
            num_training_steps=num_iterations,
            num_cycles=num_cycles,
        )
        self.logger.info(f'Setting Scheduler: {self.scheduler.__class__.__name__}')
        self.logger.info(f'Using {self.config.optim_config.cosine_scheduler_cycles} cycles for cosine scheduler with total iterations: {num_iterations} (warmup iterations: {warmup_iters})')

    def configure_and_set_optimizer_and_scheduler(self) -> None:
        optim_type = self.config.optim_config.optim_type
        lr = self.config.optim_config.lr

        if optim_type == 'adamw':
            adamw_weight_decay = self.config.optim_config.adamw_weight_decay
            adamw_betas = self.config.optim_config.adamw_betas
            adamw_eps = self.config.optim_config.adamw_eps
            adamw_fused = self.config.optim_config.adamw_fused

            param_dict = {pn: p for pn, p in self.policy_model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
            no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': adamw_weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_no_decay_params = sum(p.numel() for p in no_decay_params)
            self.logger.info(f'No. of decay params: {num_decay_params:,} || No. of no decay params: {num_no_decay_params:,}')

            self.optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=adamw_betas, eps=adamw_eps, fused=adamw_fused)
            self.logger.info(f'Setting Optimizer: {self.optimizer.__class__.__name__} with learning rate: {lr}, weight decay: {adamw_weight_decay}, betas: {adamw_betas}, eps: {adamw_eps}, fused: {adamw_fused}')
        else:
            raise NotImplementedError(f"Optimizer type {optim_type} not implemented. Supported optimizers: ['adamw']")

        self.set_scheduler()


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
    def evaluate(self, **kwargs) -> float:
        ...

    @abstractmethod
    def train(self, **kwargs) -> None:
        ...

    def save_checkpoint(self, ckpt_name: str) -> None:
        save_dir = self.config.out_dir
        checkpoint_path = os.path.join(save_dir, ckpt_name)
        os.makedirs(save_dir, exist_ok=True)

        self.policy_model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))

        self.logger.info(f'Model checkpoint with Processor and Tokenizer saved at {checkpoint_path}')
        self.logger.info(f'Optimizer state saved at {os.path.join(checkpoint_path, "optimizer.pt")}')
