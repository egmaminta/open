from abc import ABC, abstractmethod
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import os
from loguru import logger
from typing import Callable
import numpy as np
import gc


class Trainer(ABC):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str):
        self.policy_model = policy_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        
        if hasattr(self.processor, 'image_processor'):
            self.logger.info('Setting image processor size to 512x512...')
            self.processor.image_processor.size = {'height': 512, 'width': 512}

        self.logger.info('Initializing Trainer...')
        self.logger.info(f'Using policy model: {self.policy_model.config._name_or_path}')
        if torch.cuda.is_available() and self.device == 'cuda':
            self.logger.info('Moving model to GPU...')
            self.policy_model.to(self.device)
            self.logger.info('Using GPU for training...')
        else:
            self.logger.info('Not using GPU, setting device to default CPU...')
            self.logger.info('Using CPU for training...')

    def add_tokens_for_loc_and_seg_tasks(self, precision: float=0.001):
        float_numbers = np.arange(0., 1 + precision, precision)
        tokens_for_loc_and_seg_tasks = [f"{num:.{len(str(precision).split('.')[-1])}f}" for num in float_numbers]
        tokens_for_loc_and_seg_tasks += [', ']
        
        self.logger.info(f'Adding {len(tokens_for_loc_and_seg_tasks)} tokens for localization and segmentation tasks.')
        self.logger.info(f'Previous tokenizer size: {len(self.processor.tokenizer):,}')
        self.logger.info(f'Previous Embedding size: {self.policy_model.get_input_embeddings().weight.shape}')
        self.logger.info(f'Previous LM head size: {self.policy_model.lm_head.weight.shape}')
        
        self.processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks)
        self.policy_model.resize_token_embeddings(len(self.processor.tokenizer))

        self.logger.info(f'New tokenizer size: {len(self.processor.tokenizer):,} ({len(tokens_for_loc_and_seg_tasks):,} new tokens)')
        self.logger.info(f'New Embedding size: {self.policy_model.get_input_embeddings().weight.shape}')
        self.logger.info(f'New LM head size: {self.policy_model.lm_head.weight.shape}')
        
        del tokens_for_loc_and_seg_tasks, float_numbers
        gc.collect()

    @abstractmethod
    def prepare_dataset(self, messages):
        ...

    @abstractmethod
    def build_dataloader(self, dataset, batch_size: int):
        ...

    @abstractmethod
    def collate_fn(self, batch):
        ...

    @abstractmethod
    def train_step(self, batch):
        ...

    def save_checkpoint(self, step: int, save_dir: str):
        checkpoint_path = os.path.join(save_dir, f'checkpoint-{step}')
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        self.logger.info(f'Model {self.policy_model.config._name_or_path}')