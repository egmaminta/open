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
                 format_prompt: Callable,
                 device: str):
        self.policy_model = policy_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.format_prompt = format_prompt

    def add_tokens_for_loc_and_seg_tasks(self, precision: float=0.001):
        float_numbers = np.arange(0., 1 + precision, precision)
        tokens_for_loc_and_seg_tasks = [f"{num:.{len(str(precision).split('.')[-1])}f}" for num in float_numbers]
        tokens_for_loc_and_seg_tasks += [', ']
        
        self.processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks)
        self.policy_model.resize_token_embeddings(len(self.processor.tokenizer))

        del tokens_for_loc_and_seg_tasks, float_numbers
        gc.collect()

    @abstractmethod
    def prepare_inputs(self, messages):
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