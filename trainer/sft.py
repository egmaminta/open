from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
from typing import Callable
import numpy as np

from .trainer import Trainer


class SFTTrainer(Trainer):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 format_prompt: Callable,
                 device: str):
        super().__init__(policy_model=policy_model,
                         processor=processor,
                         tokenizer=tokenizer,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         format_prompt=format_prompt,
                         device=device)


    def prepare_inputs(self, prompt: str, image):
        ...


    def train_step(self, batch):
        self.policy_model.train()