from abc import ABC, abstractmethod
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import os
import loguru
from typing import Dict, List
import numpy as np
from muon import Muon
import torch.optim as optim
import gc
from datasets import Dataset
from torch.utils.data import DataLoader

class Trainer(ABC):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: List[optim.Optimizer],
                 device: str,
                 config: Dict):
        self.policy_model = policy_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = loguru.logger
        
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

    @staticmethod
    def prepare_optimizers(model, optim_config: Dict):
        lm_head_params_lr = optim_config.get('lm_head_params_lr')                   ## 2.666e-5
        embed_params_lr = optim_config.get('embed_params_lr')                       ## 2.444e-5
        scalar_params_lr = optim_config.get('scalar_params_lr')                     ## 2.333e-5
        hidden_matrix_params_lr = optim_config.get('hidden_matrix_params_lr')       ## 2.777e-5
        adamw_betas = optim_config.get('adamw_betas')                               ## (0.85, 0.95)
        muon_momentum = optim_config.get('muon_momentum')                           ## 0.95

        hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >=2 and 'embed' not in n and 'lm_head' not in n]
        embed_params = [p for n, p in model.named_parameters() if 'embed' in n and 'bias' not in n]
        scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2]
        lm_head_params = [model.lm_head.weight]
        
        adamw_params = [dict(params=lm_head_params, lr=lm_head_params_lr),
                        dict(params=embed_params, lr=embed_params_lr),
                        dict(params=scalar_params, lr=scalar_params_lr)]
        
        adamw_optimizer = optim.AdamW(adamw_params, betas=adamw_betas, eps=1e-10)

        muon_optimizer = Muon(hidden_matrix_params, lr=hidden_matrix_params_lr, momentum=muon_momentum, rank=0, world_size=1)
        
        optimizers = [adamw_optimizer, muon_optimizer]

        for opt in optimizers:
            for group in opt.param_groups:
                group['initial_lr'] = group['lr']

        return optimizers

    def get_lr(self, step: int, num_iterations: int, optim_config: Dict):     ## stable then decay
        x = step / num_iterations
        assert 0 <= x < 1
        if x < 1 - optim_config.get('cooldown_frac'):
            return 1.0
        else:
            w = (1.0 - x) / optim_config.get('cooldown_frac')
            return w * 1.0 + (1 - w) * 0.1


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
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, num_iterations: int):
        ...

    def save_checkpoint(self, step: int, save_dir: str):
        checkpoint_path = os.path.join(save_dir, f'checkpoint-{step}')
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        self.logger.info(f'Model {self.policy_model.config._name_or_path}')