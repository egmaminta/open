from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import Dict, Literal
from tqdm import tqdm
from datasets import Dataset
import sys

from .trainer_new import Trainer
from .training_config import TrainingConfig, OptimizerConfig, WandBConfig
from .utils import PreprocessorForLocalizationAndSegmentation
from .sft_pi_utils import configure_finetune_mode, compute_log_probs


class SFTPiTrainer(Trainer):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        do_add_tokens_for_loc_and_seg_tasks: bool = False,
        add_policy_loss: bool = False,
        finetune_mode: Literal['connector', 'text', 'connector_text', 'embeds'] = 'connector_text',
    ) -> None:
        super().__init__(policy_model, processor, tokenizer, config, do_add_tokens_for_loc_and_seg_tasks)

        self.add_policy_loss = add_policy_loss
        self.logger.info(f'(Experimental) Policy loss enabled: {self.add_policy_loss}')
        
        self.finetune_mode = finetune_mode
        
        configure_finetune_mode(
            finetune_mode=self.finetune_mode,
            model=self.policy_model,
        )

    def prepare_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        preprocess_fn: Literal['refcocog_sft_seg'] = 'refcocog_sft_seg',
    ) -> Dataset:
        
        assert preprocess_fn is not None and preprocess_fn in ['refcocog_sft_seg'], f"Unsupported preprocess function: {preprocess_fn}"

        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset_name=dataset_name,
            split=split,
            preprocess_fn=preprocess_fn,
        )

    def collate_fn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        pixel_attention_mask = []
        
        if self.add_policy_loss:
            advantage_points = []

        for b in batch:
            messages = b.pop('messages')
            image = b.pop('pil_img')

            _system_and_user_messages = [messages[0], messages[1]]
            _system_and_user_messages_formal = self.processor.apply_chat_template(
                _system_and_user_messages,
                add_generation_prompt=False,
                tokenize=False
            )
            _system_and_user_messages_inputs = self.processor(
                text=_system_and_user_messages_formal,
                images=[image],
                return_tensors='pt'
            )

            _pixel_values = _system_and_user_messages_inputs['pixel_values'][0]
            _pixel_attention_mask = _system_and_user_messages_inputs['pixel_attention_mask'][0]
            _system_and_user_messages_input_ids = _system_and_user_messages_inputs['input_ids'][0]
            _system_and_user_messages_attention_mask = _system_and_user_messages_inputs['attention_mask'][0]

            _asst_message = [messages[2]]
            _asst_message_formal = self.processor.apply_chat_template(
                _asst_message, add_generation_prompt=False, tokenize=False
            )
            _asst_message_inputs = self.processor(
                text=_asst_message_formal,
                images=None,
                return_tensors='pt'
            )
            _asst_message_input_ids = _asst_message_inputs['input_ids'][0][1:]  ## remove the <|im_start|> token from the beginning
            _asst_message_attention_mask = _asst_message_inputs['attention_mask'][0][1:]

            _input_ids = torch.cat((_system_and_user_messages_input_ids, _asst_message_input_ids), dim=0)
            _ignore_sys_user_ids_mask = torch.cat((torch.zeros_like(_system_and_user_messages_attention_mask), torch.ones_like(_asst_message_attention_mask)), dim=0)
            _attn_mask = torch.cat((_system_and_user_messages_attention_mask, _asst_message_attention_mask), dim=0)

            if self.config.padding_side == 'right':
                pad_tokens = torch.tensor([self.processor.tokenizer.pad_token_id] * ((self.config.max_seq_length + 1) - _input_ids.shape[0]), dim=0)
                _input_ids_padded = torch.cat((_input_ids, pad_tokens), dim=0)
                _attn_mask_padded = torch.cat((_attn_mask, torch.zeros_like(pad_tokens)), dim=0)
                _ignore_sys_user_ids_mask_padded = torch.cat((_ignore_sys_user_ids_mask, torch.zeros_like(pad_tokens)), dim=0)
                _labels = _input_ids_padded.clone()
                _labels[_ignore_sys_user_ids_mask_padded == 0] = -100

                ## make an 'advantage points' mask
                if self.add_policy_loss:
                    _advantage_points = _ignore_sys_user_ids_mask_padded * torch.cumsum(_ignore_sys_user_ids_mask_padded, dim=0)
                    _advantage_points = (_advantage_points / _advantage_points.max()) * 5 ## scale to 0-5
                    _advantage_points = _advantage_points[1:].contiguous()[:self.config.max_seq_length]

                _input_ids_padded = _input_ids_padded[:-1].contiguous()[:self.config.max_seq_length]
                _attn_mask_padded = _attn_mask_padded[:-1].contiguous()[:self.config.max_seq_length]
                _labels = _labels[1:].contiguous()[:self.config.max_seq_length]
            elif self.config.padding_side == 'left':
                raise NotImplementedError('Left padding is not supported yet.')
            else:
                raise ValueError(f'Unsupported padding side: {self.config.padding_side}')

            input_ids.append(_input_ids_padded)
            attention_mask.append(_attn_mask_padded)
            labels.append(_labels)
            pixel_values.append(_pixel_values)
            pixel_attention_mask.append(_pixel_attention_mask)

            if self.add_policy_loss:
                advantage_points.append(_advantage_points)

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_attention_mask = torch.stack(pixel_attention_mask, dim=0)

        if self.add_policy_loss:
            advantage_points = torch.stack(advantage_points, dim=0)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'pixel_values': pixel_values,
                'pixel_attention_mask': pixel_attention_mask,
                'advantage_points': advantage_points
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'pixel_values': pixel_values,
                'pixel_attention_mask': pixel_attention_mask
            }

    def build_dataloader(self, dataset: Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:

        self.policy_model.train()

        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'advantage_points']}
        labels = batch.get('labels').to(self.device)

        logits = self.policy_model(**inputs, use_cache=False).logits

        if self.add_policy_loss:
            advantage_points = batch.get('advantage_points').to(self.device)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        if self.add_policy_loss:
            log_probs, labels_mask = compute_log_probs(logits, labels)
            policy_loss = -(log_probs * advantage_points)
            policy_loss = policy_loss.sum() / labels_mask.sum()
            loss = (ce_loss + policy_loss) / self.config.gradient_accumulation_steps
        else:
            loss = ce_loss / self.config.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        if step % 100 == 0:
            if self.special_viz_tokens_added_flag:
                special_tokens = self.processor.tokenizer.convert_tokens_to_ids(['<seg_r>', '</seg_r>'])
                special_log_probs_by_id = {
                    special_token: log_probs[labels==token_id].mean().item()
                    for special_token, token_id in zip(['<seg_r>', '</seg_r>'], special_tokens)
                }

                wandb.log({k: v for k, v in special_log_probs_by_id.items()}, commit=False)
            wandb.log({'loss': loss.item()}, commit=True)
        
        return loss.item()

    def valid_step(self, valid_dataloader: torch.utils.data.DataLoader) -> float:
        self.policy_model.eval()

        valid_iter = iter(valid_dataloader)
        iter_count = 0
        valid_loss = 0.0
        while iter_count < 10:
            try:
                batch = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_dataloader)
                batch = next(valid_iter)
            
            iter_count += 1
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'advantage_points']}
                labels = batch.get('labels').to(self.device)
                logits = self.policy_model(**inputs, use_cache=False).logits
                
                if self.add_policy_loss:
                    advantage_points = batch.get('advantage_points').type_as(logits).to(self.device)

                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                if self.add_policy_loss:
                    log_probs, labels_mask = compute_log_probs(logits, labels)
                    policy_loss = -(log_probs * advantage_points)
                    policy_loss = policy_loss.sum() / labels_mask.sum()
                    loss = (ce_loss + policy_loss)
                else:
                    loss = ce_loss
                
                valid_loss += loss.item()

        valid_loss /= iter_count
        wandb.log({'valid_loss': valid_loss}, commit=True)

        self.policy_model.train()
        return valid_loss

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
    ) -> None:

        num_trainable_params = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        self.logger.info(f'No. of trainable parameters: {num_trainable_params:,}')
        trainable_vision = sum(p.numel() for p in self.policy_model.model.vision_model.parameters() if p.requires_grad)
        trainable_text = sum(p.numel() for p in self.policy_model.model.text_model.parameters() if p.requires_grad)
        trainable_connector = sum(p.numel() for p in self.policy_model.model.connector.parameters() if p.requires_grad)
        trainable_lm_head = sum(p.numel() for p in self.policy_model.lm_head.parameters() if p.requires_grad)

        assert trainable_vision + trainable_text + trainable_connector + trainable_lm_head == num_trainable_params, "Mismatch in trainable parameters."

        self.logger.info(f'Trainable parameters breakdown:')
        self.logger.info(f'    Vision model: {trainable_vision:,}')
        self.logger.info(f'    Language model: {trainable_text:,}')
        self.logger.info(f'    Connector: {trainable_connector:,}')
        self.logger.info(f'    LM head: {trainable_lm_head:,}')
        self.logger.info(f'Total trainable parameters: {num_trainable_params:,}')

        num_iterations = self.config.num_epochs * len(train_dataloader)
        self.logger.info(f'No. of training iterations: {num_iterations:,}')
        self.logger.info(f'No. of training steps per epoch: {len(train_dataloader):,}')

        progress_bar = tqdm(range(num_iterations), desc='Training')
        step = 1
        train_iter = iter(train_dataloader)
        while step <= num_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            train_loss = self.train_step(batch, step)
            progress_bar.set_postfix({'loss(train)': train_loss})
            progress_bar.update(1)
            step += 1

            if step % 100 == 0:
                valid_loss = self.valid_step(valid_dataloader)
                progress_bar.set_postfix({'loss(train)': train_loss, 'loss(valid)': valid_loss})

        progress_bar.close()
        self.logger.info('Training completed.')
        if self.config.enable_wandb_logging:
            wandb.finish()
            self.logger.info('WandB logging finished.')

if __name__ == '__main__':
    match sys.argv[1]:
        case 'with_policy':
            add_policy_loss = True
        case 'without_policy':
            add_policy_loss = False
        case _:
            raise ValueError(f"Invalid argument: {sys.argv[1]}. Use 'with_policy' or 'without_policy'.")

    match sys.argv[2]:
        case 'connector':
            finetune_mode = 'connector'
        case 'text':
            finetune_mode = 'text'
        case 'connector_text':
            finetune_mode = 'connector_text'
        case 'embeds':
            finetune_mode = 'embeds'
        case _:
            raise ValueError(f"Invalid argument: {sys.argv[2]}. Use 'connector', 'text', 'connector_text', or 'embeds'.")

    model_name = 'HuggingFaceTB/SmolVLM-Instruct'
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer_config = OptimizerConfig(
        optim_type='adamw',
        lr=3e-4,
        adamw_weight_decay=0.1,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_fused=True,
        warmup_iters=512,
        cosine_scheduler_cycles=0.5,
    )
    wandb_config = WandBConfig(
        wandb_project='SFT-Pi',
        wandb_entity=sys.argv[3],
        wandb_name=f'SFT-Pi_{sys.argv[1]}_{sys.argv[2]}',
    )
    config = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=8,
        grad_clip=1.0,
        enable_wandb_logging=True,
        out_dir='output',
        device=device,
        num_epochs=2,
        max_seq_length=4000,
        padding_side='right',
        optim_config=optimizer_config,
        wandb_config=wandb_config,
    )

    pi_trainer = SFTPiTrainer(
        policy_model=model,
        processor=processor,
        tokenizer=tokenizer,
        config=config,
        do_add_tokens_for_loc_and_seg_tasks=True,
        add_policy_loss=add_policy_loss,
        finetune_mode=finetune_mode
    )

    optimizer = Trainer.configure_optimizer(
        model=pi_trainer.policy_model,
        config=pi_trainer.config.optim_config,
    )

    pi_trainer.set_optimizer(optimizer)

    train_dataset = pi_trainer.prepare_dataset(
        dataset_name='jxu124/refcocog',
        split='train',
        preprocess_fn='refcocog_sft_seg'
    )
    valid_dataset = pi_trainer.prepare_dataset(
        dataset_name='jxu124/refcocog',
        split='validation',
        preprocess_fn='refcocog_sft_seg'
    )
    train_dataloader = pi_trainer.build_dataloader(train_dataset)
    valid_dataloader = pi_trainer.build_dataloader(valid_dataset)

    pi_trainer.set_lr_scheduler(len(train_dataloader))

    pi_trainer.train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )

    pi_trainer.save_checkpoint(ckpt_name=f'{sys.argv[1]}_{sys.argv[2]}')
