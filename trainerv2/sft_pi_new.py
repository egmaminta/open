from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import Dict, Literal, Iterator
from tqdm import tqdm
import datasets
import sys
import os
from transformers.image_utils import load_image

from .trainer import Trainer
from .config import TrainingConfig, OptimizerConfig, WandBConfig
from .data_preprocessor import PreprocessorForLocalizationAndSegmentation
from .utils import compute_log_probs, get_batch


class SFTPiTrainer(Trainer):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        finetune_mode: Literal['connector', 'text', 'connector_text', 'embeds'] = 'connector_text',
        do_add_tokens_for_loc_and_seg_tasks: bool = False,
        add_policy_loss: bool = True
    ) -> None:
        super().__init__(policy_model, processor, tokenizer, config, finetune_mode, do_add_tokens_for_loc_and_seg_tasks)

        self.add_policy_loss = add_policy_loss
        self.logger.info(f'(EXPERIMENTAL) Policy loss enabled: {add_policy_loss}')

    def prepare_dataset(
        self,
        dataset: Literal['REFCOCO', 'REFCOCOG', 'REFCOCOPLUS'],
        split: str = 'train',
        preprocess_fn: str = 'refcocog_sft_seg'
    ) -> datasets.Dataset:
        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset=dataset,
            split=split,
            preprocess_fn=preprocess_fn
        )

    def collate_fn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        pixel_attention_mask = []

        if self.add_policy_loss:    advantage_points = []

        for b in batch:
            _sys_user_messages = b.pop('sys_user_message')
            _image_path = b.pop('image_path')

            _sys_user_messages_formal = self.processor.apply_chat_template(
                _sys_user_messages, add_generation_prompt=False, tokenize=False
            )
            _image = load_image(_image_path)
            _sys_user_messages_inputs = self.processor(
                text=_sys_user_messages_formal,
                images=[_image],
                return_tensors='pt'
            )

            _pixel_values = _sys_user_messages_inputs['pixel_values'][0]
            _pixel_attention_mask = _sys_user_messages_inputs['pixel_attention_mask'][0]
            _sys_user_messages_input_ids = _sys_user_messages_inputs['input_ids'][0]
            _sys_user_messages_attention_mask = _sys_user_messages_inputs['attention_mask'][0]

            _asst_message = b.pop('asst_message')
            
            _asst_message_formal = self.processor.apply_chat_template(
                [_asst_message], add_generation_prompt=False, tokenize=False
            )
            _asst_message_inputs = self.processor(
                text=_asst_message_formal,
                images=None,
                return_tensors='pt'
            )
            _asst_message_input_ids = _asst_message_inputs['input_ids'][0][1:]  ## remove the <|im_start|> token from the beginning
            _asst_message_attention_mask = _asst_message_inputs['attention_mask'][0][1:]

            _input_ids = torch.cat((_sys_user_messages_input_ids, _asst_message_input_ids), dim=0)
            _ignore_sys_user_ids_mask = torch.cat((torch.zeros_like(_sys_user_messages_attention_mask), torch.ones_like(_asst_message_attention_mask)), dim=0)
            _attn_mask = torch.cat((_sys_user_messages_attention_mask, _asst_message_attention_mask), dim=0)

            if self.config.padding_side == 'right':
                pad_tokens = torch.tensor([self.processor.tokenizer.pad_token_id] * ((self.config.max_seq_length + 1) - _input_ids.shape[0]), dtype=torch.long)
                _input_ids_padded = torch.cat((_input_ids, pad_tokens), dim=0)
                _attn_mask_padded = torch.cat((_attn_mask, torch.zeros_like(pad_tokens)), dim=0)
                _ignore_sys_user_ids_mask_padded = torch.cat((_ignore_sys_user_ids_mask, torch.zeros_like(pad_tokens)), dim=0)
                
                if _ignore_sys_user_ids_mask_padded.sum() == 0:
                    raise ValueError("No 1s in _ignore_sys_user_ids_mask_padded. This should not happen.")
                
                _labels = _input_ids_padded.clone()
                _labels[_ignore_sys_user_ids_mask_padded == 0] = -100

                if self.add_policy_loss:    ## NOQA: add a rewarding mask
                    # _advantage_points = _ignore_sys_user_ids_mask_padded * torch.cumsum(_ignore_sys_user_ids_mask_padded, dim=0)
                    # _advantage_points = (_advantage_points / _advantage_points.max()) * 5
                    _advantage_points = _ignore_sys_user_ids_mask_padded * 0.9  ## NOQA: 0.1 is a placeholder value
                    _advantage_points = _advantage_points[1:].contiguous()[:self.config.max_seq_length]

                _input_ids_padded = _input_ids_padded[:-1].contiguous()[:self.config.max_seq_length]
                _attn_mask_padded = _attn_mask_padded[:-1].contiguous()[:self.config.max_seq_length]
                _labels = _labels[1:].contiguous()[:self.config.max_seq_length]
            elif self.config.padding_side == 'left':
                raise NotImplementedError('Left padding is not implemented yet.')
            else:
                raise ValueError(f"Invalid padding side: {self.config.padding_side}. Must be one of ['left', 'right'].")

            input_ids.append(_input_ids_padded)
            attention_mask.append(_attn_mask_padded)
            labels.append(_labels)
            pixel_values.append(_pixel_values)
            pixel_attention_mask.append(_pixel_attention_mask)
            if self.add_policy_loss:    advantage_points.append(_advantage_points)

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

    def build_dataloader(self, dataset: datasets.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def evaluate(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        curr_val_iter: Iterator[Dict[str, torch.Tensor]],
        eval_iters: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        
        self.policy_model.eval()

        valid_loss = 0.0
        
        with torch.no_grad():
            for _ in range(eval_iters):
                batch, curr_val_iter  = get_batch(val_dataloader, curr_val_iter)

                inputs = {k: v.to(self.config.device) for k, v in batch.items() if k not in ['labels', 'advantage_points']}
                labels = batch.pop('labels').to(self.config.device)
                logits = self.policy_model(**inputs, use_cache=False).logits

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                valid_loss += loss.item()

        valid_loss /= eval_iters

        if self.config.enable_wandb_logging:
            wandb.log({
                'valid_loss': valid_loss,
            }, commit=True)

        self.policy_model.train()

        return curr_val_iter


    def train(self, train_dataset: datasets.Dataset, eval_dataset: datasets.Dataset) -> None:
        self.policy_model.train()

        n_trainable_params = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable parameters: {n_trainable_params / 1e6:.2f}M")
        trainable_vision = sum(p.numel() for p in self.policy_model.model.vision_model.parameters() if p.requires_grad)
        trainable_text = sum(p.numel() for p in self.policy_model.model.text_model.parameters() if p.requires_grad)
        trainable_connector = sum(p.numel() for p in self.policy_model.model.connector.parameters() if p.requires_grad)
        trainable_lm_head = sum(p.numel() for p in self.policy_model.lm_head.parameters() if p.requires_grad)
        assert trainable_vision + trainable_text + trainable_connector + trainable_lm_head == n_trainable_params, \
            f"Trainable parameters mismatch: {trainable_vision + trainable_text + trainable_connector + trainable_lm_head} != {n_trainable_params}"
        self.logger.info(f"Breakdown of trainable parameters:\n\t\t\tVision Encoder: {trainable_vision / 1e6:.2f}M\n\t\t\tText Encoder: {trainable_text / 1e6:.2f}M\n\t\t\tConnector: {trainable_connector / 1e6:.2f}M\n\t\t\tLM Head: {trainable_lm_head / 1e6:.2f}M")

        train_dataloader = self.build_dataloader(train_dataset)
        val_dataloader = self.build_dataloader(eval_dataset)

        self.config.num_iterations = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.logger.info(f"No. of training iterations: {self.config.num_iterations:,}")

        self.configure_and_set_optimizer_and_scheduler()

        progress_bar = tqdm(range(self.config.num_iterations), desc="Training", file=sys.stdout)
        iter_num = 0

        curr_train_iter = iter(train_dataloader)
        curr_val_iter = iter(val_dataloader)

        while iter_num < self.config.num_iterations:
            for _ in range(self.config.gradient_accumulation_steps):
                batch, curr_train_iter = get_batch(train_dataloader, curr_train_iter)

                inputs = {k: v.to(self.config.device) for k, v in batch.items() if k not in ['labels', 'advantage_points']}
                labels = batch.pop('labels').to(self.config.device)

                logits = self.policy_model(**inputs, use_cache=False).logits

                if self.add_policy_loss:
                    advantage_points = batch.pop('advantage_points').to(self.config.device)

                ## ensure logits and labels are valid
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    self.logger.warning("Logits contain NaN or Inf values.")
                    raise ValueError("Logits contain NaN or Inf values.")
                ## check if all labels are -100
                if (labels == -100).all():
                    self.logger.warning("All labels are -100.")
                    raise ValueError("All labels are -100. This should not happen.")

                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                if self.add_policy_loss:
                    log_probs, labels_mask = compute_log_probs(logits, labels)
                    policy_loss = -(log_probs * advantage_points.type_as(logits))
                    policy_loss = policy_loss.sum() / labels_mask.sum()
                    loss = (ce_loss + policy_loss) / self.config.gradient_accumulation_steps
                else:
                    loss = ce_loss / self.config.gradient_accumulation_steps

                loss.backward()

            ## write true labels vs predicted labels to model_predictions.txt
            write_predictions_path = os.path.join(self.config.out_dir, 'model_predictions.txt')
            if not os.path.exists(write_predictions_path):
                with open(write_predictions_path, 'w') as f:
                    f.write('')
            with open(write_predictions_path, 'a') as f:
                f.write(f"Iteration: {iter_num + 1}\n")
                f.write(f"Ground truth: {self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}\n")
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)
                    pred_sample = self.processor.tokenizer.decode(pred[0], skip_special_tokens=True)
                    f.write(f"Prediction: {pred_sample}\n")
                f.write('=' * 100 + '\n')
                f.flush()
                os.fsync(f.fileno())

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            if (iter_num + 1) % self.config.wandb_config.log_every_n_steps == 0 and self.config.enable_wandb_logging:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'iteration': iter_num + 1,
                }, commit=False)

                curr_val_iter = self.evaluate(
                    val_dataloader=val_dataloader,
                    curr_val_iter=curr_val_iter,
                    eval_iters=self.config.eval_iters
                )

            iter_num += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=loss.item(),
                learning_rate=self.optimizer.param_groups[0]['lr'],
                iteration=iter_num,
            )

        if self.config.enable_wandb_logging:
            wandb.finish()

        progress_bar.close()
        self.logger.info("Training complete.")

if __name__ == "__main__":
    match sys.argv[1]:
        case 'with_policy':
            add_policy_loss = True
        case 'without_policy':
            add_policy_loss = False
        case _:
            raise ValueError(f"Invalid argument: {sys.argv[1]}. Must be one of ['with_policy', 'without_policy'].")

    match sys.argv[2]:
        case 'connector':
            finetune_mode = 'connector'
        case 'text':
            finetune_mode = 'text'
        case 'connector_text':
            finetune_mode = 'connector_text'
        case 'embeds':
            finetune_mode = 'embeds'
        case 'all':
            finetune_mode = 'all'
        case _:
            raise ValueError(f"Invalid argument: {sys.argv[2]}. Must be one of ['connector', 'text', 'connector_text', 'embeds', 'all'].")

    match sys.argv[3]:
        case 'add_special_loc_seg_tokens':
            do_add_tokens_for_loc_and_seg_tasks = True
        case 'no_special_loc_seg_tokens':
            do_add_tokens_for_loc_and_seg_tasks = False
        case _:
            raise ValueError(f"Invalid argument: {sys.argv[3]}. Must be one of ['add_special_loc_seg_tokens', 'no_special_loc_seg_tokens'].")

    model_name = 'HuggingFaceTB/SmolVLM-Instruct'
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto',
    )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    optimizer_config = OptimizerConfig(
        optim_type='adamw',
        lr=1e-4,
        adamw_weight_decay=0.1,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_fused=True,
        warmup_iters=1024,
        cosine_scheduler_cycles=0.5
    )

    wandb_config = WandBConfig(
        wandb_project='SFTPi-Combined-Loc-Seg',
        wandb_entity=sys.argv[4],
        wandb_name=f'SFTPi_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}',
        log_every_n_steps=100
    )

    config = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=32,
        grad_clip=1.0,
        enable_wandb_logging=True,
        out_dir='output',
        device='cuda',
        num_epochs=3,
        max_seq_length=4000,
        padding_side='right',
        optim_config=optimizer_config,
        wandb_config=wandb_config,
        eval_iters=10
    )

    trainer = SFTPiTrainer(
        policy_model=model,
        processor=processor,
        tokenizer=tokenizer,
        config=config,
        finetune_mode=finetune_mode,
        do_add_tokens_for_loc_and_seg_tasks=do_add_tokens_for_loc_and_seg_tasks,
        add_policy_loss=add_policy_loss
    )

    train_dataset_seg = trainer.prepare_dataset(
        dataset='REFCOCOG',
        split='train',
        preprocess_fn='refcocog_sft_seg'
    )
    train_dataset_loc = trainer.prepare_dataset(
        dataset='REFCOCO',
        split='train',
        preprocess_fn='refcocog_sft_loc'
    )
    train_dataset = datasets.concatenate_datasets([train_dataset_seg, train_dataset_loc])
    train_dataset = train_dataset.shuffle(seed=42)

    eval_dataset_seg = trainer.prepare_dataset(
        dataset='REFCOCOG',
        split='validation',
        preprocess_fn='refcocog_sft_seg'
    )
    eval_dataset_loc = trainer.prepare_dataset(
        dataset='REFCOCO',
        split='validation',
        preprocess_fn='refcocog_sft_loc'
    )

    eval_dataset = datasets.concatenate_datasets([eval_dataset_seg, eval_dataset_loc])
    eval_dataset = eval_dataset.shuffle(seed=42)

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.save_checkpoint(ckpt_name=f'SFTPi_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_combined_loc_seg')
