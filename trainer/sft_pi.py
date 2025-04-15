from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import Dict, Literal
from tqdm import tqdm
from datasets import Dataset


from .trainer import Trainer, TrainerConfig, WandbConfig, OptimizerConfig
from .utils import PreprocessorForLocalizationAndSegmentation


class SFTPiTrainer(Trainer):
    def __init__(
        self,
        policy_model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        device: str,
        config: TrainerConfig,
        wandb_config: WandbConfig,
        add_special_tokens_for_loc_and_seg_tasks: bool = True,
        finetune_mode: Literal['connector', 'text', 'connector_text', 'embeds'] = 'connector_text',
        add_policy_loss: bool = True,
    ) -> None:
        super().__init__(
            policy_model=policy_model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            config=config,
            wandb_config=wandb_config
        )

        if wandb_config is not None and self.config.enable_wandb_logging:
            wandb.init(
                project=self.wandb_config.project,
                entity=self.wandb_config.entity,
                name=self.wandb_config.name,
            )

            self.logger.info(f'WandB initialized with project: {self.wandb_config.project}, entity: {self.wandb_config.entity}, name: {self.wandb_config.name}')
        elif wandb_config is None and self.config.enable_wandb_logging:
            self.logger.warning("WandBConfig is None but enable_wandb_logging is True. "
                                "WandB will not be initialized.")
        else:
            self.logger.info("WandBConfig is None and enable_wandb_logging is False. "
                             "WandB will not be initialized.")


        self.special_tokens_added_flag = False
        if add_special_tokens_for_loc_and_seg_tasks:
            self.special_tokens_added_flag = True
            self.add_tokens_for_loc_and_seg_tasks(precision=0.001)


        self.add_policy_loss = add_policy_loss
        self.logger.info(f'[EXPERIMENTAL] Policy loss added: {self.add_policy_loss}')

        self.finetune_mode = finetune_mode
        self.logger.info(f'Finetune mode set to: {self.finetune_mode}')
        self._configure_finetune_mode()


    def _configure_connector(self):
        for param in self.policy_model.model.connector.parameters():
            param.requires_grad = True
        for param in self.policy_model.model.text_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.lm_head.parameters():
            param.requires_grad = False

        self.policy_model.model.connector.train()
        self.policy_model.model.text_model.eval()
        self.policy_model.model.vision_model.eval()
        self.policy_model.lm_head.eval()

        self.logger.info("Connector is set to train mode.")
        self.logger.info("Language model, Vision model, and LM head are set to eval mode.")

    def _configure_text(self):
        for param in self.policy_model.model.connector.parameters():
            param.requires_grad = False
        for param in self.policy_model.model.text_model.parameters():
            param.requires_grad = True
        for param in self.policy_model.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.lm_head.parameters():
            param.requires_grad = True

        self.policy_model.model.connector.eval()
        self.policy_model.model.text_model.train()
        self.policy_model.model.vision_model.eval()
        self.policy_model.lm_head.train()

        self.logger.info("Language model, Embedding layer, and LM head are set to train mode.")
        self.logger.info("Connector and Vision model are set to eval mode.")

    def _configure_embeds(self):
        for param in self.policy_model.model.connector.parameters():
            param.requires_grad = False
        for param in self.policy_model.model.text_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.lm_head.parameters():
            param.requires_grad = True
        for param in self.policy_model.model.text_model.embed_tokens.parameters():
            param.requires_grad = True

        self.policy_model.model.connector.eval()
        self.policy_model.model.text_model.eval()
        self.policy_model.model.vision_model.eval()
        self.policy_model.model.text_model.embed_tokens.train()
        self.policy_model.lm_head.train()

        self.logger.info("Embedding layer and LM head are set to train mode.")
        self.logger.info("Connector, Language model, and Vision model are set to eval mode.")
        

    def _configure_connector_text(self):
        for param in self.policy_model.model.connector.parameters():
            param.requires_grad = True
        for param in self.policy_model.model.text_model.parameters():
            param.requires_grad = True
        for param in self.policy_model.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.lm_head.parameters():
            param.requires_grad = True

        self.policy_model.model.connector.train()
        self.policy_model.model.text_model.train()
        self.policy_model.model.vision_model.eval()
        self.policy_model.lm_head.train()

        self.logger.info("Connector, Language model, and LM head are set to train mode.")
        self.logger.info("Vision model is set to eval mode.")

    def _configure_finetune_mode(self):
        match self.finetune_mode:
            case 'connector':
                self._configure_connector()
            case 'text':
                self._configure_text()
            case 'connector_text':
                self._configure_connector_text()
            case 'embeds':
                self._configure_embeds()
            case _:
                raise ValueError(f"Unknown finetune mode: {self.finetune_mode}. "
                                 f"Expected one of ['connector', 'text', 'connector_text', 'embeds'].")
        self.logger.info(f"Finetune mode configured: {self.finetune_mode}.")

    def prepare_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        preprocess_fn: str = 'refcocog_sft_seg'
    ):
        assert preprocess_fn is not None, 'preprocess_fn must be provided.'
        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset_name=dataset_name,
            split=split,
            preprocess_fn=preprocess_fn,
        )

    def collate_fn(self, batch, allowed_max_length: int = 4000, pad_side: str = 'right'):
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        pixel_attention_mask = []
        advantage_points = []

        for b in batch:
            messages = b.pop('messages')
            image = b.pop('pil_img')

            _system_and_user_messages = [messages[0], messages[1]]
            _system_and_user_messages_formal = self.processor.apply_chat_template(
                _system_and_user_messages, add_generation_prompt=False, tokenize=False
            )
            _system_and_user_messages_inputs = self.processor(
                text=_system_and_user_messages_formal,
                images=[image],
                return_tensors="pt",
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
                return_tensors="pt",
            )
            _asst_message_input_ids = _asst_message_inputs['input_ids'][0][1:]  ## remove the <|im_start|> token from the beginning
            _asst_message_attention_mask = _asst_message_inputs['attention_mask'][0][1:]

            _input_ids = torch.cat((_system_and_user_messages_input_ids, _asst_message_input_ids), dim=0)
            _ignore_sys_user_ids_mask = torch.cat((torch.zeros_like(_system_and_user_messages_attention_mask), torch.ones_like(_asst_message_attention_mask)), dim=0)
            _attn_mask = torch.cat((_system_and_user_messages_attention_mask, _asst_message_attention_mask), dim=0)

            if pad_side == 'right':
                pad_tokens = torch.tensor([self.processor.tokenizer.pad_token_id] * ((allowed_max_length + 1) - _input_ids.shape[0]), dtype=torch.long)
                _input_ids_padded = torch.cat((_input_ids, pad_tokens), dim=0)
                _attn_mask_padded = torch.cat((_attn_mask, torch.zeros_like(pad_tokens)), dim=0)
                _ignore_sys_user_ids_mask_padded = torch.cat((_ignore_sys_user_ids_mask, torch.zeros_like(pad_tokens)), dim=0)
                _labels = _input_ids_padded.clone()
                _labels[_ignore_sys_user_ids_mask_padded == 0] = -100
                
                ## make an 'advantage points' mask
                _advantage_points = _ignore_sys_user_ids_mask_padded * torch.cumsum(_ignore_sys_user_ids_mask_padded, dim=0)
                _advantage_points = (_advantage_points / _advantage_points.max()) * 5   ## scale to 0-5
                _advantage_points = _advantage_points[1:].contiguous()[:allowed_max_length]

                _input_ids_padded = _input_ids_padded[:-1].contiguous()[:allowed_max_length]
                _attn_mask_padded = _attn_mask_padded[:-1].contiguous()[:allowed_max_length]
                _labels = _labels[1:].contiguous()[:allowed_max_length]
            elif pad_side == 'left':
                raise NotImplementedError("Left padding is not implemented yet.")
            else:
                raise ValueError("pad_side must be either 'right' or 'left'.")

            input_ids.append(_input_ids_padded)
            attention_mask.append(_attn_mask_padded)
            labels.append(_labels)
            pixel_values.append(_pixel_values)
            pixel_attention_mask.append(_pixel_attention_mask)
            advantage_points.append(_advantage_points)

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_attention_mask = torch.stack(pixel_attention_mask, dim=0)
        advantage_points = torch.stack(advantage_points, dim=0)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'pixel_attention_mask': pixel_attention_mask,
            'advantage_points': advantage_points,
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

    def compute_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_mask = (labels != -100).float()
        pseudo_labels = labels.clone()
        pseudo_labels[labels == -100] = 0   ## set the -100 labels to 0 so that we can compute the log probs
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, dim=2, index=pseudo_labels.unsqueeze(-1))
        log_probs = log_probs.squeeze(-1)
        log_probs = log_probs * labels_mask

        del pseudo_labels

        return log_probs, labels_mask

    # def get_log_probs_no_special(self, batch_size, base_ids, seq_len, labels, log_probs):
    #     log_probs_base = []
    #     base_len = len(base_ids)
    #     for b in range(batch_size):
    #         for i in range(seq_len - base_len + 1):
    #             window = labels[b, i:i + base_len]
    #             # print(f"Window: {window}, Base IDs: {base_ids}")
    #             if torch.all(window == torch.tensor(base_ids, device=labels.device)):
    #                 log_probs_sum = log_probs[b, i:i + base_len].sum()
    #                 log_probs_base.append(log_probs_sum)
    #     return torch.stack(log_probs_base, dim=0) if log_probs_base else torch.tensor([], device=labels.device)
            

    def train_step(self, batch: Dict[str, torch.Tensor], step: int, num_iterations: int) -> float:
        self.policy_model.train()

        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'advantage_points']}
        labels = batch.get('labels').to(self.device)

        logits = self.policy_model(**inputs, use_cache=False).logits
        advantage_points = batch.get('advantage_points').type_as(logits).to(self.device)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        log_probs, labels_mask = self.compute_log_probs(logits, labels)

        if self.add_policy_loss:
            policy_loss = -(log_probs * advantage_points) * labels_mask
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
            if self.special_tokens_added_flag:
                special_tokens = self.processor.tokenizer.convert_tokens_to_ids(['<seg_r>', '</seg_r>'])
                special_log_probs_by_id = {
                    special_token: log_probs[labels == token_id].mean().item()
                    for special_token, token_id in zip(['<seg_r>', '</seg_r>'], special_tokens)
                }


                wandb.log({k: v for k, v in special_log_probs_by_id.items()}, commit=False)
            
            wandb.log({'loss': loss.item()}, commit=True)
            # else:
            #     start_seg_token_ids = self.processor.tokenizer.encode('<seg_r>')
            #     end_seg_token_ids = self.processor.tokenizer.encode('</seg_r>')
            #     start_seg_log_probs = self.get_log_probs_no_special(
            #         batch_size=log_probs.shape[0],
            #         base_ids=start_seg_token_ids,
            #         seq_len=log_probs.shape[1],
            #         labels=labels,
            #         log_probs=log_probs,
            #     )
            #     end_seg_log_probs = self.get_log_probs_no_special(
            #         batch_size=log_probs.shape[0],
            #         base_ids=end_seg_token_ids,
            #         seq_len=log_probs.shape[1],
            #         labels=labels,
            #         log_probs=log_probs,
            #     )
            #     start_seg_log_probs_mean = start_seg_log_probs.mean().item() if start_seg_log_probs.numel() > 0 else 0.0
            #     end_seg_log_probs_mean = end_seg_log_probs.mean().item() if end_seg_log_probs.numel() > 0 else 0.0
            #     log_probs_by_id = {
            #         'start_seg': start_seg_log_probs_mean,
            #         'end_seg': end_seg_log_probs_mean,
            #     }
            #     print(f"Step {step}: {log_probs_by_id}")

        return loss.item()

    def valid_step(self, valid_dataloader):
        self.policy_model.eval()

        valid_iter = iter(valid_dataloader)
        iter_count = 0
        valid_loss = 0.0
        while iter_count < 10:  ## first 10 batches
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
                advantage_points = batch.get('advantage_points').type_as(logits).to(self.device)

                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                log_probs, labels_mask = self.compute_log_probs(logits, labels)

                if self.add_policy_loss:
                    policy_loss = -(log_probs * advantage_points) * labels_mask
                    policy_loss = policy_loss.sum() / labels_mask.sum()
                    loss = (ce_loss + policy_loss)
                else:
                    loss = ce_loss
                
                valid_loss += loss.item()

        valid_loss /= iter_count
        wandb.log({'valid_loss': valid_loss}, commit=True)
        
        self.policy_model.train()
        return valid_loss
        
        
    def train(self, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, num_iterations: int):
        num_trainable_params = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable parameters: {num_trainable_params:,}")
        self.logger.info(f"Breaking down the model into trainable and non-trainable parameters:")
        trainable_vision = sum(p.numel() for p in self.policy_model.model.vision_model.parameters() if p.requires_grad)
        trainable_text = sum(p.numel() for p in self.policy_model.model.text_model.parameters() if p.requires_grad)
        trainable_connector = sum(p.numel() for p in self.policy_model.model.connector.parameters() if p.requires_grad)
        trainable_lm_head = sum(p.numel() for p in self.policy_model.lm_head.parameters() if p.requires_grad)
        assert trainable_vision + trainable_text + trainable_connector + trainable_lm_head == num_trainable_params, "Trainable parameters do not match."
        self.logger.info(f"Trainable parameters breakdown:")
        self.logger.info(f"  Vision model: {trainable_vision:,}")
        self.logger.info(f"  Language model: {trainable_text:,}")
        self.logger.info(f"  Connector: {trainable_connector:,}")
        self.logger.info(f"  LM head: {trainable_lm_head:,}")
        self.logger.info(f"Total trainable parameters: {num_trainable_params:,}")
        self.logger.info(f"Training for {num_iterations} iterations.")

        progress_bar = tqdm(range(num_iterations), desc="Training")
        step = 1
        train_iter = iter(train_dataloader)
        while step <= num_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            train_loss = self.train_step(batch, step, num_iterations)
            progress_bar.set_postfix({"loss": train_loss})
            progress_bar.update(1)
            step += 1

            if step % 100 == 0:
                valid_loss = self.valid_step(valid_dataloader)
                progress_bar.set_postfix({"loss": train_loss, "valid_loss": valid_loss})

        progress_bar.close()
        self.logger.info("Training completed.")
        if self.wandb_config is not None:
            wandb.finish()
            self.logger.info("WandB finished.")


if __name__ == "__main__":
    ## example usage
    import sys
    
    match sys.argv[1]:
        case 'with_policy':
            add_policy_loss = True
        case 'no_policy':
            add_policy_loss = False
        case _:
            raise ValueError(f"Unknown policy loss setting: {sys.argv[1]}.")

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
            raise ValueError(f"Unknown finetune mode: {sys.argv[2]}.")
    
    model_name = 'HuggingFaceTB/SmolVLM-Instruct'
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   attn_implementation='flash_attention_2',
                                                   device_map='auto')
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    optimizer_config = OptimizerConfig(
        lr=2e-5,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        num_warmup_steps=64,
        num_cycles=1.0
    )

    config = TrainerConfig(
        batch_size=2,
        optimizer_config=None,
        gradient_accumulation_steps=8,
        grad_clip=1.0,
        enable_wandb_logging=True,
    )
    wandb_config = WandbConfig(
        project='new-openvizr',
        entity='egmaminta',
        name=f'{sys.argv[1]}-{sys.argv[2]}',
    )
    trainer = SFTPiTrainer(
        policy_model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        config=config,
        wandb_config=wandb_config,
        add_special_tokens_for_loc_and_seg_tasks=True,
        finetune_mode=finetune_mode,
        add_policy_loss=add_policy_loss,
    )
    
    optimizer = Trainer.prepare_optimizer(
        model = trainer.policy_model,
        optim_config = optimizer_config,
        type = 'adamw',
    )
    trainer.set_optimizer(optimizer)

    train_dataset = trainer.prepare_dataset(
        dataset_name='jxu124/refcocog',
        split='train',
        preprocess_fn='refcocog_sft_seg'
    )
    valid_dataset = trainer.prepare_dataset(
        dataset_name='jxu124/refcocog',
        split='validation',
        preprocess_fn='refcocog_sft_seg'
    )
    
    train_dataloader = trainer.build_dataloader(train_dataset)
    valid_dataloader = trainer.build_dataloader(valid_dataset)

    num_epochs = 2
    num_iterations = len(train_dataloader) * num_epochs

    trainer.set_lr_scheduler(
        num_iterations=num_iterations,
        optim_config=optimizer_config,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_iterations=num_iterations
    )

    trainer.save_checkpoint(step=9999, save_dir=f'checkpoints/{sys.argv[1]}-{sys.argv[2]}')