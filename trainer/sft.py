from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import Dict
from tqdm import tqdm


from .trainer import Trainer
from .utils import PreprocessorForLocalizationAndSegmentation


class SFTTrainer(Trainer):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 device: str,
                 config: Dict,
                 tune_embeds_only: bool=False,
                 add_tokens_for_loc_and_seg_tasks: bool=True,
                 finetune_mode: str="connector_text",
                 wandb_config: Dict=None):
        
        super().__init__(policy_model=policy_model,
                         processor=processor,
                         tokenizer=tokenizer,
                         device=device,
                         config=config)

        if wandb_config is not None:
            self.wandb_config = wandb_config
            self.run = wandb.init(**wandb_config)
            self.run_table = wandb.Table(columns=['step', 'ground_truth', 'prediction'])
            self.logger.info(f"Wandb initialized with config: {wandb_config}")
            self.config['enable_wandb_logging'] = True
        else:
            self.wandb_config = None
            self.config['enable_wandb_logging'] = False
            self.logger.info("Wandb not initialized.")

        self.tune_embeds_only = tune_embeds_only
        self.add_tokens_for_loc_and_seg_tasks_flag = add_tokens_for_loc_and_seg_tasks

        self.finetune_mode = finetune_mode
        assert self.finetune_mode is not None and not self.tune_embeds_only, "tune_embeds_only cannot be True if finetune_mode is not None."
        assert self.finetune_mode in ["connector_text", "vision_model", "text_model"], "finetune_mode must be one of ['connector_text', 'vision', 'text', 'connector']."
        self.logger.info(f"Finetune mode: {self.finetune_mode}")
        
        if self.finetune_mode == "connector_text":
            if hasattr(self.policy_model.model, 'vision_model'):
                ## freeze the vision model
                for param in self.policy_model.model.vision_model.parameters():
                    param.requires_grad = False
                self.logger.info('Vision model parameters are frozen.')

                ## set the vision model to eval mode
                self.policy_model.model.vision_model.eval()
                self.logger.info('Vision model is set to eval mode.')
            if hasattr(self.policy_model.model, 'connector'):
                ## unfreeze the connector
                for param in self.policy_model.model.connector.parameters():
                    param.requires_grad = True
                self.logger.info('Connector parameters are unfrozen.')
                ## set the connector to train mode
                self.policy_model.model.connector.train()
                self.logger.info('Connector is set to train mode.')
            if hasattr(self.policy_model.model, 'text_model'):
                ## unfreeze the text model
                for param in self.policy_model.model.text_model.parameters():
                    param.requires_grad = True
                self.logger.info('Text model parameters are unfrozen.')
                ## set the text model to train mode
                self.policy_model.model.text_model.train()
                self.logger.info('Text model is set to train mode.')
            
            self.new_tokens_indices = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["<seg_r>", "</seg_r>"])).to(self.device)
                    
            self.new_token_log_probs_mean_vals = []
            self.new_token_log_probs_std_vals = []


        if self.tune_embeds_only:
            if hasattr(self.policy_model.model, 'vision_model'):
                ## freeze the vision model
                for param in self.policy_model.model.vision_model.parameters():
                    param.requires_grad = False
                self.logger.info('Vision model parameters are frozen.')

                ## set the vision model to eval mode
                self.policy_model.model.vision_model.eval()
                self.logger.info('Vision model is set to eval mode.')

            if hasattr(self.policy_model.model, 'connector'):
                ## freeze the connector
                for param in self.policy_model.model.connector.parameters():
                    param.requires_grad = False
                self.logger.info('Connector parameters are frozen.')

                ## set the connector to eval mode
                self.policy_model.model.connector.eval()
                self.logger.info('Connector is set to eval mode.')

            if hasattr(self.policy_model.model, 'text_model'):
                ## freeze the text model
                for param in self.policy_model.model.text_model.parameters():
                    param.requires_grad = False
                self.logger.info('Text model parameters are frozen.')

                ## set the text model to eval mode
                self.policy_model.model.text_model.eval()
                self.logger.info('Text model is set to eval mode.')

            ## tune the text model embedding layer
            if hasattr(self.policy_model.model.text_model, 'embed_tokens'):
                old_embeds_size = self.policy_model.model.text_model.embed_tokens.weight.shape[0]
                if self.add_tokens_for_loc_and_seg_tasks_flag:
                    self.add_tokens_for_loc_and_seg_tasks(precision=0.001)   ## add special tokens for localization and segmentation tasks
                
                for param in self.policy_model.model.text_model.embed_tokens.parameters():
                    param.requires_grad = True
                
                self.logger.info('Text model embedding layer parameters are unfrozen.')
                self.policy_model.model.text_model.embed_tokens.train()
                self.logger.info('Text model embedding layer is set to train mode.')
                
                if self.add_tokens_for_loc_and_seg_tasks_flag:
                    ## get the indices of the new tokens
                    # new_embeds_size = self.policy_model.model.text_model.embed_tokens.weight.shape[0]
                    # self.new_tokens_indices = torch.arange(old_embeds_size, new_embeds_size).to(self.device)
                    self.new_tokens_indices = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["<seg_r>", "</seg_r>"])).to(self.device)
                    
                    self.new_token_log_probs_mean_vals = []
                    self.new_token_log_probs_std_vals = []



    def prepare_dataset(self, dataset_name: str, split: str="train", preprocess_fn: str='refcocog_sft_seg'):
        assert preprocess_fn is not None, "Preprocess function cannot be None."
        
        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset_name=dataset_name,
            split=split,
            preprocess_fn=preprocess_fn
        )

    def collate_fn(self, batch, allowed_max_length: int=3000):
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        pixel_attention_mask = []
        
        for b in batch:
            ## encode the instructions
            messages = b.pop('messages')
            image = b.pop('pil_img')

            _system_and_user_message = [messages[0], messages[1]]    ## [0] = system message, [1] = user message
            _system_and_user_message = self.processor.apply_chat_template(_system_and_user_message, add_generation_prompt=False, tokenize=False)
            _system_and_user_message_inputs = self.processor(text=_system_and_user_message, images=[image], return_tensors="pt")

            ## get the pixel values and pixel attention mask
            _pixel_values = _system_and_user_message_inputs["pixel_values"][0]
            _pixel_attention_mask = _system_and_user_message_inputs["pixel_attention_mask"][0]

            ## get the system and user input ids
            _system_and_user_message_input_ids = _system_and_user_message_inputs["input_ids"][0]

            _assistant_message = [messages[2]]    ## [2] = assistant message
            _assistant_message = self.processor.apply_chat_template(_assistant_message, add_generation_prompt=False, tokenize=False)
            _assistant_message_inputs = self.processor(text=_assistant_message, images=None, return_tensors="pt")

            ## get the assistant input ids and attention mask
            ## offset the input ids by 1 to remove the <im_start> token
            _assistant_message_input_ids = _assistant_message_inputs["input_ids"][0][1:]    ## remove the <im_start> token from the beginning.

            ## concatenate the system and user input ids with the assistant input ids
            _input_ids = torch.cat((_system_and_user_message_input_ids, _assistant_message_input_ids), dim=0)

            ## pad the input ids to the max length
            _pad_tokens = torch.tensor([self.processor.tokenizer.pad_token_id] * ((allowed_max_length + 1) - _input_ids.shape[0]), dtype=torch.long)
            # _input_ids_padded = torch.cat((_input_ids, _pad_tokens), dim=0)
            _input_ids_padded = torch.cat((_pad_tokens, _input_ids), dim=0)   ## pad the input ids to the left side

            _inputs = _input_ids_padded[:-1].clone()
            _labels = _input_ids_padded[1:].clone()    ## shift the input ids by 1 to get the labels
            _mask_for_labels = _labels == self.processor.tokenizer.pad_token_id
            pad_token_indices = torch.nonzero(_mask_for_labels).squeeze(0)
            
            if pad_token_indices.numel() > 1:
                _labels[pad_token_indices] = -100   ## set the pad token to -100 (ignore)

            _attn_mask = _inputs != self.processor.tokenizer.pad_token_id   ## create the attention mask
            _attn_mask = _attn_mask.long()

            # _labels[:len(_system_and_user_message_input_ids)-1] = -100   ## set the instruction tokens to -100 (ignore)
            _labels[ : len(_pad_tokens) + len(_system_and_user_message_input_ids) - 1] = -100   ## set the instruction tokens to -100 (ignore)
            
            if _inputs.shape[0] > allowed_max_length:
                # _inputs = _inputs[:allowed_max_length]
                # _labels = _labels[:allowed_max_length]
                # _attn_mask = _attn_mask[:allowed_max_length]
                _inputs = _inputs[-allowed_max_length:]
                _labels = _labels[-allowed_max_length:]
                _attn_mask = _attn_mask[-allowed_max_length:]

            input_ids.append(_inputs)
            attention_mask.append(_attn_mask)
            labels.append(_labels)
            pixel_values.append(_pixel_values)
            pixel_attention_mask.append(_pixel_attention_mask)

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_attention_mask = torch.stack(pixel_attention_mask, dim=0)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask}


    def build_dataloader(self, dataset, batch_size: int=2):
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0,
                                           collate_fn=self.collate_fn)


    def train_step(self, batch, step):  ##  batch = train_loader
        self.policy_model.train()

        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)

        try:
            logits = self.policy_model(**inputs).logits
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            ## skip this batch
            return 0

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward()

        step_loss = loss.item()

        if step % 100 == 0:
            with torch.no_grad():
                output_ids = torch.argmax(logits, dim=-1)
                
                predictions = self.processor.tokenizer.batch_decode(output_ids[-2000:], skip_special_tokens=True)
                ground_truth = self.processor.tokenizer.batch_decode(inputs['input_ids'][-2000:], skip_special_tokens=True)
                
                log_probs_original = F.log_softmax(logits, dim=-1)
                
                ## labels contain -100, so we need to mask them out
                # labels = labels.masked_fill(labels == -100, 0)
                # labels = labels.unsqueeze(-1)   ## add a dimension to the labels
                
                ## gather the log probs using the labels
                # log_probs = log_probs_original.gather(2, labels)
                # log_probs = log_probs.squeeze(-1)   ## remove the last dimension
                
                ## mask the log probs using the attention mask
                # attention_mask = inputs['attention_mask']
                
                ## multiply the log probs with the attention mask
                # log_probs = log_probs * attention_mask
                
                ## sum the log probs
                # log_probs = log_probs.sum(dim=-1)
                
                ## get the mean log probs
                # mean_log_probs = log_probs.mean(dim=-1)
                
                ## get the std log probs
                # std_log_probs = log_probs.std(dim=-1)
                
                ## get the log probs of new tokens found in labels
                mask = (labels[..., None] == self.new_tokens_indices).any(dim=-1)
                new_token_ids_found = [row[mask_row] for row, mask_row in zip(labels, mask)]
                new_token_ids_found = torch.nn.utils.rnn.pad_sequence(new_token_ids_found, batch_first=True, padding_value=0)
                new_token_ids_found = new_token_ids_found.to(self.device)
                
                ## get the log probs of new tokens
                new_token_log_probs = log_probs_original.gather(2, new_token_ids_found.unsqueeze(-1))
                new_token_log_probs = new_token_log_probs.squeeze(-1)   ## remove the last dimension
                
                if not self.add_tokens_for_loc_and_seg_tasks_flag:
                    ## create interim mask where we ignore the pad value 0
                    new_token_log_probs_mask = new_token_ids_found != 0
                    new_token_log_probs_mask = new_token_log_probs_mask.long()
                    new_token_log_probs = new_token_log_probs * new_token_log_probs_mask
                
                ## create interim mask where we ignore the pad value 0
                # new_token_log_probs_mask = new_token_ids_found != 0
                # new_token_log_probs_mask = new_token_log_probs_mask.long()
                # new_token_log_probs = new_token_log_probs * new_token_log_probs_mask
                
                new_token_log_probs = new_token_log_probs.sum(dim=-1)
                mean_new_token_log_probs = new_token_log_probs.mean(dim=-1)
                std_new_token_log_probs = new_token_log_probs.std(dim=-1)

                if self.config.get('enable_wandb_logging'):
                    self.run_table.add_data(step, ground_truth, predictions)
                    
                    if self.add_tokens_for_loc_and_seg_tasks_flag or len(self.new_tokens_indices) > 0:
                        self.run.log({"<seg_r>_</seg_r>_mean_log_probs": mean_new_token_log_probs.item()}, commit=False)
                        self.run.log({"<seg_r>_</seg_r>_std_log_probs": std_new_token_log_probs.item()}, commit=False)

                        self.new_token_log_probs_mean_vals.append(mean_new_token_log_probs.float().cpu().numpy())
                        self.new_token_log_probs_std_vals.append(std_new_token_log_probs.float().cpu().numpy())
                    
                    # self.run.log({"std_log_probs": std_log_probs.item()}, commit=False)
                    # self.run.log({"mean_log_probs": mean_log_probs.item()},)

        if (step + 1) % self.config.get('gradient_accumulation_steps') == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            if self.config.get('enable_wandb_logging'):
                self.run.log({'lr': self.optimizer.param_groups[0]['lr'], 'train_loss': step_loss})

        return step_loss

    def valid_step(self, valid_dataloader):
        self.policy_model.eval()

        valid_iter = iter(valid_dataloader)
        iter_count = 0
        with torch.no_grad():
            valid_loss = 0
            while iter_count < 10:  ## validate on 10 batches
                try:
                    valid_batch = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(valid_dataloader)
                    valid_batch = next(valid_iter)

                inputs = {k: v.to(self.device) for k, v in valid_batch.items() if k != 'labels'}
                labels = valid_batch['labels'].to(self.device)
                logits = self.policy_model(**inputs).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                valid_loss += loss.item()

                iter_count += 1

            valid_loss /= 10

            if self.config.get('enable_wandb_logging'):
                self.run.log({'valid_loss': valid_loss})

        self.policy_model.train()

        return valid_loss

    def train(self, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, num_iterations: int,):
        for name, p in self.policy_model.named_parameters():
            if p.requires_grad:
                self.logger.info(f"Trainable parameter: {name}, shape: {p.shape}")
        
        self.logger.info(f"No. of trainable parameters in the vision model: {sum(p.numel() for p in self.policy_model.model.vision_model.parameters() if p.requires_grad):,}")
        self.logger.info(f"No. of trainable parameters in the text model: {sum(p.numel() for p in self.policy_model.model.text_model.parameters() if p.requires_grad):,}")
        self.logger.info(f"No. of trainable parameters in the connector: {sum(p.numel() for p in self.policy_model.model.connector.parameters() if p.requires_grad):,}")
        self.logger.info(f"No. of trainable parameters in the lm head: {sum(p.numel() for p in self.policy_model.lm_head.parameters() if p.requires_grad):,}")

        progbar = tqdm(total=num_iterations, desc="Training", unit="step")
        step = 1
        
        train_iter = iter(train_dataloader)
        
        while step <= num_iterations:
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                train_batch = next(train_iter)

            train_loss = self.train_step(train_batch, step=step)
            step += 1
            progbar.update(1)
            progbar.set_postfix({"train_loss": train_loss})

            if step % 100 == 0:
                valid_loss = self.valid_step(valid_dataloader)
                progbar.set_postfix({"valid_loss": valid_loss})
        
        progbar.close()
        self.logger.info("Training completed.")
        if self.config.get('enable_wandb_logging'):
            self.run.log({'training_sample': self.run_table}, commit=False)
            self.run.log({"mean_new_token_log_probs": wandb.Histogram(self.new_token_log_probs_mean_vals)}, commit=False)
            self.run.log({"std_new_token_log_probs": wandb.Histogram(self.new_token_log_probs_std_vals)})


if __name__ == "__main__":    
    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    
    wandb_config = {
        'project': 'samsung-openvizr',
        'entity': 'egmaminta',
        'name': 'refcocog_sft_seg-connector-text-no_special_tokens',
    }

    optim_config = {'learning_rate': 3.33333e-5,
                    'weight_decay': 0.1,
                    'momentum': 0.95,
                    'nesterov': True,
                    'ns_steps': 5,
                    'adamw_betas': (0.9, 0.95),
                    'adamw_eps': 1e-8,
                    'num_warmup_steps': 64,
                    'num_cycles': 0.5,}

    config = {'enable_wandb_logging': False,
              'gradient_accumulation_steps': 8}

    policy_model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = SFTTrainer(policy_model=policy_model, processor=processor, tokenizer=tokenizer,
                         device=device, config=config, tune_embeds_only=False,
                         finetune_mode="connector_text", wandb_config=wandb_config,
                         add_tokens_for_loc_and_seg_tasks=False)

    optimizer = Trainer.prepare_optimizers(trainer.policy_model, optim_config, type='adamw')
    trainer.set_optimizer(optimizer)

    train_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="train", preprocess_fn="refcocog_sft_seg")
    valid_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="validation", preprocess_fn="refcocog_sft_seg")

    train_dataloader = trainer.build_dataloader(train_dataset, batch_size=2)
    valid_dataloader = trainer.build_dataloader(valid_dataset, batch_size=2)

    num_training_epochs = 2
    num_iterations = int(len(train_dataloader) * num_training_epochs)

    trainer.set_lr_scheduler(num_iterations=num_iterations, optim_config=optim_config)

    trainer.train(train_dataloader=train_dataloader,
                  valid_dataloader=valid_dataloader,
                  num_iterations=num_iterations)

    trainer.save_checkpoint(step='9999', save_dir='connector-text-no_special_tokens')

    trainer.run.finish()
