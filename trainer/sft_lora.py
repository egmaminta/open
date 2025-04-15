from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import List, Dict
from datasets import Dataset
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
import wandb


from .trainer import Trainer
from .utils import PreprocessorForLocalizationAndSegmentation


run = wandb.init(project='refcocog_sft_seg-with_special_tokens-lora', entity='egmaminta', name='refcocog_sft_seg-with_special_tokens-lora')

class SFTTrainer(Trainer):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 config: Dict):
        
        super().__init__(policy_model=policy_model,
                         processor=processor,
                         tokenizer=tokenizer,
                         optimizer=optimizer,
                         device=device,
                         config=config)

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
        # self.policy_model.train()
        self.policy_model.model.text_model.train()
        self.policy_model.lm_head.train()
        # self.policy_model.model.connector.train()
        # self.policy_model.lm_head.train()
        # self.policy_model.model.vision_model.eval()   ## freeze the vision model

        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)

        logits = self.policy_model(**inputs).logits

        ## decode the logits
        if step % 1000 == 0:
            with torch.no_grad():
                output_ids = torch.argmax(logits, dim=-1)
                decoded_output = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                decoded_inputs = self.processor.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
                ## get the log probs (log_softmax)
                ## use gather (use labels as index) to get the log probs
                log_probs = F.log_softmax(logits, dim=-1)
                ## labels contain -100, so we need to mask them out
                labels = labels.masked_fill(labels == -100, 0)
                labels = labels.unsqueeze(-1)   ## add a dimension to the labels
                ## gather the log probs using the labels
                log_probs = log_probs.gather(2, labels)
                log_probs = log_probs.squeeze(-1)   ## remove the last dimension
                ## mask the log probs using the attention mask
                attention_mask = inputs['attention_mask']
                ## multiply the log probs with the attention mask
                log_probs = log_probs * attention_mask
                ## sum the log probs
                log_probs = log_probs.sum(dim=-1)
                ## get the mean log probs
                mean_log_probs = log_probs.mean(dim=-1)
                ## get the std log probs
                std_log_probs = log_probs.std(dim=-1)

                run.log({'mean_log_probs': mean_log_probs.item(),
                         'std_log_probs': std_log_probs.item(),
                         'predicted_output': decoded_output[0],
                         'ground_truth': decoded_inputs[0],
                         'step': step})
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss.backward()

        step_loss = loss.item()

        if (step + 1) % self.config.get('gradient_accumulation_steps') == 0:
            self.logger.info("Performing gradient accumulation update...")
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            run.log({'train_loss': step_loss,})
            
            if self.config.get('enable_wandb_logging'):
                to_log = {'train_loss': step_loss, 'step': step, 'learning_rate': step_loss}
                wandb.log(to_log)

        return step_loss

    def valid_step(self, batch, step):  ## if step % 1000 == 0
        self.policy_model.eval()

        with torch.no_grad():
            valid_loss = 0
            for _ in range(10): ## sample 10 batch
                valid_batch = next(iter(batch))
                inputs = {k: v.to(self.device) for k, v in valid_batch.items() if k != 'labels'}
                labels = valid_batch['labels'].to(self.device)
                logits = self.policy_model(**inputs).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                valid_loss += loss.item()
            valid_loss /= 10


            run.log({'valid_loss': valid_loss})

            if self.config.get('enable_wandb_logging'):
                wandb.log({'valid_loss': valid_loss, 'step': step})
        
        # self.policy_model.model.text_model.train()
        # self.policy_model.model.connector.train()
        # self.policy_model.lm_head.train()
        # self.policy_model.model.vision_model.eval()   ## freeze the vision model
        # self.policy_model.train()
        
        self.policy_model.model.text_model.train()
        self.policy_model.lm_head.train()
        
        return valid_loss

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              valid_dataloader: torch.utils.data.DataLoader,
              num_iterations: int,):
        
        ## print the number of trainable parameters
        for name, p in self.policy_model.named_parameters():
            if p.requires_grad:
                self.logger.info(f"Trainable parameter: {name}, shape: {p.shape}")
        
        self.logger.info(f"No. of trainable parameters in the vision model: {sum(p.numel() for p in self.policy_model.model.vision_model.parameters() if p.requires_grad)}")
        self.logger.info(f"No. of trainable parameters in the text model: {sum(p.numel() for p in self.policy_model.model.text_model.parameters() if p.requires_grad)}")
        self.logger.info(f"No. of trainable parameters in the connector: {sum(p.numel() for p in self.policy_model.model.connector.parameters() if p.requires_grad)}")
        self.logger.info(f"No. of trainable parameters in the lm head: {sum(p.numel() for p in self.policy_model.lm_head.parameters() if p.requires_grad)}")


        progbar = tqdm(total=num_iterations, desc="Training", unit="step")
        step = 1
        
        while step <= num_iterations:
            train_batch = next(iter(train_dataloader))
            train_loss = self.train_step(train_batch, step=step)
            # self.logger.info(f"Step: {step}, Train Loss: {train_loss:.4f}")
            step += 1
            progbar.update(1)
            progbar.set_postfix({"train_loss": train_loss})


            if step % 100 == 0:
                valid_loss = self.valid_step(valid_dataloader, step=step)
                self.logger.info(f"Step: {step}, Valid Loss: {valid_loss:.4f}")
                progbar.set_postfix({"valid_loss": valid_loss})
        
        progbar.close()


if __name__ == "__main__":    
    model_name = "HuggingFaceTB/SmolVLM-Instruct"

    optim_config = {'learning_rate': 2e-5,
                    'weight_decay': 0.1,
                    'momentum': 0.95,
                    'nesterov': True,
                    'ns_steps': 5,
                    'adamw_betas': (0.9, 0.9999),
                    'adamw_eps': 1e-8,
                    'num_warmup_steps': 10,
                    'num_cycles': 0.5,}

    config = {'enable_wandb_logging': False,
              'gradient_accumulation_steps': 16}

    policy_model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = Trainer.prepare_optimizers(model=policy_model, optim_config=optim_config, type='adamw')

    trainer = SFTTrainer(
        policy_model=policy_model,
        processor=processor,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        config=config
    )

    trainer.add_tokens_for_loc_and_seg_tasks(precision=0.001)   ## add special tokens for localization and segmentation tasks

    lora_r = 256
    lora_alpha = 512
    lora_dropout = 0.05
    lora_config = LoraConfig(
        init_lora_weights='gaussian',
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'lm_head'],
    )

    trainer.policy_model.model.text_model = get_peft_model(trainer.policy_model.model.text_model, lora_config)
    trainer.policy_model.model.text_model.print_trainable_parameters()

    train_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="train", preprocess_fn="refcocog_sft_seg")
    valid_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="validation", preprocess_fn="refcocog_sft_seg")

    train_dataloader = trainer.build_dataloader(train_dataset, batch_size=2)
    valid_dataloader = trainer.build_dataloader(valid_dataset, batch_size=2)

    num_training_epochs = 300
    num_iterations = int(len(train_dataloader) * num_training_epochs)

    trainer.set_lr_scheduler(num_iterations=num_iterations, optim_config=optim_config)

    trainer.train(train_dataloader=train_dataloader,
                  valid_dataloader=valid_dataloader,
                  num_iterations=num_iterations)

    trainer.save_checkpoint(step='9999', save_dir='with-special-tokens')
