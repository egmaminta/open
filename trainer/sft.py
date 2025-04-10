from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
import wandb
from typing import List, Dict
from datasets import Dataset

from .trainer import Trainer
from .utils import PreprocessorForLocalizationAndSegmentation


class SFTTrainer(Trainer):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: List[torch.optim.Optimizer],
                 device: str,
                 config: Dict):
        
        super().__init__(policy_model=policy_model,
                         processor=processor,
                         tokenizer=tokenizer,
                         optimizer=optimizer,
                         device=device,
                         config=config)


    def prepare_dataset(self, dataset_name: str, split: str="train", preprocess_fn: str='refcocog_sft_seg'):
        assert preprocess_fn is not None, "Preprocess function cannot be None."
        
        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset_name=dataset_name,
            split=split,
            preprocess_fn=preprocess_fn
        )

    def collate_fn(self, batch, allowed_max_length: int=8000):
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
            _pixel_values = _system_and_user_message_inputs["pixel_values"][0].squeeze(0)
            _pixel_attention_mask = _system_and_user_message_inputs["pixel_attention_mask"][0].squeeze(0)

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
            _input_ids_padded = torch.cat((_input_ids, _pad_tokens), dim=0)

            _inputs = _input_ids_padded[:-1].clone()
            _labels = _input_ids_padded[1:].clone()    ## shift the input ids by 1 to get the labels
            _mask_for_labels = _labels == self.processor.tokenizer.pad_token_id
            pad_token_indices = torch.nonzero(_mask_for_labels).squeeze(0)
            
            if pad_token_indices.numel() > 1:
                _labels[pad_token_indices] = -100   ## set the pad token to -100 (ignore)

            _attn_mask = _inputs != self.processor.tokenizer.pad_token_id   ## create the attention mask
            _attn_mask = _attn_mask.long()

            _labels[:len(_system_and_user_message_input_ids)-1] = -100   ## set the instruction tokens to -100 (ignore)

            if _inputs.shape[0] > allowed_max_length:
                _inputs = _inputs[:allowed_max_length]
                _labels = _labels[:allowed_max_length]
                _attn_mask = _attn_mask[:allowed_max_length]

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


    def train_step(self, batch, step, num_iterations, optim_config):  ##  batch = train_loader
        self.policy_model.train()

        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)

        logits = self.policy_model(**inputs).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100) / self.config.get('gradient_accumulation_steps')
        loss.backward()
        step_loss = loss.item()

        for opt in self.optimizer:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * self.get_lr(step=step, num_iterations=num_iterations, optim_config=optim_config)

        for group in self.optimizer[1].param_groups:    ## muon momentum
            frac = min(step / 300, 1)   ## momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        if (step + 1) % self.config.get('gradient_accumulation_steps') == 0:
            for opt in self.optimizer:  ## optimizer step
                opt.step()

            self.policy_model.zero_grad(set_to_none=True)   ## zero grad

            if self.config.get('enable_wandb_logging'):
                lr_dict_recorder = {}
                
                for opt in self.optimizer:
                    for group in opt.param_groups:
                        lr_dict_recorder[f"{opt}_{group}_lr"] = group["lr"]

                to_log = {'train_loss': step_loss, 'step': step}
                
                for k, v in lr_dict_recorder.items():
                    to_log[k] = v

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

            if self.config.get('enable_wandb_logging'):
                wandb.log({'valid_loss': valid_loss, 'step': step})
        
        self.policy_model.train()
        
        return valid_loss

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, num_epochs: int, optim_config: Dict):
        
        train_dataloader = self.build_dataloader(train_dataset, batch_size=2)
        valid_dataloader = self.build_dataloader(valid_dataset, batch_size=2)

        num_iterations = int(len(train_dataloader) * num_epochs)

        for step, _ in enumerate(range(num_iterations), start=1):
            for batch in train_dataloader:
                train_loss = self.train_step(batch, step=step, num_iterations=num_iterations, optim_config=optim_config)

            if step % 1000 == 0:
                valid_loss = self.valid_step(valid_dataloader, step=step)


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

    optim_config = {'cooldown_frac': 0.20,
                    'lm_head_params_lr': 2.666e-5,
                    'embed_params_lr': 1.777e-5,
                    'scalar_params_lr': 1.666e-5,
                    'hidden_matrix_params_lr': 2.777e-5,
                    'adamw_betas': (0.85, 0.95),
                    'muon_momentum': 0.95}

    config = {'enable_wandb_logging': False,
              'gradient_accumulation_steps': 16}

    policy_model = AutoModelForVision2Seq.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cpu'
    optimizers = Trainer.prepare_optimizers(model=policy_model, optim_config=optim_config)

    trainer = SFTTrainer(
        policy_model=policy_model,
        processor=processor,
        tokenizer=tokenizer,
        optimizer=optimizers,
        device=device
    )

    trainer.add_tokens_for_loc_and_seg_tasks(precision=0.001)

    train_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="train", preprocess_fn="refcocog_sft_seg")
    valid_dataset = trainer.prepare_dataset(dataset_name="jxu124/refcocog", split="validation", preprocess_fn="refcocog_sft_seg")

    trainer.train(train_dataset, valid_dataset, num_epochs=2, optim_config=optim_config)