from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
from transformers.image_utils import load_image

from .trainer import Trainer
from .utils import PreprocessorForLocalizationAndSegmentation


class SFTTrainer(Trainer):
    def __init__(self,
                 policy_model: AutoModelForVision2Seq,
                 processor: AutoProcessor,
                 tokenizer: AutoTokenizer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str):
        super().__init__(policy_model=policy_model,
                         processor=processor,
                         tokenizer=tokenizer,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         device=device)


    def prepare_dataset(self, dataset_name: str, split: str="train", preprocess_fn: str='refcocog_sft_seg'):
        assert preprocess_fn is not None, "Preprocess function cannot be None."
        
        return PreprocessorForLocalizationAndSegmentation.preprocess(
            dataset_name=dataset_name,
            split=split,
            preprocess_fn=preprocess_fn
        )

    def collate_fn(self, batch, pad_token_id: int, allowed_max_length: int=8000):
        # must return input_ids, labels, and images, attention_mask
        # set max length to 8000
        # return dict(input_ids, labels, pixel_values, attention_mask, pixel_attention_mask)
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        pixel_attention_mask = []
        
        for b in batch:
            b_copy = b.copy()

            ## encode the instructions
            messages = b_copy.pop('messages')
            image = b_copy.pop('pil_img')

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
            _assistant_message_input_ids = _assistant_message_inputs["input_ids"][0][1:]    # remove the <im_start> token from the beginning.

            ## concatenate the system and user input ids with the assistant input ids
            _input_ids = torch.cat((_system_and_user_message_input_ids, _assistant_message_input_ids), dim=0)

            ## pad the input ids to the max length
            _pad_tokens = torch.tensor([self.processor.tokenizer.pad_token_id] * ((allowed_max_length + 1) - _input_ids.shape[0]), dtype=torch.long)
            _input_ids_padded = torch.cat((_input_ids, _pad_tokens), dim=0)

            _inputs = _input_ids_padded[:-1].clone()   ## truncate the last token
            _labels = _input_ids_padded[1:].clone()    ## shift the input ids by 1 to get the labels
            _mask_for_labels = _labels == self.processor.tokenizer.pad_token_id
            indices = torch.nonzero(_mask_for_labels).squeeze(0)
            
            if indices.numel() > 1:
                _labels[indices] = -100   ## set the pad token to -100 (ignore) except the first one

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
                                           num_workers=2,)


    def train_step(self, batch):
        self.policy_model.train()



## What i need
# pixel_values, pixel_attention_mask, input_ids, attention_mask, labels = batch


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    trainer = SFTTrainer(
        policy_model=AutoModelForVision2Seq.from_pretrained(model_name),
        processor=AutoProcessor.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        optimizer=None,
        scheduler=None,
        device='cuda'
    )
    trainer.add_tokens_for_loc_and_seg_tasks(precision=0.001)

    sample_input = "Why is the sky blue?"
    random_image = load_image("/data/ovod/playground/data/coco/train2014/COCO_train2014_000000581857.jpg")
    messages = [
        {'role': 'system', 'content': 'System message content'},
        {'role': 'user', 'content': [{'type': 'text', 'text': sample_input}, {'type': 'image'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': "<id_1_seg_1>0.1, 0.2, 0.3, 0.4</id_1_seg_1>"}]}
    ]
    
    complete_prompt = trainer.processor.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")
    inputs = trainer.processor(text=complete_prompt, images=[random_image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    # print(f"decoded full: {trainer.processor.tokenizer.decode(input_ids)}")
    ## separated
    system_plus_user = trainer.processor.apply_chat_template([messages[0], messages[1]], add_generation_prompt=False, tokenize=False)
    system_plus_user_input_ids = trainer.processor(text=system_plus_user, images=[random_image], return_tensors="pt")
    # print(system_plus_user_input_ids)
    system_plus_user_input_ids = system_plus_user_input_ids["input_ids"][0]
    
    assistant_message = trainer.processor.apply_chat_template([messages[2]], add_generation_prompt=False, tokenize=False)
    assistant_message_input_ids = trainer.processor(text=assistant_message, images=None, return_tensors="pt")
    # print('------------------------------')
    # print(assistant_message_input_ids)
    # print('------------------------------')
    assistant_message_input_ids = assistant_message_input_ids["input_ids"][0]

    print(f"input ids original: {input_ids}")
    print(f"input ids decoded: {trainer.processor.tokenizer.decode(input_ids)}")
    print(f"token 198: {trainer.processor.tokenizer.decode(198)}")

    combined_input_ids = torch.cat((system_plus_user_input_ids, assistant_message_input_ids[1:]), dim=0)    # remove the `<im_start>` token from the beginning.
    
    # print(f"decoded combined: {trainer.processor.tokenizer.decode(combined_input_ids)}")
    assert torch.all(input_ids == combined_input_ids), "error"
    # print(f"pad token = {trainer.processor.tokenizer.pad_token} ;; pad token id = {trainer.processor.tokenizer.pad_token_id}")
    # print("GOT HERE BECAUSE THE PIXEL VALUES WERE EQUAL")
    
    pad_token = [trainer.processor.tokenizer.pad_token_id]
    max_length = 512
    pad_token = torch.tensor(pad_token * ((max_length + 1) - combined_input_ids.shape[0]), dtype=torch.long)
    combined_input_ids = torch.cat((combined_input_ids, pad_token), dim=0)
    
    inputs = combined_input_ids[:-1].clone()   # truncate the last token
    labels = combined_input_ids[1:].clone()    # shift the input ids by 1 to get the labels
    mask_for_labels = labels == trainer.processor.tokenizer.pad_token_id
    indices = torch.nonzero(mask_for_labels).squeeze(0)
    if indices.numel() > 1:
        labels[indices] = -100

    attn_mask = inputs != trainer.processor.tokenizer.pad_token_id   ## create the attention mask
    attn_mask = attn_mask.long()
    print(f"eos_token_id: {trainer.processor.tokenizer.eos_token_id}, pad_token_id: {trainer.processor.tokenizer.pad_token_id}")
    print(f"eos_token: {trainer.processor.tokenizer.eos_token}, pad_token: {trainer.processor.tokenizer.pad_token}")
    
    labels[:len(system_plus_user_input_ids)-1] = -100   ## set the instruction tokens to -100 (ignore)

    for idx, (input_id, label, attn) in enumerate(zip(inputs, labels, attn_mask)):
        if idx != len(inputs) - 1:
            print(f"idx: {idx} || input_id: {input_id} ({trainer.processor.tokenizer.decode(input_id)}) || label: {label} ({trainer.processor.tokenizer.decode(inputs[idx+1])}) || attn: {attn}")
        else:
            print(f"idx: {idx} || input_id: {input_id} ({trainer.processor.tokenizer.decode(input_id)}) || label: {label} ({trainer.processor.tokenizer.decode(inputs[idx])}) || attn: {attn}")
    
    assert len(inputs) == len(attn_mask), "error"
    
    
    
    
    
    
    
    
    
    
    
    # print(trainer.processor.image_processor([random_image], return_tensors="pt").pixel_values[0].to(trainer.device))
    
    # inputs = trainer.processor(text=prompt, images=[random_image], return_tensors="pt").to(trainer.device)
    
    # pixel_values = inputs["pixel_values"][0]
    # print(pixel_values)
    # assert torch.all(pixel_values == trainer.processor.image_processor([random_image], return_tensors="pt").pixel_values[0].to(trainer.device)), "error"
    # print("GOT HERE BECAUSE THE PIXEL VALUES WERE EQUAL")
    # print(f"pixel_values shape: {pixel_values.shape}")
    # pixel_values_mask = inputs["pixel_attention_mask"][0]
    # print(f"pixel_attention_mask shape: {pixel_values_mask.shape}")
    # input_ids = inputs["input_ids"][0]
    # input_ids_list = input_ids.tolist()
    # # make dictionary of count of each token
    # token_count = {}
    # for i in input_ids_list:
    #     if i in token_count:
    #         token_count[i] += 1
    #     else:
    #         token_count[i] = 1
    # print("=====")
    # print(token_count)
    # print("=====")
    # input_ids = torch.unique(input_ids)
    # for i in input_ids:
    #     print(f"token_id: {i}, token: {trainer.tokenizer.decode(i)}")
    
    # print(f'len tokens: {len(inputs["input_ids"][0])}')
    # with torch.no_grad():
    #     outputs = trainer.policy_model(**inputs)
    # logits = outputs.logits
    # print(logits.shape)