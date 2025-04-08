import datasets
from datasets import Dataset
from datasets.exceptions import DatasetNotFoundError
from typing import Callable
from transformers.image_utils import load_image
import random
import ast

from transformers import AutoProcessor
import numpy as np

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
float_numbers = np.arange(0., 1 + 0.001, 0.001)
tokens_for_loc_and_seg_tasks = [f"{num:.{len(str(0.001).split('.')[-1])}f}" for num in float_numbers]
tokens_for_loc_and_seg_tasks += [', ']
processor.tokenizer.add_tokens(tokens_for_loc_and_seg_tasks)



max_tokens_loaded = 0


class PreprocessorForLocalizationAndSegmentation:
    
    @staticmethod
    def preprocess(dataset_name: str, split: str="train", preprocess_fn: str="refcocog_sft_seg"):
        preprocess_fn_dict = {"refcocog_sft_seg": PreprocessorForLocalizationAndSegmentation._refcocog_sft_seg}
        try:
            return preprocess_fn_dict.get(preprocess_fn)(dataset_name=dataset_name, split=split)
        except DatasetNotFoundError:
            raise

    @staticmethod
    def system_prompt(mapped_dataset_name: str = "refcocog_sft_seg"):
        match mapped_dataset_name:
            case "refcocog_sft_seg":
                return (f"A conversation between a user and assistant whose expertise is in understanding images and performing precise segmentation and object localization. "
                        f"The user will provide an image and ask a question about the objects present in it. "
                        f"Your task is to identify the objects requested by the user and provide their precise boundaries as a list of polygon points in the format [x1, y1, x2, y2, ...]. "
                        f"Each number in the list represents a coordinate, with x representing the horizontal position and y representing the vertical position. "
                        f"If the user asks for multiple objects, provide the polygon points for each object as a separate list within a main list. "
                        f"Ensure that all x and y coordinates are normalized to the range of 0.0 to 1.0, representing the relative position within the image dimensions (0.0 being the minimum and 1.0 being the maximum for both x and y). "
                        f"If no objects matching the user's query are found in the image, respond with an empty list: []. "
                        f"If the user's request is ambiguous or cannot be fulfilled, respond with a message indicating the issue, but prioritize providing the polygon points if possible.")
            case _:
                raise ValueError('Not implemented yet.')

    @staticmethod
    def _refcocog_sft_seg(dataset_name: str, split: str="train"):
        dataset = datasets.load_dataset(dataset_name, split=f"{split}")

        def process_example(x):
            global max_tokens_loaded
            system_message = {
                "role": "system",
                "content": [
                    {"type": "text", "text": PreprocessorForLocalizationAndSegmentation.system_prompt("refcocog_sft_seg")}
                ]
            }
            user_message_content = [{"type": "image"}]     # change this to paths

            annots = x['sentences']
            len_annots = len(x['sentences'])
            
            if len_annots > 2:
                chosen_sent_id = random.randint(a=0, b=len_annots-1)
            else:   chosen_sent_id = 0
            chosen_annot = annots[chosen_sent_id]['sent']

            user_message_content.append({"type": "text", "text": f"<id_1>{chosen_annot}</id_1>"})
            user_message = {
                "role": "user",
                "content": user_message_content
            }

            seg_annots = ast.literal_eval(x['raw_anns'])['segmentation']
            image_width = ast.literal_eval(x['raw_image_info'])['width']
            image_height = ast.literal_eval(x['raw_image_info'])['height']

            if len(seg_annots) > 1:
                normalized_seg_annots = ""
                for i, seg_annot in enumerate(seg_annots):
                    normalized_seg_annot = [round(polypoint/image_width, 3) if polypoint % 2 != 0 else round(polypoint/image_height, 3) for polypoint in seg_annot]
                    normalized_seg_annot = f"<id_1_seg_{i+1}>{normalized_seg_annot}</id_1_seg_{i+1}>".replace('[', '').replace(']', '')
                    normalized_seg_annots += normalized_seg_annot
            else:
                seg_annots = seg_annots[0]
                normalized_seg_annots = [round(polypoint/image_width, 3) if polypoint % 2 != 0 else round(polypoint/image_height, 3) for polypoint in seg_annots]
                normalized_seg_annots = f"<id_1_seg_1>{normalized_seg_annots}</id_1_seg_1>".replace('[', '').replace(']', '')

            asst_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": normalized_seg_annots}
                ]
            }

            messages = [system_message, user_message, asst_message]
            full_prompt_with_answer = processor.apply_chat_template(messages, tokenize=False)
            token_count = len(processor.tokenizer.encode(full_prompt_with_answer))
            if token_count >= max_tokens_loaded:
                max_tokens_loaded = token_count

            return {"messages": [system_message, user_message, asst_message]}
        dataset = dataset.map(process_example, num_proc=1, batched=False)
        return dataset

if __name__ == '__main__':
    print(PreprocessorForLocalizationAndSegmentation.preprocess(dataset_name="jxu124/refcocog", split="train", preprocess_fn="refcocog_sft_seg"))
    print(f"max tokens encoded: {max_tokens_loaded}")