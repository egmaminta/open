import datasets
from datasets.exceptions import DatasetNotFoundError
from transformers.image_utils import load_image
import random
import ast
import os
import gc


class PreprocessorForLocalizationAndSegmentation:
    """Sample usage:
    ```python
    from utils import PreprocessorForLocalizationAndSegmentation
    dataset = PreprocessorForLocalizationAndSegmentation.preprocess(dataset_name="jxu124/refcocog", split="train", preprocess_fn="refcocog_sft_seg")
    ```
    This will load the dataset and preprocess it using the specified function.
    """
    
    @staticmethod
    def preprocess(dataset_name: str, split: str="train", preprocess_fn: str="refcocog_sft_seg"):
        assert preprocess_fn is not None, "preprocess_fn cannot be None."
        
        preprocess_fn_dict = {"refcocog_sft_seg": PreprocessorForLocalizationAndSegmentation._refcocog_sft_seg}
        try:
            return preprocess_fn_dict.get(preprocess_fn)(dataset_name=dataset_name, split=split)
        except DatasetNotFoundError:
            raise

    @staticmethod
    def system_prompt(mapped_dataset_name: str = "refcocog_sft_seg"):
        match mapped_dataset_name:
            case "refcocog_sft_seg":
                return (f"A conversation between a user and an assistant. You are an assistant that assists and performs the task provided by the user. "
                        f"The user will provide you with a task using <task> tag that is specific to the task to be performed and an input image.")
            case _:
                raise ValueError('Not implemented yet.')

    @staticmethod
    def _refcocog_sft_seg(dataset_name: str, split: str="train"):
        if split == "train":
            dataset = datasets.load_dataset(dataset_name, split=f"{split}")
        elif split == "test" or split == "validation":
            dataset = datasets.load_dataset(dataset_name, split=f"{split}[:200]")

        def process_example(x):
            
            system_message = {
                "role": "system",
                "content": [
                    {"type": "text", "text": PreprocessorForLocalizationAndSegmentation.system_prompt("refcocog_sft_seg")}
                ]
            }
            user_message_content = [{"type": "image"}]

            annots = x['sentences']
            len_annots = len(x['sentences'])
            
            image_path = x['file_name']
            image_path = "_".join(image_path.split('_')[:-1]) + ".jpg"

            test_mode = False
            if not test_mode:
                full_image_path = os.path.join('/data/vlm/playground/data/coco/train2014', image_path)
            elif test_mode:
                full_image_path = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            
            if len_annots > 2:
                chosen_sent_id = random.randint(a=0, b=len_annots-1)
            else:   chosen_sent_id = 0
            chosen_annot = annots[chosen_sent_id]['sent']

            user_message_content.append({"type": "text", "text": f"<task:segmentation>{chosen_annot}"})
            user_message = {
                "role": "user",
                "content": user_message_content
            }

            seg_annots = ast.literal_eval(x['raw_anns'])['segmentation']
            image_width = ast.literal_eval(x['raw_image_info'])['width']
            image_height = ast.literal_eval(x['raw_image_info'])['height']

            if len(seg_annots) > 1:
                normalized_seg_annots = ""
                for r, seg_annot in enumerate(seg_annots):
                    normalized_seg_annot = [round(polypoint/image_width, 3) if polypoint % 2 != 0 else round(polypoint/image_height, 3) for polypoint in seg_annot]
                    for idx, seg_annot in enumerate(normalized_seg_annot):
                        if seg_annot == 1:
                            normalized_seg_annot[idx] = '<seg1000>'
                        else:
                            normalized_seg_annot[idx] = '<seg' + f"{seg_annot:.3f}".split('.')[-1] + '>'
                    normalized_seg_annot = ('<seg_r>'+ ''.join(normalized_seg_annot) + '</seg_r>').replace('[', '').replace(']', '').replace(', ', '')
                    normalized_seg_annots += normalized_seg_annot
                normalized_seg_annots = normalized_seg_annots + chosen_annot
            else:
                seg_annots = seg_annots[0]
                normalized_seg_annots = [round(polypoint/image_width, 3) if polypoint % 2 != 0 else round(polypoint/image_height, 3) for polypoint in seg_annots]
                for idx, seg_annot in enumerate(normalized_seg_annots):
                    if seg_annot == 1:
                        normalized_seg_annots[idx] = '<seg1000>'
                    else:
                        normalized_seg_annots[idx] = '<seg' + f"{seg_annot:.3f}".split('.')[-1] + '>'
                normalized_seg_annots = ('<seg_r>' + ''.join(normalized_seg_annots) + '</seg_r>').replace('[', '').replace(']', '').replace(', ', '')
                normalized_seg_annots = normalized_seg_annots + chosen_annot

            asst_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": normalized_seg_annots}
                ]
            }

            del image_path, seg_annots
            gc.collect()

            return {"messages": [system_message, user_message, asst_message],
                    "img_new_path": full_image_path,
                    "img_width": image_width,
                    "img_height": image_height,
                    "task_target": normalized_seg_annots,
                    "chosen_annot": chosen_annot,
                    "pil_img": load_image(full_image_path)}
        
        dataset = dataset.map(process_example, num_proc=8, batched=False)
        chosen_cols = ['messages', 'img_new_path', 'img_width', 'img_height', 'task_target', 'chosen_annot', 'pil_img']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in chosen_cols])
        return dataset
