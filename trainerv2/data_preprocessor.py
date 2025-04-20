import datasets
from datasets.exceptions import DatasetNotFoundError
import random
import ast
import os
from typing import Literal, Dict, Any


_LOCALIZATION_AND_SEGMENTATION_DATASETS_ = {
    'REFCOCO': 'jxu124/refcoco',
    'REFCOCOG': 'jxu124/refcocog',
    'REFCOCOPLUS': 'jxu124/refcocoplus',
}

def normalize_segmentation(segmentation, width, height):
    normalized = [
        round(coord / height, 3) if i % 2 != 0 else round(coord / width, 3)
        for i, coord in enumerate(segmentation)
    ]
    return [
        f"<seg1000>" if coord == 1 else f"<seg{str(coord).split('.')[-1]:0>3}>"
        for coord in normalized
    ]

def normalize_bbox(bbox, width, height):
    normalized = [
        round(coord / height, 3) if i % 2 != 0 else round(coord / width, 3)
        for i, coord in enumerate(bbox)
    ]
    return [
        f"<loc1000>" if coord == 1 else f"<loc{str(coord).split('.')[-1]:0>3}>"
        for coord in normalized
    ]

def format_normalized_segmentation(normalized):
    return f"<seg_r>{''.join(normalized)}</seg_r>"

def format_normalized_bbox(bbox):
    return f"<loc_r>{''.join(bbox)}</loc_r>"


def process_sample(sample: Dict[str, Any], task: str) -> Dict[str, Any]:
    if task == 'seg':
        system_message = {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': PreprocessorForLocalizationAndSegmentation.get_system_prompt('seg')}
            ]
        }
    elif task == 'loc':
        system_message = {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': PreprocessorForLocalizationAndSegmentation.get_system_prompt('loc')}
            ]
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    user_message_content = [{'type': 'image'}]
    annot_descr = sample['sentences']
    len_annots = len(annot_descr)

    image_path = sample['file_name']
    image_path = '_'.join(image_path.split('_')[:-1]) + '.jpg'

    test_mode = False   ## NOQA: test mode
    if not test_mode:   full_image_path = os.path.join('/data/vlm/playground/data/coco/train2014', image_path)
    elif test_mode:   full_image_path = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    if len_annots > 2:  chosen_sent_id = random.randint(a=0, b=len_annots-1)
    else:   chosen_sent_id = 0
    chosen_annot = annot_descr[chosen_sent_id]['sent']

    if task == 'seg':   user_message_content.append({'type': 'text', 'text': f'<task:segmentation>{chosen_annot}'})
    elif task == 'loc':   user_message_content.append({'type': 'text', 'text': f'<task:localization>{chosen_annot}'})
    else:   raise ValueError(f"Unsupported task: {task}")

    user_message = {'role': 'user', 'content': user_message_content}
    
    image_width = ast.literal_eval(sample['raw_image_info'])['width']
    image_height = ast.literal_eval(sample['raw_image_info'])['height']

    if task == 'seg':   annots = ast.literal_eval(sample['raw_anns'])['segmentation']
    elif task == 'loc':   annots = [ast.literal_eval(sample['raw_anns'])['bbox']]

    if task == 'seg':
        if len(annots) > 1:
            normalized_annots = ""
            for annot in annots:
                normalized = normalize_segmentation(annot, image_width, image_height)
                normalized_annots += format_normalized_segmentation(normalized)
            normalized_annots += chosen_annot
        else:
            annot = annots[0]
            normalized = normalize_segmentation(annot, image_width, image_height)
            normalized_annots = format_normalized_segmentation(normalized) + chosen_annot
    elif task == 'loc':
        if len(annots) > 1:
            normalized_annots = ""
            for annot in annots:
                normalized = normalize_bbox(annot, image_width, image_height)
                normalized_annots += format_normalized_bbox(normalized)
            normalized_annots += chosen_annot
        else:
            annot = annots[0]
            normalized = normalize_bbox(annot, image_width, image_height)
            normalized_annots = format_normalized_bbox(normalized) + chosen_annot
    else:   raise ValueError(f"Unsupported task: {task}")
    
    asst_message = {'role': 'assistant', 'content': [{'type': 'text', 'text': normalized_annots}]}

    return {
        'sys_user_message': [system_message, user_message],
        'asst_message': asst_message,
        'image_path': full_image_path,
        'task_target': normalized_annots,
    }

class PreprocessorForLocalizationAndSegmentation(object):
    """Sample usage:
    ```python
    from open.trainerv2.data_preprocessor import PreprocessorForLocalizationAndSegmentation
    dataset = PreprocessorForLocalizationAndSegmentation.preprocess(
        dataset='REFCOCOG',
        split='train',
        preprocess_fn='refcocog_sft_seg'
    )
    ```"""

    @staticmethod
    def preprocess(
        dataset: Literal['REFCOCO', 'REFCOCOG', 'REFCOCOPLUS'],
        split: str = 'train',
        preprocess_fn: str = 'refcocog_sft_seg'
    ):
        assert dataset is not None, 'dataset_name cannot be None.'
        assert dataset in _LOCALIZATION_AND_SEGMENTATION_DATASETS_.keys(), f'dataset_name must be one of {_LOCALIZATION_AND_SEGMENTATION_DATASETS_.keys()}.'
        dataset = _LOCALIZATION_AND_SEGMENTATION_DATASETS_[dataset]

        assert split is not None, 'split cannot be None.'
        assert preprocess_fn is not None, 'preprocess_fn cannot be None.'

        preprocess_fn_dict = {
            'refcocog_sft_seg': PreprocessorForLocalizationAndSegmentation._refcocog_sft_seg,
            'refcocog_sft_loc': PreprocessorForLocalizationAndSegmentation._refcocog_sft_loc,
        }

        try:    return preprocess_fn_dict.get(preprocess_fn)(dataset=dataset, split=split)
        except DatasetNotFoundError:    raise DatasetNotFoundError(f"Dataset not found for {dataset} with split {split}.")

    @staticmethod
    def get_system_prompt(
        task: Literal['loc', 'seg'] = 'seg'
    ) -> str:
        match task:
            case 'seg':
                return (f"A conversation between user and assistant. You will perform an image segmentation task. "
                        f"The user will provide you an input image and a query. Your goal is to output polygon points "
                        f"that outline the object in the image that corresponds to the query.")
            case 'loc':
                return (f"A conversation between user and assistant. You will perform an image localization task. "
                        f"The user will provide you an input image and a query. Your goal is to output the bounding box "
                        f"that contains the object in the image that corresponds to the query.")
            case _:
                raise ValueError('Not implemented yet.')


    @staticmethod
    def _refcocog_sft_seg(
        dataset: str,
        split: str = 'train'
    ) -> datasets.Dataset:
        if "train" in split:
            dataset: datasets.Dataset = datasets.load_dataset(dataset, split=f"{split}")
        elif "test" in split or "validation" in split or "valid" in split:
            dataset: datasets.Dataset = datasets.load_dataset(dataset, split=f"{split}[:100]")    ## NOQA: 100 for testing purposes
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'test', 'validation', 'valid'].")

        dataset_map_fn = lambda x: process_sample(x, task='seg')

        dataset = dataset.map(
            dataset_map_fn,
            num_proc=1,
            batched=False,
        )

        chosen_cols = ['sys_user_message', 'asst_message', 'image_path', 'task_target']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in chosen_cols])
        return dataset

    @staticmethod
    def _refcocog_sft_loc(
        dataset: str,
        split: str = 'train'
    ) -> datasets.Dataset:
        if "train" in split:
            dataset: datasets.Dataset = datasets.load_dataset(dataset, split=f"{split}")
        elif "test" in split or "validation" in split or "valid" in split:
            dataset: datasets.Dataset = datasets.load_dataset(dataset, split=f"{split}[:100]")  ## NOQA: 100 for testing purposes
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'test', 'validation', 'valid'].")

        dataset_map_fn = lambda x: process_sample(x, task='loc')

        dataset = dataset.map(
            dataset_map_fn,
            num_proc=8,
            batched=False,
        )

        chosen_cols = ['sys_user_message', 'asst_message', 'image_path', 'task_target']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in chosen_cols])
        return dataset
