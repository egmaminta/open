import torch
from transformers import AutoModelForVision2Seq
import torch.nn.functional as F
from typing import Tuple, Literal, Dict, Iterator
from loguru import logger


def get_batch(dataloader: torch.utils.data.DataLoader, curr_iter: Iterator[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    try:
        batch = next(curr_iter)
    except StopIteration:
        ## reset the iterator
        curr_iter = iter(dataloader)
        batch = next(curr_iter)
    return batch, curr_iter

def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_mask = (labels != -100).float()
    pseudo_labels = labels.clone()
    pseudo_labels[labels == -100] = 0   ## set -100 labels to 0 so that we can compute the log probs
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=2, index=pseudo_labels.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)
    log_probs = log_probs * labels_mask

    del pseudo_labels

    return log_probs, labels_mask

def configure_connector(model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    for param in model.model.connector.parameters():
        param.requires_grad = True
    for param in model.model.text_model.parameters():
        param.requires_grad = False
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

    model.model.connector.train()
    model.model.text_model.eval()
    model.model.vision_model.eval()
    model.lm_head.eval()

    logger.info('Connector is set to train mode.')
    logger.info('Language model, Vision model, and LM head are set to eval mode.')

def configure_text(model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    for param in model.model.connector.parameters():
        param.requires_grad = False
    for param in model.model.text_model.parameters():
        param.requires_grad = True
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    model.model.connector.eval()
    model.model.text_model.train()
    model.model.vision_model.eval()
    model.lm_head.train()

    logger.info('Language model, Embedding layer, and LM head are set to train mode.')
    logger.info('Connector and Vision model are set to eval mode.')

def configure_embeds(model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    for param in model.model.connector.parameters():
        param.requires_grad = False
    for param in model.model.text_model.parameters():
        param.requires_grad = False
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.model.text_model.embed_tokens.parameters():
        param.requires_grad = True

    model.model.connector.eval()
    model.model.text_model.eval()
    model.model.vision_model.eval()
    model.model.text_model.embed_tokens.train()
    model.lm_head.train()

    logger.info('Embedding layer and LM head are set to train mode.')
    logger.info('Connector, Language model, and Vision model are set to eval mode.')

def configure_connector_text(model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    for param in model.model.connector.parameters():
        param.requires_grad = True
    for param in model.model.text_model.parameters():
        param.requires_grad = True
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    model.model.connector.train()
    model.model.text_model.train()
    model.model.vision_model.eval()
    model.lm_head.train()

    logger.info('Connector, Language model, and LM head are set to train mode.')
    logger.info('Vision model is set to eval mode.')

def configure_all(model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    for param in model.parameters():
        param.requires_grad = True

    model.train()

    logger.info('Vision model, Language model, Connector, and LM head are set to train mode.')
    logger.info('All parameters are set to require gradients.')

def configure_finetune_mode(finetune_mode: Literal['connector', 'text', 'connector_text', 'embeds', 'all'], model: torch.nn.Module | AutoModelForVision2Seq) -> None:
    match finetune_mode:
        case 'connector':
            configure_connector(model)
        case 'text':
            configure_text(model)
        case 'connector_text':
            configure_connector_text(model)
        case 'embeds':
            configure_embeds(model)
        case 'all':
            configure_all(model)
        case _:
            raise ValueError(f'Invalid finetune mode: {finetune_mode}. Expected one of ["connector", "text", "connector_text", "embeds"].')

    logger.info(f'Finetune mode configured: {finetune_mode}')
