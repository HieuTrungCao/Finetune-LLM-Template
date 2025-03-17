import torch 

from omegaconf import ListConfig, DictConfig

def convert_list_config_to_list(obj):
    if isinstance(obj, dict) or isinstance(obj, DictConfig):
        return {k: convert_list_config_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, ListConfig):
        return [convert_list_config_to_list(elem) for elem in obj]
    
    return obj

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels