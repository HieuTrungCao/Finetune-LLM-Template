from omegaconf import ListConfig, DictConfig

def convert_list_config_to_list(obj):
    if isinstance(obj, dict) or isinstance(obj, DictConfig):
        return {k: convert_list_config_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, ListConfig):
        return [convert_list_config_to_list(elem) for elem in obj]
    
    return obj