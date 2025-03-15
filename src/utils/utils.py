from omegaconf import ListConfig

def convert_list_config_to_list(obj):
    if isinstance(obj, ListConfig):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_list_config_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_list_config_to_list(elem) for elem in obj]
    return obj