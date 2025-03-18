import torch 

from omegaconf import ListConfig, DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import setup_chat_format

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
    # pred_ids = torch.argmax(logits[0], dim=-1)
    return logits, labels

def save_model(base_model, new_model_path, huggingface_repo):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model_reload = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
    )

    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model_reload, new_model_path)

    model = model.merge_and_unload()
    model.push_to_hub(huggingface_repo)
    tokenizer.push_to_hub(huggingface_repo)