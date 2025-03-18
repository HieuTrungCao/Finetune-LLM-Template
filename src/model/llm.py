import threading
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import setup_chat_format


class LLM:   
    @staticmethod
    def load_model(config):
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.float16,
            **config["qlora"]
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            quantization_config=bnb_config,
            **config["model"]
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_name_or_path"])
        tokenizer.chat_template = None
        model, tokenizer = setup_chat_format(model, tokenizer)

        # LoRA config
        peft_config = LoraConfig(
            **config["lora"]
        )
        model = get_peft_model(model, peft_config)

        return model, tokenizer, peft_config