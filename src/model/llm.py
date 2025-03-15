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
            load_in_4bit=config["qlora"]["load_in_4bit"],
            bnb_4bit_quant_type=config["qlora"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=config["qlora"]["bnb_4bit_use_double_quant"]
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            quantization_config=bnb_config,
            device_map=config["model"]["device_map"],
            attn_implementation=config["model"]["attn_implementation"]
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        tokenizer.chat_template = None
        model, tokenizer = setup_chat_format(model, tokenizer)

        # LoRA config
        peft_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias=config["lora"]['bias'],
            task_type=config["lora"]["task_type"],
            target_modules=config["lora"]["target_modules"]
        )
        model = get_peft_model(model, peft_config)

        return model, tokenizer, peft_config