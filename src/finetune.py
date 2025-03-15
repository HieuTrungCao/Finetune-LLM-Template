import os
import wandb
import hydra
import rootutils

from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model import LLM
from src.data import LLM_Dataset
from src.utils import convert_list_config_to_list

def train(config):

    training_args_dict = OmegaConf.to_container(config["training_arg"], resolve=True)
    trainer_config_dict = OmegaConf.to_container(config["trainer"], resolve=True)

    # Ensure all nested OmegaConf objects are resolved
    for key, value in training_args_dict.items():
        if isinstance(value, (DictConfig, ListConfig)):
            training_args_dict[key] = OmegaConf.to_container(value, resolve=True)
    for key, value in trainer_config_dict.items():
        if isinstance(value, (DictConfig, ListConfig)):
            trainer_config_dict[key] = OmegaConf.to_container(value, resolve=True)

    model, tokenizer, peft_config = LLM.load_model(config)
    
    if isinstance(peft_config, (DictConfig, dict)):
        peft_config = OmegaConf.to_container(peft_config, resolve=True) if isinstance(peft_config, DictConfig) else peft_config

    llm_dataset = LLM_Dataset(config, tokenizer)
    finetune_dataset, valid_dataset = llm_dataset.get_dataset()


    training_arguments = SFTConfig(**training_args_dict)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=finetune_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
        **trainer_config_dict
    )

    trainer.train()
    
    trainer.save_model(os.path.join(config["training_args"]['output_dir'], "best"))
    
@hydra.main(version_base=None, config_path="../config", config_name="finetune.yaml")
def main(config : DictConfig) -> None:
    train(config)

if __name__ == "__main__":
    main()