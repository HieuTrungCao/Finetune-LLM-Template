import os
import hydra
import rootutils
import wandb

from omegaconf import DictConfig
from transformers import TrainingArguments
from trl import SFTTrainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model import LLM
from src.data import LLM_Dataset
from src.utils import convert_list_config_to_list, save_model

def train(config):

    if config.get("log", None) is not None:
        if config["log"]["name"] == "wandb":
            print("Wandb init!")
            wandb.init(project=config["log"]["project_name"])
    
    model, tokenizer, peft_config = LLM.load_model(config)
    
    peft_config = convert_list_config_to_list(peft_config)

    llm_dataset = LLM_Dataset(config, tokenizer)
    finetune_dataset, valid_dataset = llm_dataset.get_dataset()

    training_arguments = hydra.utils.instantiate(config.training_arg)
    
    kwargs = {
        "model": model,
        "train_dataset": finetune_dataset,
        "eval_dataset": valid_dataset,
        "peft_config": peft_config,
        "tokenizer": tokenizer,
        "args": training_arguments
    }

    trainer = hydra.utils.instantiate(config.trainer, kwargs)

    trainer.train()
    
    trainer.save_model(os.path.join(config["training_arg"]['output_dir'], "best"))
    
    save_model(
        config["model"]["pretrained_model_name_or_path"],
        os.path.join(os.path.join(config["training_arg"]['output_dir'], "best")),
        config["training_arg"]["hub_model_id"]
        )
    if config.get("log", None) is not None:
        if config["log"]["name"] == "wandb":
            wandb.finish()

@hydra.main(version_base=None, config_path="../config", config_name="finetune.yaml")
def main(config : DictConfig) -> None:
    train(config)

if __name__ == "__main__":
    main()