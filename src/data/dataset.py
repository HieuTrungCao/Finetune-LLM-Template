import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer
from omegaconf import DictConfig

class LLM_Dataset:

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer: AutoTokenizer = tokenizer

    def load_dataset(self, path: str):
        data = pd.read_csv(path)
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(
            self.format_chat_template,
            num_proc=self.config["data"]["num_proc"]
            )
        
        return dataset
    
    def format_chat_template(self, data):
            
            data["user"] = self.config["data"]['prompt_content']
            for item in self.config["data"]["user"]:
                 data["user"] = data["user"] + "\n" + item + ": " + data[item]

            row_json = [
                {"role": "system", "content": self.config["data"]["prompt_system"]},
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data[self.config["data"]["assistant"]]}
            ]
            
            data["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
            return data
    
    def get_dataset(self):
         self.finetune_dataset = self.load_dataset(self.config["data"]["finetune_path"])
         self.val_dataset = self.load_dataset(self.config["data"]["val_path"])
         return self.finetune_dataset, self.val_dataset