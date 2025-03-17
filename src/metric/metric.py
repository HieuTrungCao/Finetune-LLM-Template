import numpy as np
import evaluate
import hydra

from omegaconf import DictConfig
from transformers import AutoTokenizer

@hydra.main(version_base=None, config_path="../../config", config_name="finetune.yaml")
def get_config(conf : DictConfig) -> DictConfig:
    global config
    config = conf
    # return config

get_config()

tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

bleu = evaluate.load("bleu")


def compute_bleu(eval_preds):
    preds, labels = eval_preds
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in labels (used for padding) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute BLEU score
    results = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    return {"bleu": results["bleu"]}