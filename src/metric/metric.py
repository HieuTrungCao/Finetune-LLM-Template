import numpy as np
import evaluate
import hydra
import torch

from omegaconf import DictConfig
from transformers import AutoTokenizer, EvalPrediction

@hydra.main(version_base=None, config_path="../../config", config_name="finetune.yaml")
def get_config(conf : DictConfig) -> DictConfig:
    global config
    config = conf
    # return config

get_config()

tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_name_or_path"])

bleu = evaluate.load("bleu")


def compute_bleu(eval_preds: EvalPrediction):
    preds, labels = eval_preds
    preds = preds[-1]
    # print("preds: ", preds)
    # print("labels: ", labels)
    # print("preds: ", preds.size())
    # print("labels: ", labels.size())
    # Convert preds to a NumPy array if it’s a list or tensor
    preds = np.argmax(preds, axis=1)
    print("preds: ", preds.shape)
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in labels (used for padding) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("Sample predictions:", decoded_preds[:3])
    print("Sample references:", decoded_labels[:3])
    # Compute BLEU score
    results = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"bleu": results["bleu"]}