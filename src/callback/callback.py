import wandb

from transformers import TrainerCallback, Trainer

from src.metric import compute_bleu

class TrainBLEUCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer: Trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {state.epoch}: Computing BLEU on training dataset...")
        train_results = self.trainer.predict(self.trainer.train_dataset)
        train_metrics = compute_bleu((train_results.predictions, train_results.label_ids))
        bleu_score = train_metrics["bleu"]
        print(f"Training BLEU score: {bleu_score:.4f}")
        wandb.log({"train_bleu": bleu_score, "epoch": state.epoch})
