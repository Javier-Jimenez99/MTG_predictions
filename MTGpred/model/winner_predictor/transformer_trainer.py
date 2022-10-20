import random

import numpy as np
import torch
import typer
import wandb
from MTGpred.model.dataset import MatchesDataset4
from MTGpred.utils.database import get_all_matches_ids, get_matches_id_by_format
from MTGpred.utils.mtgjson import load_cards_df
from torch import nn
from transformers import (
    BigBirdForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional
import os


def get_space_left():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    return f


@dataclass
class WinnerPredictorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class WinnerPredictor(nn.Module):
    def __init__(
        self,
        encoder_model_name="google/bigbird-roberta-large",
        encoder_output_size=256,
    ):
        super(WinnerPredictor, self).__init__()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        self.encoder = BigBirdForSequenceClassification.from_pretrained(
            encoder_model_name, num_labels=encoder_output_size
        )
        # self.batch_norm = nn.BatchNorm1d(encoder_output_size * 2)
        self.final_classifier = nn.Linear(encoder_output_size * 2, 1)

    @property
    def config(self):
        return self.encoder.config

    def forward(self, deck1, deck2, label=None):
        deck1 = self.encoder(**deck1).logits
        deck2 = self.encoder(**deck2).logits
        x = torch.cat((deck1, deck2), dim=1)
        # x = self.batch_norm(x)
        output = self.final_classifier(x)
        winner = torch.sigmoid(output).squeeze(1)

        if label is not None:
            loss = self.criterion(winner, label.to(torch.float32))
            return WinnerPredictorOutput(loss=loss, logits=winner)
        else:
            return WinnerPredictorOutput(logits=winner)


def accuracy(y_pred, y_true):
    y_pred = y_pred > 0
    y_true = y_true > 0
    return (y_pred == y_true).sum().item() / len(y_pred)


def compute_metrics(pred):
    print(pred)
    labels = pred.label_ids
    preds = pred.predictions
    return {"accuracy": accuracy(preds, labels)}


def collate_decks(decks):
    encoded_deck = {}
    for key in decks[0].keys():
        for deck in decks:
            if key not in encoded_deck:
                encoded_deck[key] = []
            encoded_deck[key].append(deck[key])

    for key in encoded_deck.keys():
        encoded_deck[key] = torch.stack(encoded_deck[key])

    return encoded_deck


def custom_data_collator(features):
    decks1 = []
    decks2 = []
    labels = []
    for f in features:
        decks1.append(f["deck1"])
        decks2.append(f["deck2"])
        labels.append(f["label"])

    decks1 = collate_decks(decks1)
    decks2 = collate_decks(decks2)
    labels = torch.tensor(labels)

    return {"deck1": decks1, "deck2": decks2, "label": labels}


def train(
    split_ratio: float = 0.9,
    batch_size: int = 1,
    epochs: int = 10,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    cards_path: str = "data/AtomicCards.json",
    cuda: bool = True,
    save_path: str = "models/",
    use_wandb: bool = True,
    transformer_model: str = "google/bigbird-roberta-large",
    gradient_accumulation_steps: int = 64,
    checkpoint_path: str = "models/chekpoints/",
    warmup_ratio: float = 0.1,
):
    if use_wandb:
        run = wandb.init(project="MTGpred", reinit=True)
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Load data
    # all_matches_ids = get_all_matches_ids()
    all_matches_ids = get_matches_id_by_format("legacy")
    random.shuffle(all_matches_ids)
    # all_matches_ids = all_matches_ids[:4000]
    split = int(len(all_matches_ids) * split_ratio)
    train_matches_ids = all_matches_ids[:split]
    test_matches_ids = all_matches_ids[split:]

    cards_df = load_cards_df(cards_path)
    train_dataset = MatchesDataset4(
        cards_df, train_matches_ids, model_name=transformer_model
    )
    test_dataset = MatchesDataset4(
        cards_df, test_matches_ids, model_name=transformer_model
    )

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        # "fp16": {
        #    "enabled": True,
        #    "loss_scale": 0,
        #    "loss_scale_window": 1000,
        #    "hysteresis": 2,
        #    "min_loss_scale": 1,
        # },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": True,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "adam_w_mode": True,
                "lr": lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
    }

    # Train model
    model = WinnerPredictor(encoder_model_name=transformer_model)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine_with_restarts",
        save_strategy="epoch",
        save_total_limit=1,
        no_cuda=not cuda,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        report_to="wandb" if use_wandb else None,
        run_name=run.name if use_wandb else None,
        deepspeed=deepspeed_config,
        fp16=True,
        logging_steps=1,
        # sharded_ddp=["zero_dp_3", "offload"],
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )

    trainer.train()
    trainer.save_model(save_path)


if __name__ == "__main__":
    typer.run(train)
