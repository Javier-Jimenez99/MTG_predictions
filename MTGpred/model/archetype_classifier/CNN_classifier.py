import random
import numpy as np

import torch
import typer
import wandb
from MTGpred.model.dataset import DecksDataset
from MTGpred.utils.database import get_all_decks
from MTGpred.utils.mtgjson import load_cards_df
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional
import os
from typing import Iterable


def get_space_left():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    return f


@dataclass
class ArchetypeClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class EncoderCNN(nn.Module):
    def __init__(
        self,
        input_shape: Iterable[int] = (100, 128),
        output_size: int = 256,
    ):
        super(EncoderCNN, self).__init__()
        input_shape_array = np.array(input_shape)

        # Suppose that decks are 2D tensors of shape (1,100,128) and output should be (256)
        self.conv1 = nn.Conv2d(1, 16, 3, padding="same")  # (16,100,128)
        self.maxpool1 = nn.MaxPool2d(2)  # (16,50,64)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")  # (32,50,64)
        self.maxpool2 = nn.MaxPool2d(2)  # (32,25,32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")  # (64,25,32)
        self.maxpool3 = nn.MaxPool2d(2)  # (64,12,16)
        self.conv4 = nn.Conv2d(64, 128, 3, padding="same")  # (128,12,16)
        self.maxpool4 = nn.MaxPool2d(2)  # (128,6,8)
        self.conv5 = nn.Conv2d(128, 256, 3, padding="same")  # (256,6,8)
        self.maxpool5 = nn.MaxPool2d(2)  # (256,3,4)
        self.flatten = nn.Flatten()  # (3072)
        flatten_output_size = np.prod(input_shape_array // 2**5) * 256
        self.last_layer = nn.Linear(flatten_output_size, output_size)  # (256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.last_layer(x)

        return x


class DeckEncoder(nn.Module):
    def __init__(
        self,
        transformer_model_name: str = "xlnet-base-cased",
        encoder_output_size: int = 256,
        deck_size: int = 100,
        transformer_output_size: int = 128,
    ):
        super(DeckEncoder, self).__init__()
        self.transformer_output_size = transformer_output_size
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_model_name, num_labels=transformer_output_size
        )
        self.deck_size = deck_size
        self.CNN = EncoderCNN(
            input_shape=(deck_size, transformer_output_size),
            output_size=encoder_output_size,
        )

    @property
    def config(self):
        return self.transformer.config

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return "cuda"
        else:
            return "cpu"

    def forward(self, deck, mask_length):
        cards_encoded = (
            # torch.zeros((*deck["input_ids"].shape[:2], self.transformer_output_size))
            torch.zeros(
                (
                    deck["input_ids"].shape[0],
                    self.deck_size,
                    self.transformer_output_size,
                )
            )
            .to(self.get_device())
            .half()
        )

        # Build a matrix mask for the transformer
        # matrix_mask = torch.zeros(deck["input_ids"].shape[:2]).to(self.get_device())\
        matrix_mask = torch.zeros(deck["input_ids"].shape[0], self.deck_size).to(
            self.get_device()
        )
        for i, length in enumerate(mask_length):
            matrix_mask[i, :length] = torch.ones(length)

        for sequence_index in range(deck["input_ids"].shape[1]):
            batch_index = matrix_mask[:, sequence_index].bool()

            if batch_index.sum() == 0:
                break

            encoding_data = {}
            for k in deck.keys():
                encoding_data[k] = deck[k][batch_index, sequence_index, :]

            cards_encoded[batch_index, sequence_index, :] = self.transformer(
                **encoding_data
            )["logits"]

        cards_encoded = cards_encoded.unsqueeze(1)
        encoded_deck = self.CNN(cards_encoded)
        return encoded_deck


class ArchetypeClassifier(nn.Module):
    def __init__(
        self,
        transformer_model_name="xlnet-base-cased",
        encoder_output_size=256,
        transformer_output_size: int = 128,
        num_labels: int = 3,
    ):
        super(ArchetypeClassifier, self).__init__()
        if num_labels < 2:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.encoder = DeckEncoder(
            transformer_model_name=transformer_model_name,
            encoder_output_size=encoder_output_size,
            transformer_output_size=transformer_output_size,
        )
        self.final_classifier = nn.Linear(encoder_output_size, num_labels)

    @property
    def config(self):
        return self.encoder.config

    def forward(self, deck, mask_length, label=None):
        deck_encoded = self.encoder(deck, mask_length)
        output = self.final_classifier(deck_encoded)
        winner = torch.sigmoid(output).squeeze(1)

        if label is not None:
            loss = self.criterion(winner, label)
            return ArchetypeClassifierOutput(loss=loss, logits=winner)
        else:
            return ArchetypeClassifierOutput(logits=winner)


def accuracy(y_pred, y_true):
    y_pred = y_pred > 0
    y_true = y_true > 0
    return (y_pred == y_true).sum().item() / len(y_pred)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    return {"accuracy": accuracy(preds, labels)}


def custom_data_collator(dataset_elements):
    keys = dataset_elements[0]["deck"][0].keys()
    max_lengths = torch.tensor([deck["mask_length"] for deck in dataset_elements])
    max_length = max_lengths.max().item()

    encoded_data = {}
    for k in keys:
        decks = []
        for dataset_element in dataset_elements:
            cards = []
            for card in dataset_element["deck"]:
                cards.append(card[k])

            cards.extend([torch.zeros(1, 256)] * (max_length - len(cards)))

            decks.append(torch.cat(cards, dim=0))

        encoded_data[k] = torch.stack(decks, dim=0).to(torch.long)

    return {
        "deck": encoded_data,
        "mask_length": max_lengths,
        "label": torch.tensor([deck["label"] for deck in dataset_elements]),
    }


def train(
    split_ratio: float = 0.9,
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    cards_path: str = "data/AtomicCards.json",
    cuda: bool = True,
    save_path: str = "models/",
    use_wandb: bool = True,
    transformer_model: str = "bert-base-cased",
    gradient_accumulation_steps: int = 128,
    checkpoint_path: str = "models/chekpoints/",
    warmup_ratio: float = 0.1,
    fp16: bool = True,
):
    if use_wandb:
        run = wandb.init(project="MTGpred", reinit=True)
        os.environ["WANDB_DISABLED"] = "false"
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Load data
    all_decks_data = list(get_all_decks(with_archetype=True))
    random.shuffle(all_decks_data)
    split = int(len(all_decks_data) * split_ratio)
    train_decks = all_decks_data[:split]
    test_decks = all_decks_data[split:]

    cards_df = load_cards_df(cards_path)
    train_dataset = DecksDataset(
        cards_df,
        train_decks,
        model_name=transformer_model,
        labels=["aggro", "control", "combo"],
    )
    test_dataset = DecksDataset(
        cards_df,
        test_decks,
        model_name=transformer_model,
        labels=["aggro", "control", "combo"],
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
    model = ArchetypeClassifier(transformer_model_name=transformer_model)

    # if fp16:
    #    model = model.half()

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
        # deepspeed=deepspeed_config,
        fp16=fp16,
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
