import random

import numpy as np
import torch
import typer
import wandb
from MTGpred.model.dataset import MatchesDataset3
from MTGpred.utils.database import get_all_matches_ids
from MTGpred.utils.mtgjson import load_cards_df
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BigBirdForSequenceClassification,
    AdamW,
    get_cosine_schedule_with_warmup,
)
import deepspeed


def get_space_left():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    return f


class WinnerPredictor(nn.Module):
    def __init__(
        self,
        encoder_model_name="google/bigbird-roberta-large",
        encoder_output_size=256,
    ):
        super(WinnerPredictor, self).__init__()
        self.encoder = BigBirdForSequenceClassification.from_pretrained(
            encoder_model_name, num_labels=encoder_output_size
        )
        self.batch_norm = nn.BatchNorm1d(encoder_output_size * 2)
        self.final_classifier = nn.Linear(encoder_output_size * 2, 1)

    def forward(self, deck1, deck2):
        deck1 = self.encoder(**deck1).logits
        deck2 = self.encoder(**deck2).logits
        x = torch.cat((deck1, deck2), dim=1)
        x = self.batch_norm(x)
        output = self.final_classifier(x)
        winner = torch.tanh(output).squeeze()
        return winner


def accuracy(y_pred, y_true):
    y_pred = y_pred > 0
    y_true = y_true > 0
    return (y_pred == y_true).sum().item() / len(y_pred)


def train(
    split_ratio: float = 0.9,
    batch_size: int = 2,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    cards_path: str = "data/AtomicCards.json",
    cuda: bool = True,
    save_path: str = "models/transformer.pt",
    use_wandb: bool = True,
    transformer_model: str = "google/bigbird-roberta-large",
    gradient_accumulation_steps: int = 64,
    checkpoint_path: str = None,
):
    if use_wandb:
        run = wandb.init(project="MTGpred", reinit=True)

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # Load data
    all_matches_ids = get_all_matches_ids()
    random.shuffle(all_matches_ids)
    # all_matches_ids = all_matches_ids[:4000]
    split = int(len(all_matches_ids) * split_ratio)
    train_matches_ids = all_matches_ids[:split]
    test_matches_ids = all_matches_ids[split:]

    cards_df = load_cards_df(cards_path)
    train_dataset = MatchesDataset3(
        cards_df, train_matches_ids, model_name=transformer_model
    )
    test_dataset = MatchesDataset3(
        cards_df, test_matches_ids, model_name=transformer_model
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # TRAIN CONFIG
    total_num_steps = int(len(train_dataloader) * epochs / gradient_accumulation_steps)
    num_warmup_steps = int(total_num_steps * 0.1)

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {
            "enabled": True,
        },
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
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "Adam",
            "params": {
                "adam_w_mode": True,
                "lr": 3e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
    }

    # Train model
    model = WinnerPredictor(encoder_model_name=transformer_model).to(device)
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_dataloader) * epochs / gradient_accumulation_steps,
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=deepspeed_config,
        lr_scheduler=scheduler,
    )

    if checkpoint_path is not None:
        model_engine.load_checkpoint(checkpoint_path)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f"======= EPOCH {epoch+1}/{epochs} =======")
        train_losses = []
        cum_train_losses = []
        train_accuracies = []
        cum_train_accuracies = []
        for train_step, (deck1, deck2, target) in tqdm(
            enumerate(train_dataloader), desc="TRAIN", total=len(train_dataloader)
        ):
            deck1 = deck1.to(device)
            deck2 = deck2.to(device)
            target = target.to(device).half()
            winner = model_engine(deck1, deck2)
            loss = criterion(winner, target)

            model_engine.backward(loss)

            train_accuracy = accuracy(winner, target)
            train_accuracies.append(train_accuracy)
            train_losses.append(loss.item())
            cum_train_losses.append(loss.item())
            cum_train_accuracies.append(train_accuracy)

            if (train_step + 1) % gradient_accumulation_steps == 0:
                if use_wandb:
                    wandb.log(
                        {
                            "train_steps_loss": np.mean(cum_train_losses),
                            "train_steps_accuracy": np.mean(cum_train_accuracies),
                            "steps_lr": model_engine.get_lr()[0],
                        }
                    )

                cum_train_losses = []
                cum_train_accuracies = []

            model_engine.step()

        print(f"Train loss: {sum(train_losses)/len(train_losses)}")

        if use_wandb:
            wandb.log(
                {
                    "train_loss": sum(train_losses) / len(train_losses),
                    "train_accuracy": sum(train_accuracies) / len(train_accuracies),
                }
            )

        with torch.no_grad():
            test_losses = []
            accuracies = []
            for deck1, deck2, target in tqdm(
                test_dataloader, desc="TEST", total=len(test_dataloader)
            ):
                deck1 = deck1.to(device)
                deck2 = deck2.to(device)
                target = target.to(device).to(torch.float32)
                winner = model_engine(deck1, deck2)
                loss = criterion(winner, target)
                test_losses.append(loss.item())
                accuracies.append(accuracy(winner, target))

            print(f"Test loss: {sum(test_losses)/len(test_losses)}")
            print(f"Test accuracy: {sum(accuracies)/len(accuracies)}")
            if use_wandb:
                wandb.log(
                    {
                        "test_accuracy": sum(accuracies) / len(accuracies),
                        "test_loss": sum(test_losses) / len(test_losses),
                    }
                )

        if checkpoint_path is not None:
            model_engine.save_checkpoint(
                f"{checkpoint_path}/epoch_{epoch+1}.pt", client_state={"epoch": epoch}
            )

    # Save model_engine
    torch.save(model_engine.state_dict(), save_path)


if __name__ == "__main__":
    typer.run(train)
