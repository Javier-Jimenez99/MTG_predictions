import random
from pickle import FALSE

import numpy as np
import torch
import torch.nn.functional as F
import typer
import wandb
from MTGpred.model.dataset import MatchesDataset3
from MTGpred.utils.database import get_all_matches_ids
from MTGpred.utils.mtgjson import load_cards_df
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    BigBirdForSequenceClassification,
    get_cosine_schedule_with_warmup,
)


def get_space_left():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    return f


class WinnerPredictor(nn.Module):
    def __init__(
        self,
        encoder_model_name="google/bigbird-roberta-base",
        encoder_output_size=256,
    ):
        super(WinnerPredictor, self).__init__()
        self.encoder = BigBirdForSequenceClassification.from_pretrained(
            "google/bigbird-roberta-base", num_labels=encoder_output_size
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
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    cards_path: str = "data/AtomicCards.json",
    cuda: bool = True,
    save_path: str = "models/transformer.pt",
    use_wandb: bool = True,
    transformer_model: str = "google/bigbird-roberta-base",
    dropout: float = 0.2,
    gradient_accumulation_steps: int = 32,
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train model
    model = WinnerPredictor(encoder_model_name=transformer_model).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    squeduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            len(train_dataloader) * epochs / gradient_accumulation_steps / 100
        ),
        num_training_steps=len(train_dataloader) * epochs / gradient_accumulation_steps,
    )
    criterion = nn.MSELoss()

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
            target = target.to(device).to(torch.float32)
            winner = model(deck1, deck2)
            loss = criterion(winner, target)
            loss.backward()

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
                            "steps_lr": optimizer.param_groups[0]["lr"],
                        }
                    )

                optimizer.step()
                squeduler.step()
                optimizer.zero_grad()

                cum_train_losses = []
                cum_train_accuracies = []

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
                winner = model(deck1, deck2)
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

        # Save model
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    typer.run(train)
