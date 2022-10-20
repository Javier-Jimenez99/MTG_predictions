import random
from pickle import FALSE

import torch
import torch.nn.functional as F
import typer
import wandb
from MTGpred.model.dataset import MatchesDataset2
from MTGpred.utils.database import get_all_matches_ids
from MTGpred.utils.mtgjson import load_cards_df
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
    get_cosine_schedule_with_warmup,
)


def get_space_left():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    return f


class DeckEncoder(nn.Module):
    def __init__(
        self,
        base_model="distilbert-base-uncased",
        hidden_size=128,
        transformer_output_size=100,
        output_size=128,
        lstm_bidirectionnal=True,
        dropout=0.2,
    ):
        super(DeckEncoder, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=transformer_output_size, dropout=dropout
        )
        self.LSTM = nn.LSTM(
            input_size=100,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=lstm_bidirectionnal,
        )

        self.lstm_output_size = hidden_size * 2 if lstm_bidirectionnal else hidden_size
        self.last_layer = nn.Linear(self.lstm_output_size, output_size)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.transformer_output_size = transformer_output_size

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return "cuda"
        else:
            return "cpu"

    def forward(self, deck, mask_length):
        cards_encoded = torch.zeros(
            (*deck["input_ids"].shape[:2], self.transformer_output_size)
        ).to(self.get_device())

        # Build a matrix mask for the transformer
        matrix_mask = torch.zeros(deck["input_ids"].shape[:2]).to(self.get_device())
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

        pack_padded = pack_padded_sequence(
            cards_encoded, mask_length, batch_first=True, enforce_sorted=False
        )
        LSTM_output = self.LSTM(pack_padded)[1][0]
        LSTM_output = LSTM_output.swapaxes(0, 1).reshape(-1, self.lstm_output_size)

        output = self.last_layer(LSTM_output)

        return output


class WinnerPredictor(nn.Module):
    def __init__(
        self,
        encoder_model_name="bert-base-uncased",
        transformer_output_size=100,
        lstm_hidden_size=128,
        lstm_bidirectionnal=True,
        dropout=0.2,
    ):
        super(WinnerPredictor, self).__init__()
        self.encoder = DeckEncoder(
            base_model=encoder_model_name,
            hidden_size=lstm_hidden_size,
            transformer_output_size=transformer_output_size,
            lstm_bidirectionnal=lstm_bidirectionnal,
            dropout=dropout,
        )
        self.final_classifier = nn.Linear(self.encoder.output_size * 2, 1)

    def forward(self, deck1, deck2, mask1, mask2):
        deck1 = self.encoder(deck1, mask1)
        deck2 = self.encoder(deck2, mask2)
        output = self.final_classifier(torch.cat((deck1, deck2), dim=1))
        winner = torch.tanh(output).squeeze()
        return winner


def accuracy(y_pred, y_true):
    y_pred = y_pred > 0
    y_true = y_true > 0
    return (y_pred == y_true).sum().item() / len(y_pred)


def train(
    split_ratio: float = 0.9,
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    cards_path: str = "data/AtomicCards.json",
    cuda: bool = True,
    save_path: str = "models/LSTM.pt",
    use_wandb: bool = True,
    transformer_model: str = "distilbert-base-uncased",
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
    train_dataset = MatchesDataset2(
        cards_df, train_matches_ids, model_name=transformer_model
    )
    test_dataset = MatchesDataset2(
        cards_df, test_matches_ids, model_name=transformer_model
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train model
    model = WinnerPredictor(encoder_model_name=transformer_model, dropout=dropout).to(
        device
    )
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
        train_accuracies = []
        for train_step, (deck1, deck2, mask1, mask2, target) in tqdm(
            enumerate(train_dataloader), desc="TRAIN"
        ):
            deck1 = deck1.to(device)
            deck2 = deck2.to(device)
            target = target.to(device).to(torch.float32)
            winner = model(deck1, deck2, mask1, mask2)
            loss = criterion(winner, target)
            loss.backward()

            if (train_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                squeduler.step()
                optimizer.zero_grad()

            train_accuracy = accuracy(winner, target)
            train_accuracies.append(train_accuracy)
            train_losses.append(loss.item())

            if use_wandb:
                wandb.log(
                    {
                        "train_steps_loss": loss.item(),
                        "train_steps_accuracy": train_accuracy,
                        "steps_lr": optimizer.param_groups[0]["lr"],
                    }
                )

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
            for deck1, deck2, mask1, mask2, target in tqdm(
                test_dataloader, desc="TEST"
            ):
                deck1 = deck1.to(device)
                deck2 = deck2.to(device)
                target = target.to(device).to(torch.float32)
                winner = model(deck1, deck2, mask1, mask2)
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
