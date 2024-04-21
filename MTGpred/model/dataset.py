from torch.utils.data import Dataset
from MTGpred.utils.mtgjson import parse_mana_cost,simplify_name
from MTGpred.utils.database import get_match, get_deck
from transformers import AutoTokenizer, AutoModel, BatchEncoding, BigBirdTokenizer
import torch
import re
import pandas as pd
from typing import Iterable
import numpy as np

class MatchesDataset(Dataset):
    def __init__(
        self,
        cards_df,
        matches_ids,
        device,
        cache: dict = None,
        query_cache: dict = None,
    ):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.device = device

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)

        self.cache = cache if cache is not None else {}
        self.query_cache = query_cache if query_cache is not None else {}

    def __len__(self):
        return len(self.matches_ids)

    def preprocess_card(self, card):
        name = card["name"]
        simplified_name = simplify_name(name)
        quantity = int(card["quantity"])
        all_variations = []

        selected_card = self.cards_df[
            (self.cards_df["faceName"] == simplified_name)
            | (self.cards_df["name"] == simplified_name)
        ]
        if len(selected_card) == 0:
            print(
                f"WARNING: {name} cant be found in the database. Will be removed from the deck."
            )
            return torch.zeros((1, 768))

        for index, variations in selected_card.iterrows():
            card_encoded = None

            if index in self.cache.keys():
                card_encoded = self.cache[index]
                self.query_cache[index] += 1
            else:
                mana_cost = (
                    parse_mana_cost(variations["manaCost"])
                    if not pd.isna(variations["manaCost"])
                    else ""
                )

                card_type = variations["type"]

                text = variations["text"] if not pd.isna(variations["text"]) else ""
                mana_in_text = re.findall(r"\{.*\}", text)
                for mana in mana_in_text:
                    text = text.replace(mana, parse_mana_cost(mana))

                stats = f"{variations['power']} power, {variations['power']} power"
                input_text = " <SEP> ".join([name, mana_cost, card_type, text, stats])

                tokenized_card = self.tokenizer(input_text, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    card_encoded = self.encoder(**tokenized_card)["pooler_output"]

                if str(self.device) == "cuda":
                    card_encoded = card_encoded.cpu()

                if len(self.cache.keys()) >= 1000:
                    # Remove the lest used card from the cache
                    min_key = min(self.query_cache, key=self.query_cache.get)
                    del self.cache[min_key]
                    del self.query_cache[min_key]

                self.cache[index] = card_encoded
                self.query_cache[index] = 1

            all_variations.append(torch.cat([card_encoded] * quantity))

        if len(all_variations) > 1:
            return torch.cat(all_variations)
        else:
            return all_variations[0]

    def preprocess_deck(self, deck_id):
        deck_data = get_deck(deck_id)

        all_cards = []

        for card in deck_data["main_deck"]:
            all_cards.append(self.preprocess_card(card))

        for card in deck_data["sideboard"]:
            all_cards.append(self.preprocess_card(card))

        return torch.cat(all_cards)

    def __getitem__(self, idx):
        match_id = self.matches_ids[idx]
        match_data = get_match(match_id)

        p1_deck = self.preprocess_deck(match_data["p1_deck"])
        p2_deck = self.preprocess_deck(match_data["p2_deck"])

        p1_deck_shape = p1_deck.shape
        p2_deck_shape = p2_deck.shape

        p1_empty_matrix = torch.zeros((120 - p1_deck_shape[0], p1_deck_shape[1]))
        p2_empty_matrix = torch.zeros((120 - p2_deck_shape[0], p2_deck_shape[1]))

        p1_deck = torch.cat([p1_deck, p1_empty_matrix])
        p2_deck = torch.cat([p2_deck, p2_empty_matrix])

        # Random number 0 or 1
        random_number = torch.randint(0, 2, (1, 1))
        # If 0, p1 wins, if 1, p2 wins
        if random_number == 0:
            return torch.stack((p1_deck, p2_deck)), int(
                match_data["p1_points"] < match_data["p2_points"]
            )
        else:
            return torch.stack((p2_deck, p1_deck)), int(
                match_data["p2_points"] < match_data["p1_points"]
            )


class MatchesDataset2(Dataset):
    def __init__(
        self,
        cards_df,
        matches_ids,
        model_name="distilbert-base-uncased",
        random_output=True,
    ):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)
        self.random_output = random_output

    def __len__(self):
        return len(self.matches_ids)

    def preprocess_card(self, card):
        name = card["name"]
        simplified_name = simplify_name(name)
        quantity = int(card["quantity"])
        all_variations = []

        selected_card = self.cards_df[
            (self.cards_df["faceName"] == simplified_name)
            | (self.cards_df["name"] == simplified_name)
        ]
        if len(selected_card) == 0:
            print(
                f"WARNING: {name} cant be found in the database. Will be removed from the deck."
            )
            return []

        for index, variations in selected_card.iterrows():
            mana_cost = (
                parse_mana_cost(variations["manaCost"])
                if not pd.isna(variations["manaCost"])
                else ""
            )

            card_type = variations["type"]

            text = variations["text"] if not pd.isna(variations["text"]) else ""
            mana_in_text = re.findall(r"\{.*\}", text)
            for mana in mana_in_text:
                text = text.replace(mana, parse_mana_cost(mana))

            stats = f"{variations['power']} power, {variations['power']} power"
            quantity_text = f"{quantity} copies"
            input_text = self.tokenizer.sep_token.join(
                [quantity_text, name, mana_cost, card_type, text, stats]
            )

            all_variations.append(
                self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
            )

        return all_variations

    def get_deck_text(self, deck_id):
        deck_data = get_deck(deck_id)

        all_cards = []

        for card in deck_data["main_deck"]:
            all_cards.extend(self.preprocess_card(card))

        for card in deck_data["sideboard"]:
            all_cards.extend(self.preprocess_card(card))

        mask_length = len(all_cards)

        # Add padding
        for i in range(120 - len(all_cards)):
            all_cards.append(
                self.tokenizer(
                    self.tokenizer.pad_token,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
            )

        batch_encoding_data = {}
        for k in all_cards[0].keys():
            values = [d[k] for d in all_cards]
            batch_encoding_data[k] = torch.cat(values)

        return BatchEncoding(batch_encoding_data), mask_length

    def get_match_id(self, idx):
        return self.matches_ids[idx]

    def __getitem__(self, idx):
        match_id = self.matches_ids[idx]
        match_data = get_match(match_id)

        p1_deck, mask_length1 = self.get_deck_text(match_data["p1_deck"])
        p2_deck, mask_length2 = self.get_deck_text(match_data["p2_deck"])

        # Random number 0 or 1
        random_number = torch.randint(0, 2, (1, 1))
        # If 0, p1 wins, if 1, p2 wins
        if not self.random_output or random_number == 0:
            return (
                p1_deck,
                p2_deck,
                mask_length1,
                mask_length2,
                int(match_data["p1_points"]) * 0.5 - int(match_data["p2_points"]) * 0.5,
            )
        else:
            return (
                p2_deck,
                p1_deck,
                mask_length2,
                mask_length1,
                int(match_data["p2_points"]) * 0.5 - int(match_data["p1_points"]) * 0.5,
            )


class MatchesDataset3(Dataset):
    def __init__(
        self,
        cards_df,
        matches_ids,
        model_name="google/bigbird-roberta-base",
        random_output=True,
    ):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)
        self.random_output = random_output

    def __len__(self):
        return len(self.matches_ids)

    def preprocess_card(self, card):
        name = card["name"]
        simplified_name = simplify_name(name)
        quantity = int(card["quantity"])
        all_variations = []

        selected_card = self.cards_df[
            (self.cards_df["faceName"] == simplified_name)
            | (self.cards_df["name"] == simplified_name)
        ]
        if len(selected_card) == 0:
            print(
                f"WARNING: {name} cant be found in the database. Will be removed from the deck."
            )
            return []

        for index, variations in selected_card.iterrows():
            mana_cost = (
                parse_mana_cost(variations["manaCost"])
                if not pd.isna(variations["manaCost"])
                else ""
            )

            card_type = variations["type"]

            text = variations["text"] if not pd.isna(variations["text"]) else ""
            mana_in_text = re.findall(r"\{.*\}", text)
            for mana in mana_in_text:
                text = text.replace(mana, parse_mana_cost(mana))

            stats = f"{variations['power']} power, {variations['power']} power"
            quantity_text = f"{quantity} copies"
            input_text = self.tokenizer.sep_token.join(
                [quantity_text, name, mana_cost, card_type, text, stats]
            )

            all_variations.append(input_text)

        return all_variations

    def get_deck_text(self, deck_id):
        deck_data = get_deck(deck_id)

        all_cards = ["Main deck:"]

        for card in deck_data["main_deck"]:
            all_cards.extend(self.preprocess_card(card))

        for card in deck_data["sideboard"]:
            all_cards.extend(self.preprocess_card(card))

        all_cards.append("Sideboard:")

        deck_text = self.tokenizer.sep_token.join(all_cards)

        tokenized_deck = self.tokenizer(
            deck_text,
            return_tensors="pt",
            padding="max_length",
            max_length=2500,
            truncation=True,
        )

        batch_encoding_data = {}
        for k, v in tokenized_deck.items():
            batch_encoding_data[k] = v.squeeze()

        return BatchEncoding(batch_encoding_data)

    def get_match_id(self, idx):
        return self.matches_ids[idx]

    def __getitem__(self, idx):
        match_id = self.matches_ids[idx]
        match_data = get_match(match_id)

        p1_deck = self.get_deck_text(match_data["p1_deck"])
        p2_deck = self.get_deck_text(match_data["p2_deck"])

        # Random number 0 or 1
        random_number = torch.randint(0, 2, (1, 1))
        # If 0, p1 wins, if 1, p2 wins
        if not self.random_output or random_number == 0:
            return (
                p1_deck,
                p2_deck,
                int(int(match_data["p1_points"]) < int(match_data["p2_points"])),
            )
        else:
            return (
                p2_deck,
                p1_deck,
                int(int(match_data["p2_points"]) < int(match_data["p1_points"])),
            )


class JoinedMatchesDataset(Dataset):
    def __init__(
        self,
        cards_df,
        matches_df,
        model_name="google/bigbird-roberta-base",
        random_output=True,
        sideboard=False,
        max_length=4096,
        truncation=True,
        return_raw_text=False,
        cased=False,
    ):
        self.cards_df = cards_df
        self.matches_df = matches_df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)
        self.random_output = random_output
        self.sideboard = sideboard
        self.max_length = max_length
        self.truncation = truncation
        self.return_raw_text = return_raw_text
        self.cased = cased

    def __len__(self):
        return len(self.matches_df)

    def preprocess_card(self, card):
        name = card["name"]
        simplified_name = simplify_name(name)
        quantity = int(card["quantity"])
        all_variations = []

        selected_card = self.cards_df[
            (self.cards_df["faceName"] == simplified_name)
            | (self.cards_df["name"] == simplified_name)
        ]
        if len(selected_card) == 0:
            print(
                f"WARNING: {name} cant be found in the database. Will be removed from the deck."
            )
            return []

        for index, variations in selected_card.iterrows():
            mana_cost = (
                parse_mana_cost(variations["manaCost"])
                if not pd.isna(variations["manaCost"])
                else ""
            )

            card_type = variations["type"]

            text = variations["text"] if not pd.isna(variations["text"]) else ""
            mana_in_text = re.findall(r"\{.*\}", text)
            for mana in mana_in_text:
                text = text.replace(mana, parse_mana_cost(mana))

            quantity_text = f"{quantity} copies"
            input_text = " - ".join([quantity_text, name, mana_cost, card_type, text])

            if variations["power"] != np.nan or variations["toughness"] != np.nan:
                input_text += f" - {variations['power']} power, {variations['toughness']} toughness"

            if not self.cased:
                input_text = input_text.lower()

            all_variations.append(input_text)

        return all_variations

    def get_deck_text(self, deck_data):
        all_cards = []

        for card in deck_data["main_deck"]:
            all_cards.extend(self.preprocess_card(card))

        if self.sideboard:
            for card in deck_data["sideboard"]:
                all_cards.extend(self.preprocess_card(card))

        all_cards = (
            self.tokenizer.sep_token.join(all_cards) + " " + self.tokenizer.sep_token
        )

        return all_cards

    def __getitem__(self, idx):
        match_data = self.matches_df.iloc[idx]

        p1_deck = self.get_deck_text(match_data["p1_deck"])
        p2_deck = self.get_deck_text(match_data["p2_deck"])

        # Random number 0 or 1
        random_number = torch.randint(0, 2, (1, 1))
        # If 0, p1 wins, if 1, p2 wins
        if not self.random_output or random_number == 0:
            # TODO: BE CAREFUL TWO CLS TOKENS
            all_text = p1_deck + f" {self.tokenizer.cls_token} " + p2_deck
            label = int(int(match_data["p1_points"]) < int(match_data["p2_points"]))
        else:
            all_text = p2_deck + f" {self.tokenizer.cls_token} " + p1_deck
            label = int(int(match_data["p2_points"]) < int(match_data["p1_points"]))

        tokenized_match = self.tokenizer(
            all_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=self.truncation,
        )

        if "longformer" in self.model_name:
            tokenized_match["global_attention_mask"] = np.array(
                [
                    (
                        input_id == self.tokenizer.sep_token_id
                        or input_id == self.tokenizer.cls_token_id
                    )
                    for input_id in tokenized_match["input_ids"]
                ],
                dtype=int,
            )

        if self.return_raw_text:
            tokenized_match["raw_text"] = all_text

        tokenized_match["labels"] = torch.tensor(label)

        return tokenized_match


class DecksDataset(Dataset):
    def __init__(
        self,
        cards_df: pd.DataFrame,
        decks_data: dict,
        model_name: str = "xlnet-base-cased",
        cased: bool = True,
        join_tokens: bool = False,
        max_length: int = 256,
        truncation: bool = True,
        sideboard: bool = False,
        return_raw_text: bool = False,
        labels_field: str = "archetype",
        labels: Iterable[str] = ["aggro", "control", "combo"],
    ):
        """
        Dataset for decks classification

        Parameters
        ----------
        cards_df : pd.DataFrame
            Dataframe with all the cards
        decks_data : dict[str, str]
            Dictionary with the decks data
        model_name : str, optional
            Name of the transformer model, by default "xlnet-base-cased"
        cased : bool, optional
            Cased the card output or not, by default True
        labels : Iterable[str], optional
            Output option labels, by default ["aggro", "control", "combo"]
        join_tokens : bool, optional
            Join all texts and tokenize all or not, by default False
        max_length : int, optional
            Max length of the tokenizer, by default 256
        truncation : bool, optional
            Truncate the text or not, by default True
        sideboard : bool, optional
            Include the sideboard or not, by default False
        return_raw_text : bool, optional
            Return the raw text or not, by default False
        labels_field : str, optional
            Field of the labels in the decks data, by default "archetype"
        """

        assert not join_tokens or (
            join_tokens and truncation
        ), "If join_tokens is True, truncation must be True"

        self.cards_df = cards_df
        self.decks_data = decks_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)
        self.labels = labels
        self.labels_field = labels_field
        self.cased = cased
        self.join_tokens = join_tokens
        self.max_length = max_length
        self.truncation = truncation
        self.sideboard = sideboard
        self.return_raw_text = return_raw_text

    def __len__(self):
        return len(self.decks_data)

    def preprocess_card(self, card):
        name = card["name"]
        simplified_name = simplify_name(name)
        quantity = int(card["quantity"])
        all_variations = []

        selected_card = self.cards_df[
            (self.cards_df["faceName"] == simplified_name)
            | (self.cards_df["name"] == simplified_name)
        ]
        if len(selected_card) == 0:
            print(
                f"WARNING: {name} cant be found in the database. Will be removed from the deck."
            )
            return []

        for index, variations in selected_card.iterrows():
            mana_cost = (
                parse_mana_cost(variations["manaCost"])
                if not pd.isna(variations["manaCost"])
                else ""
            )

            card_type = variations["type"]

            text = variations["text"] if not pd.isna(variations["text"]) else ""
            mana_in_text = re.findall(r"\{.*\}", text)
            for mana in mana_in_text:
                text = text.replace(mana, parse_mana_cost(mana))

            quantity_text = f"{quantity} copies"
            input_text = " - ".join([quantity_text, name, mana_cost, card_type, text])

            if variations["power"] != np.nan or variations["toughness"] != np.nan:
                input_text += f" - {variations['power']} power, {variations['toughness']} toughness"

            if not self.cased:
                input_text = input_text.lower()

            all_variations.append(input_text)

        return all_variations

    def get_tokenized_text(self, deck_data):
        all_cards = []

        for card in deck_data["main_deck"]:
            all_cards.extend(self.preprocess_card(card))

        if self.sideboard:
            for card in deck_data["sideboard"]:
                all_cards.extend(self.preprocess_card(card))

        if self.join_tokens:
            all_cards = (
                self.tokenizer.sep_token.join(all_cards)
                + " "
                + self.tokenizer.sep_token
            )

            tokenized_deck = self.tokenizer(
                all_cards,
                padding="max_length",
                max_length=self.max_length,
                truncation=self.truncation,
            )

            if "longformer" in self.model_name:
                tokenized_deck["global_attention_mask"] = np.array(
                    [
                        (
                            input_id == self.tokenizer.sep_token_id
                            or input_id == self.tokenizer.cls_token_id
                        )
                        for input_id in tokenized_deck["input_ids"]
                    ],
                    dtype=int,
                )

            if self.return_raw_text:
                tokenized_deck["raw_text"] = all_cards
        else:
            tokenized_deck = []
            for card in all_cards:
                tokenized_deck.append(
                    self.tokenizer(
                        card,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=self.truncation,
                    )
                )

        return tokenized_deck

    def __getitem__(self, idx):
        deck_data = self.decks_data[idx]

        deck = self.get_tokenized_text(deck_data)

        if self.join_tokens:
            deck["labels"] = torch.tensor(
                self.labels.index(deck_data[self.labels_field])
            )
            return deck
        else:
            return {
                "input": deck,
                "label": self.labels.index(deck_data[self.labels_field]),
                "mask_length": len(deck),
            }
