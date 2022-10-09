from torch.utils.data import Dataset
from MTGpred.utils.mtgjson import parse_mana_cost
from MTGpred.utils.database import get_match, get_deck
from transformers import AutoTokenizer, AutoModel, BatchEncoding
import torch
import re
import pandas as pd

ACCENTS = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ü": "u",
    "ñ": "n",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "Ü": "U",
    "Ñ": "N",
    "è": "e",
    "È": "E",
    "à": "a",
    "À": "A",
    "ì": "i",
    "Ì": "I",
    "ò": "o",
    "Ò": "O",
    "ù": "u",
    "Ù": "U",
    "ç": "c",
    "Ç": "C",
    "ÿ": "y",
    "Ÿ": "Y",
    "â": "a",
    "Â": "A",
    "ê": "e",
    "Ê": "E",
    "î": "i",
    "Î": "I",
    "ô": "o",
    "Ô": "O",
    "û": "u",
    "Û": "U",
    "ä": "a",
    "Ä": "A",
    "ë": "e",
    "Ë": "E",
    "ï": "i",
    "Ï": "I",
    "ö": "o",
    "Ö": "O",
    "ü": "u",
    "Ü": "U",
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
}


def strip_accents(s):
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    return s


def simplify_name(name):
    name = name.lower()
    name = strip_accents(name)
    name = (
        name.replace("-", " ")
        .replace("'", "")
        .replace(",", "")
        .replace(".", "")
        .replace("?amp?", "")
        .replace("&", "")
    )
    name = name.strip()
    return name


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
    def __init__(self, cards_df, matches_ids, model_name="distilbert-base-uncased"):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.cards_df["faceName"] = self.cards_df["faceName"].apply(simplify_name)
        self.cards_df["name"] = self.cards_df["name"].apply(simplify_name)

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

    def __getitem__(self, idx):
        match_id = self.matches_ids[idx]
        match_data = get_match(match_id)

        p1_deck, mask_length1 = self.get_deck_text(match_data["p1_deck"])
        p2_deck, mask_length2 = self.get_deck_text(match_data["p2_deck"])

        # Random number 0 or 1
        random_number = torch.randint(0, 2, (1, 1))
        # If 0, p1 wins, if 1, p2 wins
        if random_number == 0:
            return (
                p1_deck,
                p2_deck,
                mask_length1,
                mask_length2,
                int(match_data["p1_points"]) * 0.5 - int(match_data["p2_points"]) * 0.5,
            )
        else:
            return (
                p1_deck,
                p2_deck,
                mask_length2,
                mask_length1,
                int(match_data["p2_points"]) * 0.5 - int(match_data["p1_points"]) * 0.5,
            )
