from torch.utils.data import Dataset
from MTGpred.utils.mtgjson import load_cards_df,parse_mana_cost
from MTGpred.utils.database import get_all_matches_ids, get_match, get_deck
from transformers import AutoTokenizer, AutoModel
import torch
import re
import pandas as pd

class Matches(Dataset):
    def __init__(self,cards_df,matches_ids):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.matches_ids)

    def preprocess_card(self,card):
        name = card["name"]
        quantity = int(card["quantity"])
        all_variations = []
        for index,variations in self.cards_df[self.cards_df["name"] == name].iterrows():
            mana_cost = parse_mana_cost(variations["manaCost"]) if not pd.isna(variations["manaCost"]) else ""

            card_type = variations["type"]

            text = variations["text"] if not pd.isna(variations["text"]) else ""
            mana_in_text = re.findall(r"\{\}",text)
            for mana in mana_in_text:
                text = text.replace(mana,parse_mana_cost(mana))

            stats = f"{variations['power']} power, {variations['power']} power"
            input_text = " <SEP> ".join([name,mana_cost,card_type,text,stats])

            tokenized_card = self.tokenizer(input_text,return_tensors="pt")
            
            card_encoded = self.encoder(**tokenized_card)["pooler_output"]
            print(card_encoded.shape)

            all_variations.append(torch.cat([card_encoded]))

        if len(all_variations) > 1:
            return torch.cat(all_variations)
        else:
            return all_variations[0]
        
    def preprocess_deck(self,deck_id):
        deck_data = get_deck(deck_id)

        all_cards = []

        for card in deck_data["main_deck"]:
            all_cards.append(self.preprocess_card(card))

        for card in deck_data["sideboard"]:
            all_cards.append(self.preprocess_card(card))

        return torch.cat(all_cards)            

    def __getitem__(self,idx):
        match_id = self.matches_ids[idx]
        match_data = get_match(match_id)

        p1_deck = self.preprocess_deck(match_data["p1_deck"])
        p2_deck = self.preprocess_deck(match_data["p2_deck"])

        return p1_deck,p2_deck