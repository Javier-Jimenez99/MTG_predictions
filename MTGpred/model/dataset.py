from torch.utils.data import Dataset
from MTGpred.utils.mtgjson import load_cards_df,parse_mana_cost
from MTGpred.utils.database import get_match, get_deck
from transformers import AutoTokenizer, AutoModel
import torch
import re
import pandas as pd

class MatchesDataset(Dataset):
    def __init__(self,cards_df,matches_ids,device):
        self.cards_df = cards_df
        self.matches_ids = matches_ids
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.device = device

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

            tokenized_card = self.tokenizer(input_text,return_tensors="pt").to(self.device)
            
            card_encoded = self.encoder(**tokenized_card)["pooler_output"]

            if self.device == "cuda":
                card_encoded = card_encoded.cpu()

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
        
        p1_deck_shape = p1_deck.shape
        p2_deck_shape = p2_deck.shape

        p1_empty_matrix = torch.zeros((100 - p1_deck_shape[0],p1_deck_shape[1]))
        p2_empty_matrix = torch.zeros((100 - p2_deck_shape[0],p2_deck_shape[1]))

        p1_deck = torch.cat([p1_deck,p1_empty_matrix])
        p2_deck = torch.cat([p2_deck,p2_empty_matrix])

        # Random number 0 or 1
        random_number = torch.randint(0,2,(1,1))
        # If 0, p1 wins, if 1, p2 wins
        if random_number == 0:
            return torch.stack((p1_deck,p2_deck)), int(match_data["p1_points"] < match_data["p2_points"])
        else:
            return torch.stack((p2_deck,p1_deck)), int(match_data["p2_points"] < match_data["p1_points"])

        