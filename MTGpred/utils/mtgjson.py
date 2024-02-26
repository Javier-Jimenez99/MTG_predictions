import json
import pandas as pd


def load_cards_df(data_path: str = "data/AtomicCards.json"):
    all_cards_json = json.load(open(data_path, encoding="utf8"))["data"]

    all_cards = []
    for name, value in all_cards_json.items():
        for i, card in enumerate(value):
            new_name = name
            if i != 0:
                new_name = f"{name}_{i}"

            parsed_card = card
            parsed_card["name"] = new_name

            if "faceName" not in card.keys():
                parsed_card["faceName"] = new_name

            all_cards.append(parsed_card)

    df = pd.DataFrame(all_cards)

    renames = {col: col.strip() for col in df.columns}
    df = df.rename(columns=renames)

    return df


def parse_mana_cost(mana_cost):
    mana_parsed = ""
    for v in mana_cost[:-1]:
        if v == "W":
            mana_parsed += "white"
        elif v == "U":
            mana_parsed += "blue"
        elif v == "B":
            mana_parsed += "black"
        elif v == "R":
            mana_parsed += "red"
        elif v == "G":
            mana_parsed += "green"
        elif v == "P":
            mana_parsed += "life"
        elif v == "T":
            mana_parsed += "tap"
        elif v == "{":
            continue
        elif v == "}":
            mana_parsed += ","
        elif v.isdigit():
            mana_parsed += " ".join(["corlorless"] * int(v))
        else:
            mana_parsed += " "

    return mana_cost
