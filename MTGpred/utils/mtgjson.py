import json
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
        .replace("//", "/")
        .replace("/s+", " ")
    )
    name = name.strip()
    return name

def load_cards_df(data_path:str="data/AtomicCards.json", simplify_names:bool=True):
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

    if simplify_names:
        df["name"] = df["name"].apply(simplify_name)
        df["faceName"] = df["faceName"].apply(simplify_name)
        
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
