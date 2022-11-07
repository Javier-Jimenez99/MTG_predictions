from pymongo import MongoClient
import json
import typer


def insert_tournament(tournament_data):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    if not isinstance(tournament_data, list):
        tournament_data = [tournament_data]

    for tournament in tournament_data:
        tournament_insert_data = {
            "name": tournament["name"],
            "date": tournament[
                "date"
            ],  # dt.datetime.strptime(tournament["date"], "%Y-%m-%d"),
            "format": tournament["format"],
        }

        for round_name in ["quarterfinals", "semifinals", "finals"]:
            round_matches = []
            for match in tournament["rounds"][round_name]:
                player1 = match[0]
                player2 = match[1]
                p1_deck_insert_result = db.decks.insert_one(player1["deck"])
                p2_deck_insert_result = db.decks.insert_one(player2["deck"])
                p1_deck_id = p1_deck_insert_result.inserted_id
                p2_deck_id = p2_deck_insert_result.inserted_id

                match_data = {
                    "p1_name": player1["name"],
                    "p1_deck": p1_deck_id,
                    "p1_points": player1["points"],
                    "p2_name": player2["name"],
                    "p2_deck": p2_deck_id,
                    "p2_points": player2["points"],
                }

                match_inserted_results = db.matches.insert_one(match_data)
                round_matches.append(match_inserted_results.inserted_id)

            tournament_insert_data[round_name] = round_matches

        db.tournaments.insert_one(tournament_insert_data)


def insert_mtgtop8_decks(file_path):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]

    with open(file_path, "r") as f:
        data = json.load(f)

    for deck in data:
        if len(deck["main_deck"]) > 0:
            db.decks.insert_one(deck)


def load_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    insert_tournament(data)
    return data


def get_all_matches_ids():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    matches = db.matches.find().distinct("_id")
    return matches


def get_all_decks(with_archetype=False):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    if with_archetype:
        decks = db.decks.find({"archetype": {"$exists": True}})
    else:
        decks = db.decks.find()
    return decks


def get_all_decks_ids(with_archetype=False):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    decks = None
    if with_archetype:
        decks = db.decks.find({"archetype": {"$exists": True}})
    else:
        decks = db.decks.find()

    return decks.distinct("_id")


def get_all_tournaments():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    tournaments = db.tournaments.find()
    return tournaments


def get_matches_id_by_format(tournament_format):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    tournaments = db.tournaments.find({"format": tournament_format})
    matches = []
    for t in tournaments:
        for round_name in ["quarterfinals", "semifinals", "finals"]:
            for match_id in t[round_name]:
                match = db.matches.find_one({"_id": match_id})["_id"]
                matches.append(match)

    return matches


def get_match(id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    match = db.matches.find_one({"_id": id})

    return match


def get_deck(id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    deck = db.decks.find_one({"_id": id})
    return deck


def main(tournaments: str = None, mtgtop8: str = None, clean_all: bool = False):
    if clean_all:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["MTGpred"]
        db.decks.drop()
        db.matches.drop()
        db.tournaments.drop()

    if tournaments is not None:
        load_from_json(tournaments)
    if mtgtop8 is not None:
        insert_mtgtop8_decks(mtgtop8)
    if tournaments is not None and mtgtop8 is not None:
        typer.echo("No file provided")


if __name__ == "__main__":
    typer.run(main)
