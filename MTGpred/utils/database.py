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
            "date":tournament["date"],#dt.datetime.strptime(tournament["date"], "%Y-%m-%d"),
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


def load_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    insert_tournament(data)
    return data

def get_all_matches_ids():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    matches = db.matches.find().distinct('_id')
    return matches

def get_match(id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    match = db.matches.find_one({"_id":id})

    return match

def get_deck(id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MTGpred"]
    deck = db.decks.find_one({"_id":id})
    return deck

def main(file_path: str = "data/tournaments.json"):
    load_from_json(file_path)

if __name__ == "__main__":
    typer.run(load_from_json)


