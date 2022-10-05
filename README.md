# MTG Predictions
Tool to predict the result of a match given the text of the cards

## Data collection
To collect the data [MTGO standings](https://magic.wizards.com/en/content/deck-lists-magic-online-products-game-info) has been scraped. Steps to reproduce it:
1. Run `scraper/tournaments_list.py`: it generates a file containing the links of every tournament.
2. Run `scraper/tournament_data.py`: it generates a json file that contains all the tournaments data (metadata, players, decks and matches). It takes as input the file generated in the previous step. 
3. Run `load_from_json` from `utils/database.py`: it insert all the data in a mongo database.