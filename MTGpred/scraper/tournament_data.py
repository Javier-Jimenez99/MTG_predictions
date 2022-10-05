import typer
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import re
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from tqdm import tqdm
import datetime as dt


FORMATS_LISTS = ["modern","standard","legacy","pauper","pioneer","historic","commander","sealed","draft"]

def scrape_game(game_element):
    player_elements = game_element.find_elements(By.CLASS_NAME,"player")
    
    winner_text = player_elements[0].get_attribute('innerText')
    loser_text = player_elements[1].get_attribute('innerText')

    name_re =r'(\(\d+\))\s*(.*)'
    points_re = r'\,\s*(\d+)\-(\d+)'
    winner_re = name_re + points_re
    winner_search = re.search(winner_re,winner_text)

    loser_re = name_re
    if re.search(points_re,loser_text):
        loser_re += points_re

    loser_search = re.search(loser_re,loser_text)
    
    return [
        { 
            "name": winner_search.group(2),
            "points": winner_search.group(3),
        },
        {
            "name": loser_search.group(2),
            "points": winner_search.group(4),
        }
    ]


def scrape_round(round_element):
    game_elements = round_element.find_elements(By.CLASS_NAME,"dual-players")
    
    games = []
    for e in game_elements:
        game = scrape_game(e)
        games.append(game)

    return games

def scrape_card(card_element):
    card = {}
    card["name"] = card_element.find_element(By.CLASS_NAME,"card-name").get_attribute('innerText')
    card["quantity"] = card_element.find_element(By.CLASS_NAME,"card-count").get_attribute('innerText')
    return card

def scrape_deck(deck_element):
    data = {}
    data["name"] = deck_element.find_element(By.CLASS_NAME,"deck-meta").find_element(By.TAG_NAME,"h4").get_attribute('innerText').split("(")[0].strip().lower()
    
    main_cards_elements = WebDriverWait(deck_element,5).until(EC.presence_of_element_located((By.CLASS_NAME,"sorted-by-overview-container"))).find_elements(By.CLASS_NAME,"row")
    try:
        sideboard_cards_elements = WebDriverWait(deck_element,5).until(EC.presence_of_element_located((By.CLASS_NAME,"sorted-by-sideboard-container"))).find_elements(By.CLASS_NAME,"row")
    except TimeoutException:
        sideboard_cards_elements = []

    data["main_deck"] = [scrape_card(e) for e in main_cards_elements]
    data["sideboard"] = [scrape_card(e) for e in sideboard_cards_elements]

    return data

def scrape_tournament_data(tournament_link,driver):
    driver.get(tournament_link)
    data = {}

    # NAME, DATE AND FORMAT
    name_date = tournament_link.split("/")[-1].split("-")

    str_date = driver.find_element(By.CLASS_NAME,"posted-in").get_attribute('innerText').split("on")[-1].strip()
    data["date"] = dt.datetime.strptime(str_date, "%B %d, %Y").strftime("%Y-%m-%d")
    data["name"] = " ".join(name_date[:-3])
    data["link"] = tournament_link

    data["format"] = "other"
    for format in FORMATS_LISTS:
        if format in data["name"]:
            data["format"] = format
            break
    
    # DECKS
    deck_elements = driver.find_elements(By.CLASS_NAME,"beanSpacing")[:8]
    decks = {}
    for e in deck_elements:
        deck_data = scrape_deck(e)
        player_name = deck_data.pop("name")
        decks[player_name] = deck_data
    
    # MATCHES
    rounds = {}
    for n in ["quarterfinals","semifinals","finals"]:
        round_element = driver.find_element(By.CLASS_NAME,n)
        round_data = scrape_round(round_element)

        matches = []
        for match in round_data:
            # Add the deck to each player of the match 
            for i in range(len(match)):
                match[i]["deck"] = decks[match[i]["name"].lower()]
            
            matches.append(match)
        
        rounds[n] = matches

    data["rounds"] = rounds

    return data

def main(links_file_path:str = "data/tournament_links.txt", save_path: str = "data/tournaments.json"):
    driver = webdriver.Chrome(ChromeDriverManager().install())

    with open(links_file_path, "r") as f:
        links = f.read().splitlines()

    # Make sure the links are unique
    links = list(set(links))

    all_tournaments = []

    for i,link in tqdm(enumerate(links),desc="Scrapping tournaments",total=len(links)):
        tournament_data = scrape_tournament_data(link,driver)
        all_tournaments.append(tournament_data)

        if i % 10 == 0:
            with open(save_path, "w") as f:
                json.dump(all_tournaments,f)
        
if __name__ == "__main__":
    typer.run(main)