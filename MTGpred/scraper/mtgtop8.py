import json
import re
from os.path import exists

import typer
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


def scrape_deck_links(driver, output_file):
    formats_links = {
        "standard": "https://mtgtop8.com/format?f=ST",
        "modern": "https://mtgtop8.com/format?f=MO",
        "legacy": "https://mtgtop8.com/format?f=LE",
        "pauper": "https://mtgtop8.com/format?f=PAU",
        "pioneer": "https://mtgtop8.com/format?f=PI",
        "vintage": "https://mtgtop8.com/format?f=VI",
    }

    avoid_options_regex = r"^(last\s*\d\s*(weeks|months))|(all\s*(standard|modern|legacy|pauper|pioneer|vintage)\s*decks)|(all\s*decks)"

    all_options = []
    print("GETTING AVAILABLE SECTIONS...")
    for format_name, format_link in formats_links.items():
        driver.get(format_link)
        drop_down_xpath = "//tbody/tr/td[1]/div[2]/div"
        driver.find_element(By.XPATH, drop_down_xpath).click()
        options = driver.find_elements(By.XPATH, drop_down_xpath + "//div/a")

        for o in options:
            if o.text.strip().lower() != "" and not re.match(
                avoid_options_regex, o.text.strip().lower()
            ):
                all_options.append(
                    {"section_link": o.get_attribute("href"), "format": format_name}
                )

    all_decks_data = []
    links_saved = []
    for option in tqdm(all_options, desc="SCRAPING SECTIONS"):
        driver.get(option["section_link"])
        all_divs = driver.find_elements(By.XPATH, "//tbody/tr/td[1]/div")

        # Get links to archetypes
        base_archetype_text = None
        archetype_decks = {}
        for div in all_divs:
            if div.get_attribute("class") == "meta_arch":
                base_archetype_text = div.text.lower().split(" ")[0]
                archetype_decks[base_archetype_text] = []

            elif (
                base_archetype_text is not None
                and div.get_attribute("class") == "hover_tr"
            ):
                a_element = div.find_element(By.TAG_NAME, "a")
                archetype_decks[base_archetype_text].append(
                    {
                        "link": a_element.get_attribute("href"),
                        "detailed_archetype": a_element.text.lower().strip(),
                    }
                )

        # Get links to decks inside archetypes
        # {"aggro":[{"Rakdos Sacrifice": "https://mtgtop8.com/event?e=..."}, ...], ...}
        for archetype, decks_data in archetype_decks.items():
            # ["Rakdos Sacrifice", "https://mtgtop8.com/event?e=..."]
            for deck_data in decks_data:
                detailed_archetype = deck_data["detailed_archetype"]
                archetype_link = deck_data["link"]
                driver.get(archetype_link)
                decks_table = driver.find_elements(
                    By.XPATH, "//tbody/tr/td[2]//tbody//tr[@class='hover_tr']/td[2]/a"
                )
                # ["https://mtgtop8.com/event?e=...", ...]
                for deck in decks_table:
                    deck_link = deck.get_attribute("href")
                    if deck_link not in links_saved:
                        deck_data = {
                            "archetype": archetype,
                            "detailed_archetype": detailed_archetype,
                            "name": deck.text.lower(),
                            "format": option["format"],
                            "link": deck_link,
                        }
                        all_decks_data.append(deck_data)
                        links_saved.append(deck_link)

        json.dump(all_decks_data, open(output_file, "w"))

    return all_decks_data


def scrape_cards(column_element):
    deck_lines = column_element.find_elements(By.CLASS_NAME, "deck_line")
    cards = []
    for line in deck_lines:
        card_name = line.find_element(By.CLASS_NAME, "L14").text
        card_quantity = int(
            line.text.replace(card_name, "").replace("banned", "").strip()
        )

        cards.append({"name": card_name, "quantity": card_quantity})

    return cards


def scrape_decks_data(driver, link):
    driver.get(link)
    deck_columns = driver.find_elements(
        By.XPATH, "//body/div/div/div[7]/div[2]/div[3]/div"
    )
    main_deck_columns = deck_columns[:-1]
    sideboard_column = deck_columns[-1]

    deck_data = {
        "main_deck": [],
        "sideboard": [],
    }

    for column in main_deck_columns:
        deck_data["main_deck"].extend(scrape_cards(column))

    deck_data["sideboard"].extend(scrape_cards(sideboard_column))

    return deck_data


def main(
    output_links_file: str = "data/mtgtop8_decks_links.json",
    output_decks_file: str = None,
    run_all: bool = False,
):
    assert not run_all or (
        output_links_file is not None and output_decks_file is not None
    ), "If run_all is True, output_links_file and output_decks_file must be specified"
    driver = webdriver.Chrome(ChromeDriverManager().install())
    if output_decks_file is None:
        scrape_deck_links(driver, output_links_file)

    if run_all or output_decks_file is not None:
        with open(output_links_file, "r") as f:
            meta_datas = json.load(f)

        filtered_meta_datas = meta_datas
        final_results = []
        if exists(output_decks_file):
            with open(output_decks_file, "r") as f:
                final_results = json.load(f)

            decks_data_links = [d["link"] for d in final_results]

            filtered_meta_datas = [
                data for data in meta_datas if data["link"] not in decks_data_links
            ]

        for i, meta in tqdm(
            enumerate(filtered_meta_datas),
            desc="SCRAPING DECKS",
            total=len(filtered_meta_datas),
        ):
            scraped_deck = scrape_decks_data(driver, meta["link"])
            meta.update(scraped_deck)
            final_results.append(meta)

            if i % 100 == 0:
                json.dump(final_results, open(output_decks_file, "w"))
                print(f"Saved {len(final_results)} decks")

    driver.quit()


if __name__ == "__main__":
    typer.run(main)
