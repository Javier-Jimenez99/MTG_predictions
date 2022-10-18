from threading import current_thread
import typer
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json
from tqdm import tqdm


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
                    a_element.get_attribute("href")
                )

        # Get links to decks inside archetypes
        for archetype, archetype_links in archetype_decks.items():
            for archetype_link in archetype_links:
                driver.get(archetype_link)
                decks_table = driver.find_elements(
                    By.XPATH, "//tbody/tr/td[2]//tbody//tr[@class='hover_tr']/td[2]/a"
                )

                for deck in decks_table:
                    deck_link = deck.get_attribute("href")
                    if deck_link not in links_saved:
                        deck_data = {
                            "archetype": archetype,
                            "detailed_archetype": deck.text.lower(),
                            "link": deck_link,
                            "format": option["format"],
                        }
                        all_decks_data.append(deck_data)
                        links_saved.append(deck_link)

        json.dump(all_decks_data, open(output_file, "w"))

    return all_decks_data


def main(output_file: str = "data/mtgtop8_decks_links.json"):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    scrape_deck_links(driver, output_file)
    driver.quit()


if __name__ == "__main__":
    typer.run(main)
