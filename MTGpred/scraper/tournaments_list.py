from lib2to3.pgen2.driver import Driver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException,TimeoutException,UnexpectedAlertPresentException,ElementClickInterceptedException,StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime as dt
import typer

def get_links(date_start,date_end,driver):

    # Filter dates
    date_start_field = driver.find_element(By.ID,"datepickerFrom")
    date_start_field.clear()
    date_start_field.send_keys(date_start.strftime("%m/%d/%Y"))

    date_end_field = driver.find_element(By.ID,"datepickerTo")
    date_end_field.clear()
    date_end_field.send_keys(date_end.strftime("%m/%d/%Y"))

    # Filter only brackets
    search_field = driver.find_element(By.CLASS_NAME, "search-field-group").find_element(By.CLASS_NAME, "form-text")
    search_field.clear()
    search_field.send_keys("Bracket")

    search_buttom = driver.find_element(By.ID,"custom-search-submit")
    search_buttom.click()

    try:
        see_more_button = driver.find_element(By.CLASS_NAME,"see-more-article-listing-section")
    except UnexpectedAlertPresentException as e:
        if "TO must be greater than or equal to FROM" in e.alert_text:
            try:
                WebDriverWait(driver, 2).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                alert.accept()
            except TimeoutException:
                pass

            return []

    driver.execute_script('arguments[0].setAttribute("data-see-more-limit","50")', see_more_button)

    while see_more_button:
        try:
            see_more_buttom = WebDriverWait(driver, 1).until(EC.element_to_be_clickable((By.CLASS_NAME,"see-more-article-listing-section")))
            see_more_buttom.click()
        except (ElementNotInteractableException,TimeoutException,ElementClickInterceptedException):
            break

    try:
        WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CLASS_NAME,"no-result")))
        return []
    except TimeoutException:
        pass

    while True:
        try:
            WebDriverWait(driver,10).until(EC.presence_of_element_located((By.CLASS_NAME,"article-item-extended")))
            div_links = driver.find_elements(By.CLASS_NAME,"article-item-extended")
            list_links = [div.find_element(By.TAG_NAME,"a").get_attribute("href") for div in div_links]
            return list_links
        except TimeoutException:
            return []
        except StaleElementReferenceException:
            pass

    

def main(save_path: str = "data/links.txt", date_start: str = "06-03-2014", date_end: str = "today"):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://magic.wizards.com/en/content/deck-lists-magic-online-products-game-info')
    driver.find_element(By.CLASS_NAME,"decline-button").click()
    time.sleep(2)

    current_date = dt.datetime.strptime(date_start, '%m-%d-%Y')

    if date_end == "today":
        date_end = dt.datetime.now()
    else:
        date_end = dt.datetime.strptime(date_end, '%m-%d-%Y')

    # Create empty file
    open(save_path, "w")

    while current_date < date_end:
        next_date = current_date + dt.timedelta(days=1)

        if next_date > date_end:
            next_date = date_end

        links = get_links(current_date,next_date,driver)
        
        if len(links)>0:
            with open(save_path, "a") as f:
                f.write("\n".join(links))
                f.write("\n")

        print(f"{len(links)} tournaments from {current_date.strftime('%m-%d-%Y')} to {next_date.strftime('%m-%d-%Y')} saved")

        current_date = next_date
        

if __name__ == "__main__":
    typer.run(main)
