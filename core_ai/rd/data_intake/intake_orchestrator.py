# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from .scraper import scrape_competitor_site, fetch_api_data
import logging

def run_data_intake():
    competitor_urls = [
        "https://example.com/model1",
        "https://example.com/model2"
    ]
    all_data = []
    for url in competitor_urls:
        all_data.append(scrape_competitor_site(url))
    # TODO: Add API data intake as needed
    logging.info(f"Data intake collected {len(all_data)} sources.")
    return all_data 