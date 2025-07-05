import json
import os

import requests
from common.utils.timing_logger import LOGGER, log_execution_time
from dotenv import load_dotenv

load_dotenv()


@log_execution_time
def get_search_results(
    query,
    num_results=10,
    num_pages=2,
):
    url = "https://google.serper.dev/search"

    payload = json.dumps(
        [
            {
                "q": query,
                "gl": "sg",
                "num": num_results,
                "page": num_pages,
            }
        ]
    )
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        return response.json()  # Convert response to Python dictionary
    except json.JSONDecodeError:
        print("Serper: get_search_results: Error decoding JSON response")
        LOGGER.error("Serper: get_search_results: Error decoding JSON response")
        return None


def get_webpage(url_):
    url = "https://scrape.serper.dev"
    payload = json.dumps({"url": url_})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        return response.json()
    except json.JSONDecodeError:
        print("Serper: get_webpage: Error decoding JSON response")
        LOGGER.error("Serper: get_webpage: Error decoding JSON response")
        return None


# response_obj = get_search_results("Do adults getting vaccinated for COVID-19?")
# print(response_obj[0])

# for result in response_obj[0]["organic"]:
#     if "date" in result:
#         print(result["date"])
#     else:
#         print("Date attribute not found")
#     print(result["link"])
#     page = get_webpage(result["link"])
#     print(page["metadata"]["og:title"])
#     exit()
