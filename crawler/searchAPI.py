import requests
from common.utils.timing_logger import LOGGER, log_execution_time
from dotenv import load_dotenv

load_dotenv()


@log_execution_time
def get_search_results(
    query,
    num_results=10,
    num_pages=2,
    country_code="sg",
):
    print("get_search_results")
    url = "https://www.searchapi.io/api/v1/search"

    params = {
        "engine": "google",
        "q": query,
        "gl": country_code,
        "page": num_pages,
        "num": num_results,
        # "api_key": os.getenv("SEARCH_API_KEY"),
        "api_key": "179uJNEszyAMibJ4PAZybxZf",
        # "api_key": "n2rsLrW1ht91UHUGyWopgRyJ",
    }

    response = requests.get(url, params=params)
    print("repsonse", response.json())
    try:
        return response.json()
    except Exception as e:
        print("Search API: get_search_results: Error decoding JSON response: ", e)
        LOGGER.error("Serper: get_search_results: Error decoding JSON response: ", e)
        return None


# url = "https://www.searchapi.io/api/v1/search"
# params = {
#     "engine": "google",
#     "q": "Youth unemployment trends in Europe?",
#     "gl": "sg",
#     "api_key": "n2rsLrW1ht91UHUGyWopgRyJ",
# }

# response = requests.get(url, params=params)
# response = response.json()
# print("res", response)
# organic_results = response["organic_results"]
# print(organic_results)
