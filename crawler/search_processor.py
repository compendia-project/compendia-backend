import csv
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from common.utils.timing_logger import LOGGER, log_execution_time

from .searchAPI import get_search_results

# from .serper import get_search_results
from .serper import get_webpage


@log_execution_time
def get_google_search_results(
    query, num_results=10, csv_filename="search_results.csv", country_code="sg"
):

    ## serper settings for search results
    # response = get_search_results(query, num_results)
    # organic_results = response[0]["organic"]
    print("get_google_search_results")

    ## searchapi settings for search results
    response = get_search_results(query, num_results, country_code=country_code)
    organic_results = response["organic_results"]

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Title",
            "Link",
            "Date",
            "Meta_Description",
            "Headings",
            "Image_URLs",
            "Page_Content",
            # for searchapi
            "Favicon",
            "Source",
            "Domain",
            "Displayed_Link",
            "Snippet_Highlighted_Words",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks with their indices
            futures = [
                (idx, executor.submit(process_result, result))
                for idx, result in enumerate(organic_results)
            ]

            # Collect and sort results by index to maintain original order
            sorted_results = []
            for idx, future in futures:
                try:
                    result_data = future.result()
                    sorted_results.append((idx, result_data))
                except Exception as e:
                    LOGGER.error(f"Error processing result {idx}: {e}")

            # Sort by original index and write to CSV
            sorted_results.sort(key=lambda x: x[0])
            for _, data in sorted_results:
                writer.writerow(data)


def process_result(result):
    """Process individual search result and return data for CSV row"""
    print("process_result")
    page_data = crawl_page(result["link"])

    return {
        "Title": result["title"],
        "Link": result["link"],
        "Date": (
            result.get("date")
            if result.get("date") and str(result.get("date")) != "NaN"
            else ""
        ),
        "Meta_Description": result["snippet"],
        "Headings": "; ".join(page_data["Headings"]),
        "Image_URLs": "; ".join(page_data["Image_URLs"]),
        "Page_Content": page_data["Page_Content"],
        # for searchapi
        "Favicon": result["favicon"],
        "Source": result["source"],
        "Domain": result["domain"],
        "Displayed_Link": result["displayed_link"],
        "Snippet_Highlighted_Words": result["snippet_highlighted_words"],
    }


def crawl_page(url):
    """Crawl a webpage with enhanced error handling and retries"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            print("success getting response")
            return extract_specific_data(BeautifulSoup(response.text, "html.parser"))

        # Fallback to serper.dev if direct request fails
        if website := get_webpage(url):
            return {
                "Meta_Description": website["metadata"]["description"],
                "Headings": [],
                "Image_URLs": [],
                "Page_Content": (
                    website["text"].split("\n\n") if len(website["text"]) >= 200 else ""
                ),
            }

        return error_response(f"HTTP Error {response.status_code}")

    except Exception as e:
        return error_response(str(e))


def extract_specific_data(soup):
    """Extract data from BeautifulSoup parsed content"""
    return {
        "Meta_Description": get_meta_description(soup),
        "Headings": get_headings(soup),
        "Image_URLs": get_image_urls(soup),
        "Page_Content": (
            get_page_content(soup)
            if get_page_content(soup) and len(get_page_content(soup)) != 0
            else ""
        ),
    }


def get_meta_description(soup):
    meta = soup.find("meta", attrs={"name": "description"})
    return meta["content"] if meta and meta.get("content") else "N/A"


def get_headings(soup):
    return [
        f"h{level}: {h.get_text(strip=True)}"
        for level in range(1, 7)
        for h in soup.find_all(f"h{level}")
        if h.get_text(strip=True)
    ]


def get_image_urls(soup):
    return [
        resolve_relative_url(soup, img["src"]) for img in soup.find_all("img", src=True)
    ]


def get_page_content(soup):
    return [
        p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
    ]


def resolve_relative_url(soup, url):
    """Convert relative URLs to absolute"""
    base = soup.find("base", href=True)
    base_url = (
        base["href"]
        if base
        else soup.original_url if hasattr(soup, "original_url") else ""
    )
    return requests.compat.urljoin(base_url, url)


def error_response(error_msg):
    """Return standardized error response"""
    return {
        "Meta_Description": error_msg,
        "Headings": [],
        "Image_URLs": [],
        "Page_Content": [],
    }
