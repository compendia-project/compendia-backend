import csv
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from serper import get_search_results, get_webpage
from common.utils.timing_logger import LOGGER, log_execution_time


@log_execution_time
def get_google_search_results(query, num_results=10, csv_filename="search_results.csv"):

    response = get_search_results(query, num_results)

    # Open CSV file for writing
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Title",
            "Link",
            "Date",
            "Meta_Description",
            "Headings",
            "Image_URLs",
            "Page_Content",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in response[0]["organic"]:
            title = result["title"]
            link = result["link"]
            snippet = result["snippet"]
            position = result["position"]
            date = None
            if "date" in result:
                date = result["date"]

            page_data = crawl_page(link)

            writer.writerow(
                {
                    "Title": title,
                    "Link": link,
                    "Date": date,
                    "Meta_Description": snippet,
                    "Headings": "; ".join(page_data["Headings"]),
                    "Image_URLs": "; ".join(page_data["Image_URLs"]),
                    "Page_Content": page_data["Page_Content"],
                }
            )
            time.sleep(2)


def crawl_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        # Send a GET request to the page
        response = requests.get(url, headers=headers, timeout=10)

        # If the request was successful, parse the content
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return extract_specific_data(soup)
        else:
            website = get_webpage(url)
            if website is not None:
                page_content = website["text"]
                if len(page_content) < 200:
                    page_content = ""
                else:
                    page_content = page_content.split("\n\n")
                return {
                    "Meta_Description": website["metadata"]["description"],
                    "Headings": [],
                    "Image_URLs": [],
                    "Page_Content": page_content,
                }
            else:
                return {
                    "Meta_Description": f"Failed to retrieve page content. Status code: {response.status_code}",
                    "Headings": [],
                    "Image_URLs": [],
                    "Page_Content": "",
                }

    except Exception as e:
        return {
            "Meta_Description": f"An error occurred: {e}",
            "Headings": [],
            "Image_URLs": [],
            "Page_Content": "",
        }


def extract_specific_data(soup):
    """
    Extracts specific data elements from the BeautifulSoup-parsed HTML content.

    Returns a dictionary with:
    - Meta Description
    - Headings (h1, h2, h3, etc.)
    - Image URLs
    - Page Content
    """
    # Extract meta description
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = (
        meta_desc_tag["content"]
        if meta_desc_tag and "content" in meta_desc_tag.attrs
        else "N/A"
    )

    # Extract all headings (h1 to h6)
    headings = []
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                headings.append(f"h{level}: {heading_text}")

    # Extract all image URLs
    images = []
    for img in soup.find_all("img", src=True):
        img_url = img["src"]
        # Handle relative URLs
        if not img_url.startswith(("http://", "https://")):
            img_url = resolve_relative_url(soup, img_url)
        images.append(img_url)

    # Extract main content (all paragraph texts)
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    print(paragraphs)
    # page_content = " ".join(paragraphs)
    page_content = paragraphs

    return {
        "Meta_Description": meta_description,
        "Headings": headings,
        "Image_URLs": images,
        "Page_Content": page_content,
    }


def resolve_relative_url(soup, relative_url):
    """
    Resolves a relative URL to an absolute URL based on the soup's base URL.
    """
    base_tag = soup.find("base", href=True)
    if base_tag:
        base_url = base_tag["href"]
    else:
        # Fallback: Use the first link tag as base or default to empty
        parsed_url = urlparse(
            soup.original_encoding
            if hasattr(soup, "original_encoding")
            else "http://localhost"
        )
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return requests.compat.urljoin(base_url, relative_url)


# # Example usage
# if __name__ == "__main__":
#     query = "site:pewresearch.org Do adults getting vaccinated for COVID-19?"
#     get_google_search_results(
#         query, num_results=10, csv_filename="google_search_pew.csv"
#     )
#     print("Search results have been saved to 'google_search_results.csv'")
