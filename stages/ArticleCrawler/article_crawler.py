import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from common.config import PROMPTS
from common.gpt_helper import GPTHelper
from common.utils import console, write_to_json
from common.utils.timing_logger import LOGGER, log_execution_time
from crawler.search_processor import get_google_search_results
from models.models import SearchQueryList

gpt_helper = GPTHelper()


@log_execution_time
def generate_search_query(search_query):

    system_prompt = PROMPTS["GENERATE_SEARCH_QUERIES"]

    user_prompt = f"""
    
    ####
    {search_query}
    ####
    
    """
    console.print("[bold yellow]Generating search queries...[/bold yellow]")
    queries = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=SearchQueryList
    )
    console.print("[bold green]Generating search queries completed![/bold green]")
    new_queries = json.loads(queries)
    new_queries["search_queries"].append(search_query)
    return new_queries


@log_execution_time
def consolidate_unique_articles(directory, output_filename="unique_articles.csv"):

    # Collect all the CSV files in the specified directory
    csv_files = glob.glob(f"{directory}/google_search_results_*.csv")

    all_data = []

    # Load each CSV file and append its data to the list
    for file in csv_files:
        try:
            data = pd.read_csv(file)
            all_data.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine all the data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Drop duplicates based on the 'Link' column
    unique_data = combined_data.drop_duplicates(subset="Link", keep="first")

    # Drop rows where 'Page_Content' is empty or NaN
    unique_data = unique_data.dropna(subset=["Page_Content"])
    # unique_data = unique_data[unique_data["Page_Content"].str.strip() != ""]
    unique_data = unique_data[
        (unique_data["Page_Content"].str.strip() != "")
        & (unique_data["Page_Content"].str.strip() != "[]")
    ]

    unique_data["id"] = range(len(unique_data))

    # Save the unique data to a new CSV file
    unique_data.to_csv(output_filename, index=False)
    console.print(f"Unique articles saved to {output_filename}")


@log_execution_time
def collect_search_results(
    query,
    web,
    num_results=1,
    csv_filename="google_search_results.csv",
    country_code="sg",
):
    queries = generate_search_query(query)
    directory = csv_filename.rsplit("/", 1)[0]
    write_to_json(queries, directory, "queries.json")

    def process_query(q, i):
        try:
            new_file_path = f"{directory}/google_search_results_{i}.csv"
            qr = f"site:{web} {q}"
            get_google_search_results(
                qr,
                num_results=num_results,
                csv_filename=new_file_path,
                country_code=country_code,
            )
            return True
        except Exception as e:
            LOGGER.error(f"Failed query {q}: {str(e)}")
            return False

    # Controlled parallelism
    with ThreadPoolExecutor(max_workers=10) as executor:  # Limit concurrency
        futures = {
            executor.submit(process_query, q, i): (i, q)
            for i, q in enumerate(queries["search_queries"])
        }

        # Monitor progress
        for future in as_completed(futures):
            i, q = futures[future]
            if future.result():
                print(f"Completed query {i}: {q}")
            else:
                print(f"Failed query {i}: {q}")

    consolidate_unique_articles(directory, csv_filename)
    return
