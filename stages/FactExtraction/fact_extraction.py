import concurrent.futures
import json
from typing import List

import tiktoken
from common.config import MODEL_CONFIG, PROMPTS, THEME_CONFIGS
from common.gpt_helper import GPTHelper
from common.utils import console
from common.utils.timing_logger import LOGGER, log_execution_time
from common.utils.utils import check_is_date, format_date, safe_convert_to_list
from models.models import (
    Article_v2,
    ArticleDataFacts,
    ArticleDataFactVisData,
    ValidationOutput,
)

gpt_helper = GPTHelper()

MAX_OUTPUT_TOKENS = MODEL_CONFIG["MAX_OUTPUT_TOKENS"]
TOKEN_SAFETY_MARGIN = MODEL_CONFIG["TOKEN_SAFETY_MARGIN"]
MAX_POSSIBLE_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS - TOKEN_SAFETY_MARGIN
THEME_COLORS = THEME_CONFIGS["COLORS"]


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")  # Compatible with GPT-4o
    return len(encoding.encode(text))


def prepare_chunks(article_text):
    try:
        paragraphs = safe_convert_to_list(article_text)
    except:
        paragraphs = article_text.split("\n")

    chunks: List[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if count_tokens(current_chunk + paragraph) < MAX_POSSIBLE_OUTPUT_TOKENS - 500:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def select_date(original_date, extracted_date):
    if check_is_date(original_date):
        return format_date(original_date)
    if check_is_date(extracted_date):
        return format_date(extracted_date)
    return "Unknown"


@log_execution_time
def extract_and_filter_paragraphs(id, title, date, article_text, search_query):
    system_prompt = PROMPTS["EXTRACT_FILTER_PARA"]

    user_prompt = """
    RAW TITLE:
    {title}

    RAW DATE:
    {date}

    DOCUMENT INPUT:
    {article_text}

    SEARCH QUERY:
    {search_query}
    """

    formatted_user_prompt = user_prompt.format(
        title=title, date=date, article_text=article_text, search_query=search_query
    )

    total_tokens = count_tokens(formatted_user_prompt) + count_tokens(system_prompt)

    def process_chunk(text_chunk: str, chunk_number: int):
        user_prompt_chunk = user_prompt.format(
            title=title, date=date, article_text=text_chunk, search_query=search_query
        )
        result = gpt_helper.ask_gpt_with_response_format(
            system_prompt, user_prompt_chunk, response_format=Article_v2
        )

        if result is None:
            return None

        try:
            result = json.loads(result)
            result["chunk_number"] = chunk_number
            return result
        except Exception as e:
            LOGGER.error(f"{id} - {title} - JSON parsing failed {e}")
            console.print(
                f"[bold red]Error: {id} - {title} - Failed to parse JSON response {e}[/bold red]"
            )
            return None

    final_result = None

    if total_tokens < MAX_POSSIBLE_OUTPUT_TOKENS:
        console.print(
            f"[bold yellow]{id} - {title} - Processing document in optimized single pass...[/bold yellow]"
        )
        LOGGER.info(f"{id} - {title} - extract_and_filter_paragraphs started")

        result = gpt_helper.ask_gpt_with_response_format(
            system_prompt, formatted_user_prompt, response_format=Article_v2
        )
        try:
            obj = json.loads(result)
            console.print(
                f"[bold green]{id} - {title} - extract_and_filter_paragraphs Processing Completed![/bold green]"
            )
            LOGGER.info(
                f"{id} - {title} - extract_and_filter_paragraphs Processing Completed"
            )
            final_result = obj
        except Exception as e:
            console.print(
                f"[bold red]Error: {id} - {title} - Failed to parse JSON response {e}[/bold red]"
            )
            LOGGER.error(f"{id} - {title} - JSON parsing failed")
            return None
    else:
        console.print(
            f"[bold yellow]{id} - {title} - Processing document in optimized single pass...[/bold yellow]"
        )
        LOGGER.info(f"{id} - {title} - extract_and_filter_paragraphs started")

        chunks = prepare_chunks(article_text)
        all_filtered_paragraphs = []
        retrieved_title = None
        retrieved_date = None
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    futures.append(executor.submit(process_chunk, chunk, i))

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        if result.get("chunk_number") == 0:
                            retrieved_title = result.get("title")
                            retrieved_date = result.get("date")
                        for para in result.get("paragraphs", []):
                            all_filtered_paragraphs.append(para)

            final_result = {
                "title": retrieved_title,
                "date": retrieved_date,
                "paragraphs": all_filtered_paragraphs,
            }
            console.print(
                f"[bold green]{id} - {title} - extract_and_filter_paragraphs Processing completed successfully![/bold green]"
            )
        except Exception as e:
            console.print(
                f"[bold red]Error: {id} - {title} - Failed to parse JSON response {e}[/bold red]"
            )
            LOGGER.error(f"{id} - {title} - JSON parsing failed")
            return None

    final_result["date"] = select_date(date, final_result["date"])
    return final_result


@log_execution_time
def get_data_facts(id, title, filtered_paragraphs):
    system_prompt = PROMPTS["EXTRACT_FACTS"]

    user_prompt = f"""
    ####
    {filtered_paragraphs["paragraphs"]}
    ####
    """

    console.print(
        f"[bold yellow]{id} - {title} - Extracting Data Facts...[/bold yellow]"
    )
    data_facts_with_para = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleDataFacts
    )

    if data_facts_with_para is None:
        console.print(
            f"[bold red]{id} - {title} - Fact extraction failed. No relevant fact found.[/bold red]"
        )
        LOGGER.error(
            f"{id} - {title} - Fact extraction failed. No relevant fact found."
        )
        return None

    try:
        obj = json.loads(data_facts_with_para)
        console.print(
            f"[bold green]{id} - {title} - Extracting Data Facts Completed![/bold green]"
        )
        return obj
    except Exception as e:
        console.print(
            f"[bold red]{id} - {title} - Failed to parse the response into JSON.[/bold red] {e}"
        )
        LOGGER.error(f"{id} - {title} - Failed to parse the response into JSON. {e}")
        return None


def remove_facts_with_empty_vis(data):
    data["data_facts_with_vis_data"] = [
        {**item, "facts": [fact for fact in item["facts"] if fact.get("vis_data")]}
        for item in data.get("data_facts_with_vis_data", [])
    ]

    data["data_facts_with_vis_data"] = [
        item for item in data["data_facts_with_vis_data"] if item["facts"]
    ]

    return data if data["data_facts_with_vis_data"] else None


# -------------- Start: Get Data Values - all para ----------
@log_execution_time
def get_data_values(id, title, date, data_fact_with_related_sentence, article):
    system_prompt = PROMPTS["DATA_EXTRACTION"]

    user_prompt = f"""
        ####
        {data_fact_with_related_sentence}
        ####

        ****
        Published date: {date}
        
        {article}
        ****
        """

    console.print(
        f"[bold yellow]{id} - {title} - Extracting Data Values...[/bold yellow]"
    )
    data_fact_with_vis_data = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleDataFactVisData
    )

    if data_fact_with_vis_data is None:
        console.print(
            f"[bold red]{id} - {title} - Data value extraction failed. No relevant data values found.[/bold red]"
        )
        LOGGER.error(
            f"{id} - {title} - Data value extraction failed. No relevant data values found."
        )
        return None

    try:
        obj = json.loads(data_fact_with_vis_data)
        console.print(
            f"[bold green]{id} - {title} - Extracting Data Values Completed![/bold green]"
        )
        removed_empty_vis = remove_facts_with_empty_vis(obj)
        return removed_empty_vis
    except Exception as e:
        console.print(
            f"[bold red]{id} - {title} - Failed to parse the response into JSON.[/bold red] {e}"
        )
        LOGGER.error(f"{id} - {title} - Failed to parse the response into JSON. {e}")
        return None


def update_has_error(data):
    has_errors = any(
        fact.get("error")
        for entry in data.get("vis_data_error", [])
        for fact in entry.get("facts", [])
    )

    data["has_error"] = has_errors
    return data


#  Better with temperature 0.7
@log_execution_time
def validate_data_extraction(id, title, extracted_data):
    system_prompt = PROMPTS["VALIDATE_DATA_EXTRACTION"]

    user_prompt = f"""
    ####
    {extracted_data}
    ####

    """

    console.print(
        f"[bold yellow]{id} - {title} - Validating Extracted Data...[/bold yellow]"
    )
    data_errors = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, temperature=0.7, response_format=ValidationOutput
    )

    if data_errors is None:
        console.print(f"[bold red]{id} - {title} - Data validation failed.[/bold red]")
        LOGGER.error(f"{id} - {title} - Data validation failed.")
        return None

    try:
        obj = json.loads(data_errors)
        console.print(
            f"[bold green]{id} - {title} - Validating Extracted Data Completed![/bold green]"
        )
        # updated_data = update_has_error(obj)
        return obj
    except json.JSONDecodeError:
        console.print(
            f"[bold red]{id} - {title} - Failed to parse the response into JSON.[/bold red]"
        )
        LOGGER.error(f"{id} - {title} - Failed to parse the response into JSON.")
        return None


@log_execution_time
def refine_data(id, title, vis_data_erros):
    system_prompt = PROMPTS["REFINE_DATA_EXTRACTION"]

    user_prompt = f"""
    ####
    {vis_data_erros}
    ####
    """
    console.print(
        f"[bold yellow]{id} - {title} - Refining Extracted Data...[/bold yellow]"
    )
    refined_data = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleDataFactVisData
    )

    if refined_data is None:
        console.print(f"[bold red]{id} - {title} - Refining failed.[/bold red]")
        LOGGER.error(f"{id} - {title} - Refining failed.")
        return None

    try:
        obj = json.loads(refined_data)
        console.print(
            f"[bold green]{id} - {title} - Refining Extracted Data Completed![/bold green]"
        )
        return obj
    except json.JSONDecodeError:
        console.print(
            f"[bold red]{id} - {title} - Failed to parse the response into JSON.[/bold red]"
        )
        LOGGER.error(f"{id} - {title} - Failed to parse the response into JSON.")
        return None
