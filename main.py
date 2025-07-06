import hashlib
import os
import re
import time
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from common.utils import load_id, save_id, setup_logging
from stages.story_generator import generate_story

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_unique_identifier(request: Request) -> str:
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    unique_id = f"{client_ip}-{user_agent}"

    hash_value = hashlib.md5(unique_id.encode()).hexdigest()
    return hash_value


def generate_unique_id(file_name):
    return file_name + str(int(time.time() * 1000))


def ensure_directory_exists(path: str):
    os.makedirs(path, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/stories")
def get_stories(
    request: Request,
    query: Optional[str] = Query(
        None,
        min_length=1,
        description="Search term to filter items by name or description",
    ),
    web: Optional[str] = Query("pewresearch.org", description="Website to search"),
    result_count_per_page: Optional[int] = Query(
        1, ge=1, description="Number of results to retrieve per page"
    ),
    country_code: Optional[str] = Query(
        "sg", description="Country code to search from"
    ),
    num_pages: Optional[int] = Query(
        2, ge=1, description="Number of max pages per search"
    ),
) -> Any:

    start_time = time.time()
    try:
        query = query.strip()
        file_name = re.sub(r"[^a-zA-Z0-9\s]", "", query)
        id = load_id()
        file_name = f"{id}_{file_name}"
        unique_id = generate_unique_id(file_name)
        file_path = os.path.join("", f"results/{file_name}")
        save_id(id + 1)
        ensure_directory_exists(file_path)

        iterations = 1
        search_result_file = f"{file_path}/google_search_results.csv"
        output_path = f"{file_path}/story.html"
        results_path = f"{file_path}/JsonOutputs"
        log_path = f"{file_path}/logs_{unique_id}.log"

        global LOGGER
        LOGGER = setup_logging(log_path)
        LOGGER.info("Application started.")

        results = generate_story(
            search_query=query,
            web=web,
            result_count_per_page=result_count_per_page,
            iterations=iterations,
            search_result_file=search_result_file,
            output_path=output_path,
            results_path=results_path,
            country_code=country_code,
            num_pages=num_pages,
        )
        return results

    finally:
        save_id(id + 1)
        total_time = time.time() - start_time
        LOGGER.info(f"Total time taken: {total_time} seconds")
