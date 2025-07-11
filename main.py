import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.utils import load_id, save_id, setup_logging
from stages.story_generator import generate_story

# Simple in-memory cache for ongoing requests
request_cache = {}
result_cache = {}  # Maps cache_key (hash) to completed results

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "https://compendia-new.onrender.com",
        "https://compendia.onrender.com/",
    ],  # Allows all origins
    allow_credentials=True,  # Disable credentials to prevent preflight issues
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight responses for 1 hour
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


def get_cached_result(cache_key: str) -> Any:
    return result_cache.get(cache_key)


def cache_result(cache_key: str, result: Any) -> None:
    result_cache[cache_key] = result


@app.get("/")
async def root():
    return {"message": "Hello World"}


class StoryRequest(BaseModel):
    query: str
    web: str = "pewresearch.org"
    result_count_per_page: int = 1
    country_code: str = "sg"
    num_pages: int = 2


@app.post("/stories")
async def create_story(story_request: StoryRequest) -> Any:

    start_time = time.time()

    # Create a cache key based on the request parameters
    cache_key = hashlib.md5(
        f"{story_request.query.strip()}_{story_request.web}_{story_request.result_count_per_page}_{story_request.country_code}_{story_request.num_pages}".encode()
    ).hexdigest()

    # Check if this request is already being processed
    if cache_key in request_cache:
        current_time = time.time()
        cache_entry = request_cache[cache_key]

        # If request is still being processed (within 30 minutes)
        if current_time - cache_entry["start_time"] < 1800:  # 30 minutes
            # Check if there's a cached result for this request
            cached_result = get_cached_result(cache_key)
            if cached_result:
                return cached_result
            else:
                raise HTTPException(
                    status_code=429,
                    detail="A similar request is already being processed. Please wait or try a different query.",
                )
        else:
            # Remove expired cache entry
            del request_cache[cache_key]

    # Add this request to the cache
    request_cache[cache_key] = {
        "start_time": start_time,
        "query": story_request.query.strip(),
    }

    try:
        query = story_request.query.strip()
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

        # Run the CPU-intensive story generation in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: generate_story(
                search_query=query,
                web=story_request.web,
                result_count_per_page=story_request.result_count_per_page,
                iterations=iterations,
                search_result_file=search_result_file,
                output_path=output_path,
                results_path=results_path,
                country_code=story_request.country_code,
                num_pages=story_request.num_pages,
            ),
        )

        # Cache the successful result
        cache_result(cache_key, results)

        return results

    finally:
        # Remove request from cache when completed
        if cache_key in request_cache:
            del request_cache[cache_key]

        save_id(id + 1)
        total_time = time.time() - start_time
        LOGGER.info(f"Total time taken: {total_time} seconds")


@app.get("/results/{folder_identifier}")
def get_final_result(folder_identifier: str) -> Any:
    """
    Get the final successful result from the results folder.

    Args:
        folder_identifier: Either the folder name or the ID of the result

    Returns:
        The content of 20_final_styled_analysis.json if the process was successful
    """
    results_base_path = "results"

    # Check if folder_identifier is a direct folder name or needs to be found by ID
    potential_folder_path = os.path.join(results_base_path, folder_identifier)

    if not os.path.exists(potential_folder_path):
        # Try to find folder by ID prefix
        if not os.path.exists(results_base_path):
            raise HTTPException(status_code=404, detail="Results directory not found")

        # List all folders and find the one that starts with the given ID
        found_folder = None
        for folder_name in os.listdir(results_base_path):
            folder_path = os.path.join(results_base_path, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith(
                f"{folder_identifier}_"
            ):
                found_folder = folder_name
                break

        if found_folder is None:
            raise HTTPException(
                status_code=404,
                detail=f"No result folder found for identifier: {folder_identifier}",
            )

        potential_folder_path = os.path.join(results_base_path, found_folder)

    # Check if the final analysis file exists (indicates successful completion)
    final_analysis_path = os.path.join(
        potential_folder_path, "JsonOutputs", "20_final_styled_analysis.json"
    )

    if not os.path.exists(final_analysis_path):
        raise HTTPException(
            status_code=404,
            detail="Process not completed successfully. The 20_final_styled_analysis.json file was not found.",
        )

    try:
        with open(final_analysis_path, "r", encoding="utf-8") as file:
            final_result = json.load(file)

        return {
            "status": "success",
            "folder_name": os.path.basename(potential_folder_path),
            "data": final_result,
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail="Error reading the final analysis file"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/recent")
def list_results() -> Any:
    """
    List all completed results ordered by ID in descending order.

    Returns:
        A list of completed result folders ordered by ID descending
    """
    results_base_path = "results"

    if not os.path.exists(results_base_path):
        return {"results": []}

    results = []

    for folder_name in os.listdir(results_base_path):
        folder_path = os.path.join(results_base_path, folder_name)

        if os.path.isdir(folder_path):
            # Check if the process completed successfully
            final_analysis_path = os.path.join(
                folder_path, "JsonOutputs", "20_final_styled_analysis.json"
            )
            is_completed = os.path.exists(final_analysis_path)

            # Only include completed results
            if is_completed:
                # Extract ID from folder name (assuming format: "ID_description")
                folder_id = (
                    folder_name.split("_")[0] if "_" in folder_name else folder_name
                )

                # Remove ID from query (everything after first underscore)
                query = (
                    "_".join(folder_name.split("_")[1:])
                    if "_" in folder_name
                    else folder_name
                )

                results.append(
                    {
                        "id": folder_id,
                        "query": query,
                        "is_completed": is_completed,
                        "has_final_analysis": is_completed,
                    }
                )

    # Sort results by ID in descending order
    try:
        results.sort(key=lambda x: int(x["id"]), reverse=True)
    except ValueError:
        # If ID is not numeric, sort alphabetically in descending order
        results.sort(key=lambda x: x["id"], reverse=True)

    return {"results": results}
