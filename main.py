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
processing_events = {}  # Maps cache_key to asyncio.Event for waiting requests

app = FastAPI()


class StoryRequest(BaseModel):
    query: str
    web: str = "pewresearch.org"
    result_count_per_page: int = 1
    country_code: str = "sg"
    num_pages: int = 2


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "https://compendia-new.onrender.com",
        "https://compendia.onrender.com",
    ],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
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


def cleanup_expired_cache():
    """Clean up expired cache entries"""
    current_time = time.time()
    expired_keys = []

    # Clean up request cache (entries older than 30 minutes)
    for cache_key, cache_entry in request_cache.items():
        if current_time - cache_entry["start_time"] > 1800:  # 30 minutes
            expired_keys.append(cache_key)

    for key in expired_keys:
        if key in request_cache:
            del request_cache[key]
        # Also clean up corresponding processing events
        if key in processing_events:
            processing_events[key].set()  # Signal any waiting requests
            del processing_events[key]

    # Clean up result cache (entries older than 24 hours)
    # Note: This is a simple implementation. In production, you might want to store timestamps with results
    if len(result_cache) > 100:  # Simple size-based cleanup
        # Remove oldest 20% of entries (this is a simplified approach)
        keys_to_remove = list(result_cache.keys())[: len(result_cache) // 5]
        for key in keys_to_remove:
            if key in result_cache:
                del result_cache[key]


@app.get("/")
async def root():
    return {"message": "Hello World"}


class StoryRequest(BaseModel):
    query: str
    web: str = "pewresearch.org"
    result_count_per_page: int = 1
    country_code: str = "sg"
    num_pages: int = 2


@app.options("/stories")
async def stories_options():
    """Handle preflight OPTIONS requests for /stories endpoint"""
    return {"message": "OK"}


@app.post("/stories")
async def create_story(story_request: StoryRequest) -> Any:

    start_time = time.time()

    # Clean up expired cache entries before processing
    cleanup_expired_cache()

    # Create a cache key based on the request parameters
    cache_key = hashlib.md5(
        f"{story_request.query.strip()}_{story_request.web}_{story_request.result_count_per_page}_{story_request.country_code}_{story_request.num_pages}".encode()
    ).hexdigest()

    # First, check if there's already a cached result for this request
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Check if this request is already being processed
    if cache_key in request_cache:
        current_time = time.time()
        cache_entry = request_cache[cache_key]

        # If request is still being processed (within 30 minutes)
        if current_time - cache_entry["start_time"] < 1800:  # 30 minutes
            # Wait for the ongoing request to complete and return its result
            try:
                # Wait for the processing to complete (with timeout)
                if cache_key in processing_events:
                    await asyncio.wait_for(
                        processing_events[cache_key].wait(), timeout=1800
                    )  # 30 minutes max

                # Check if result is now available after waiting
                cached_result = get_cached_result(cache_key)
                if cached_result:
                    return cached_result

                # If no result after waiting, the processing likely failed
                raise HTTPException(
                    status_code=500,
                    detail="Request processing completed but no result was generated. Please try again.",
                )

            except asyncio.TimeoutError:
                # If we've waited too long, return 429
                raise HTTPException(
                    status_code=429,
                    detail="Request processing is taking longer than expected. Please try again later.",
                )
        else:
            # Remove expired cache entry
            del request_cache[cache_key]

    # Create an event for waiting requests
    processing_events[cache_key] = asyncio.Event()

    # Add this request to the cache with processing status
    request_cache[cache_key] = {
        "start_time": start_time,
        "query": story_request.query.strip(),
        "status": "processing",
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

        # Mark request as completed and notify waiting requests
        if cache_key in request_cache:
            request_cache[cache_key]["status"] = "completed"

        # Signal waiting requests that processing is complete
        if cache_key in processing_events:
            processing_events[cache_key].set()

        return results

    except Exception as e:
        # Mark request as failed and remove from cache
        if cache_key in request_cache:
            del request_cache[cache_key]

        # Signal waiting requests that processing failed
        if cache_key in processing_events:
            processing_events[cache_key].set()

        raise e

    finally:
        # Clean up: Remove request from cache when completed (if not already removed)
        if (
            cache_key in request_cache
            and request_cache[cache_key].get("status") == "completed"
        ):
            del request_cache[cache_key]

        # Clean up processing events
        if cache_key in processing_events:
            del processing_events[cache_key]

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
