import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.utils import load_id, save_id, setup_logging
from stages.story_generator import generate_story

# Simple in-memory cache for ongoing requests
request_cache = {}
result_cache = {}  # Maps cache_key (hash) to completed results
processing_events = {}  # Maps cache_key to asyncio.Event for waiting requests

# Global lock to ensure only one story generation runs at a time
story_generation_lock = asyncio.Lock()

# Initialize global logger
LOGGER = None

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


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global LOGGER
    try:
        # Initialize a basic logger if setup_logging fails
        import logging

        LOGGER = logging.getLogger(__name__)
        LOGGER.setLevel(logging.INFO)

        # Create console handler if no handlers exist
        if not LOGGER.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            LOGGER.addHandler(handler)

        LOGGER.info("Application startup completed")

        # Clean up any stale cache on startup
        cleanup_expired_cache()

    except Exception as e:
        print(f"Warning: Logger initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        # Signal all pending requests to stop
        for event in processing_events.values():
            event.set()

        # Clear all caches
        request_cache.clear()
        result_cache.clear()
        processing_events.clear()

        if LOGGER:
            LOGGER.info("Application shutdown completed")
        else:
            print("Application shutdown completed")

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Error during shutdown: {e}")
        else:
            print(f"Error during shutdown: {e}")


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
    """Clean up expired cache entries with improved error handling"""
    try:
        current_time = time.time()
        expired_keys = []

        # Clean up request cache (entries older than 30 minutes)
        for cache_key, cache_entry in list(request_cache.items()):
            try:
                if current_time - cache_entry["start_time"] > 1800:  # 30 minutes
                    expired_keys.append(cache_key)
            except (KeyError, TypeError) as e:
                # Handle corrupted cache entries
                expired_keys.append(cache_key)
                if LOGGER:
                    LOGGER.warning(
                        f"Corrupted cache entry found: {cache_key}, error: {e}"
                    )

        for key in expired_keys:
            try:
                if key in request_cache:
                    del request_cache[key]
                # Also clean up corresponding processing events
                if key in processing_events:
                    processing_events[key].set()  # Signal any waiting requests
                    del processing_events[key]
            except Exception as e:
                if LOGGER:
                    LOGGER.error(f"Error cleaning up cache key {key}: {e}")

        # Clean up result cache with better size management
        if len(result_cache) > 50:  # Reduced threshold for better memory management
            try:
                # Remove oldest 40% of entries for more aggressive cleanup
                keys_to_remove = list(result_cache.keys())[: len(result_cache) * 2 // 5]
                for key in keys_to_remove:
                    if key in result_cache:
                        del result_cache[key]
                if LOGGER:
                    LOGGER.info(
                        f"Cleaned up {len(keys_to_remove)} result cache entries"
                    )
            except Exception as e:
                if LOGGER:
                    LOGGER.error(f"Error during result cache cleanup: {e}")

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Critical error in cache cleanup: {e}")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    """Health check endpoint to monitor service status"""
    try:
        # Basic health checks
        cache_size = len(result_cache)
        active_requests = len(request_cache)

        return {
            "status": "healthy",
            "cache_size": cache_size,
            "active_requests": active_requests,
            "timestamp": time.time(),
        }
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Better OPTIONS handler
@app.options("/stories")
async def stories_options():
    return Response(
        content='{"message": "OK"}',
        media_type="application/json",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
            "Cache-Control": "public, max-age=86400",
        },
    )


# Global dictionary to track client retry attempts
client_retry_tracker = {}


@app.post("/stories")
async def create_story(story_request: StoryRequest, request: Request) -> Any:

    start_time = time.time()

    # Get client identifier for retry tracking
    client_id = get_unique_identifier(request)
    current_time = time.time()

    # Clean up expired retry tracking entries (older than 5 minutes)
    expired_clients = [
        cid
        for cid, data in client_retry_tracker.items()
        if current_time - data.get("last_attempt", 0) > 300
    ]
    for cid in expired_clients:
        del client_retry_tracker[cid]

    # Track client retry attempts
    if client_id not in client_retry_tracker:
        client_retry_tracker[client_id] = {"count": 0, "last_attempt": current_time}
    else:
        client_data = client_retry_tracker[client_id]
        time_since_last = current_time - client_data["last_attempt"]

        # Reset count if enough time has passed (5 minutes)
        if time_since_last > 300:
            client_data["count"] = 0

        client_data["count"] += 1
        client_data["last_attempt"] = current_time

        # Implement exponential backoff for rapid retries
        if client_data["count"] > 1 and time_since_last < 30:  # Less than 30 seconds
            backoff_time = min(2 ** (client_data["count"] - 1), 60)  # Max 60 seconds
            raise HTTPException(
                status_code=429,
                detail=f"Too many rapid requests. Please wait {backoff_time} seconds before retrying.",
                headers={"Retry-After": str(backoff_time)},
            )

    # Clean up expired cache entries before processing
    cleanup_expired_cache()

    # Create a cache key based on the request parameters
    cache_key = hashlib.md5(
        f"{story_request.query.strip()}_{story_request.web}_{story_request.result_count_per_page}_{story_request.country_code}_{story_request.num_pages}".encode()
    ).hexdigest()

    # Create a client-specific cache key for better retry handling
    client_cache_key = f"{client_id}_{cache_key}"

    # First, check if there's already a cached result for this request
    cached_result = get_cached_result(cache_key)
    if cached_result:
        # Reset retry count on successful cache hit
        if client_id in client_retry_tracker:
            client_retry_tracker[client_id]["count"] = 0
        return cached_result

    # Check if another story generation is already in progress (regardless of cache key)
    if story_generation_lock.locked():
        # For browser retries, provide more informative response
        if (
            client_id in client_retry_tracker
            and client_retry_tracker[client_id]["count"] > 1
        ):
            raise HTTPException(
                status_code=202,
                detail="Your request is being processed. Please wait and avoid refreshing the page.",
                headers={"Retry-After": "30"},
            )

        # Wait for the ongoing story generation to complete
        try:
            await story_generation_lock.acquire()
            story_generation_lock.release()

            # After waiting, check if our specific request now has a cached result
            cached_result = get_cached_result(cache_key)
            if cached_result:
                # Reset retry count on successful result
                if client_id in client_retry_tracker:
                    client_retry_tracker[client_id]["count"] = 0
                return cached_result

        except Exception:
            pass

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
        # Acquire the global lock to ensure only one story generation runs at a time
        async with story_generation_lock:
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
            try:
                # Add timeout to prevent indefinite hanging
                results = await asyncio.wait_for(
                    loop.run_in_executor(
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
                    ),
                    timeout=1800,  # 30 minutes timeout
                )
            except asyncio.TimeoutError:
                if LOGGER:
                    LOGGER.error(f"Story generation timed out for query: {query}")

                # Provide different timeout messages based on retry count
                if (
                    client_id in client_retry_tracker
                    and client_retry_tracker[client_id]["count"] > 1
                ):
                    detail = "Request timed out after multiple attempts. The query may be too complex. Please try a simpler query or wait before retrying."
                else:
                    detail = "Story generation timed out. Please try again with a simpler query or wait a moment before retrying."

                raise HTTPException(
                    status_code=504, detail=detail, headers={"Retry-After": "60"}
                )

            # Check if results are valid
            if results is None:
                if LOGGER:
                    LOGGER.error(f"Story generation returned None for query: {query}")
                raise HTTPException(
                    status_code=500,
                    detail="Story generation failed to produce results. Please try again.",
                )

            # Cache the successful result
            cache_result(cache_key, results)

            # Reset client retry count on successful completion
            if client_id in client_retry_tracker:
                client_retry_tracker[client_id]["count"] = 0

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

        # Log the error for debugging
        if LOGGER:
            LOGGER.error(f"Story generation failed: {str(e)}")

        # Return a proper HTTP error instead of re-raising
        raise HTTPException(
            status_code=500, detail=f"Story generation failed: {str(e)}"
        )

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
        if LOGGER:
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
