import hashlib
import json
import os
import re
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
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
