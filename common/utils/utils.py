import ast
import json
import os

from common.config import DATE_CONFIGS
from dateutil import parser

from .console import console
from .timing_logger import LOGGER

FILE_PATH = "id_store.json"


def write_to_json(data, folder_path, file_name):
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    file_name = f"{folder_path}/{file_name}"
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)


def read_file(file_path):
    content = None
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        console.print(f"Error: The file {file_path} was not found.")
    except IOError as e:
        console.print(f"Error reading file {file_path}: {e}")
    return content


def read_text_files(folder_path):
    file_contents = []

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".text"):  # Check if it's a text file
            file_path = os.path.join(folder_path, filename)

            # Open and read the file
            content = read_file(file_path)
            file_contents.append(content)

    return file_contents


def merge_arrays(arr1, arr2):
    merged_dict = {}

    # Merge first array
    for obj in arr1:
        if isinstance(obj, dict):  # Ensure object is a dictionary
            merged_dict[obj["order_id"]] = obj.copy()

    # Merge second array
    for obj in arr2:
        if isinstance(obj, dict):  # Ensure object is a dictionary
            if obj["order_id"] in merged_dict:
                merged_dict[obj["order_id"]].update(obj)
            else:
                merged_dict[obj["order_id"]] = obj.copy()

    return list(merged_dict.values())


def chunk_array(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_convert_to_list(s):
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, list):  # Ensure it's actually a list
            return arr
        else:
            raise ValueError("Converted value is not a list")
    except (SyntaxError, ValueError) as e:
        console.print(f"Error: {e}")
        return None  # Return None or an empty list []


def load_id():
    """Load the stored ID from the file, or return a default ID if not found."""
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            data = json.load(file)
            return data.get("id", 0)
    return 0  # Default ID if file does not exist


def save_id(new_id):
    """Save the updated ID to the file."""
    with open(FILE_PATH, "w") as file:
        json.dump({"id": new_id}, file)


def convert_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date
    except ValueError:
        console.print(f"Error: {date_str} is not a valid date")
        return None


def check_is_date(date_str):
    try:
        parser.parse(date_str)
        return True
    except Exception as e:
        return False


def format_date(
    date_str: str,
    date_format: str = DATE_CONFIGS["DATE_FORMAT"],
):
    try:
        date = parser.parse(date_str)
        formatted_date = date.strftime(date_format)
        return formatted_date
    except Exception as e:
        return "Unknown"
