import os

from dotenv import load_dotenv
from openai import OpenAI

from .config import MODEL_CONFIG
from .utils.console import console
from .utils.timing_logger import LOGGER

load_dotenv()


class GPTHelper:
    """Helper class for GPT API interactions"""

    MODEL = MODEL_CONFIG["DEFAULT_MODEL"]
    DEFAULT_TEMPERATURE = MODEL_CONFIG["DEFAULT_TEMPERATURE"]
    EMBEDDING_MODEL = MODEL_CONFIG["EMBEDDING_MODEL"]
    TOP_P = MODEL_CONFIG["TOP_P"]

    def __init__(self):
        """Initialize OpenAI client with API key"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_gpt(
        self,
        system_prompt,
        user_prompt,
        model=MODEL,
        temperature=DEFAULT_TEMPERATURE,
    ):
        """Basic GPT request without response format"""
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            console.print(f"Error from GPTs: {e}")
            return None

    def ask_gpt_with_response_format(
        self,
        system_prompt,
        user_prompt,
        model=MODEL,
        temperature=DEFAULT_TEMPERATURE,
        response_format=None,
        top_p=TOP_P,
    ):
        """GPT request with response format, returns content"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format=response_format,
                top_p=top_p,
            )
            response_content = completion.choices[0].message.content
            return response_content
        except Exception as e:
            console.print(f"Error from GPT: {e}")
            LOGGER.error(f"Error from GPT: {e}")
            return None

    def ask_gpt_with_response_format_parsed(
        self,
        system_prompt,
        user_prompt,
        model=MODEL,
        temperature=DEFAULT_TEMPERATURE,
        response_format=None,
    ):
        """GPT request with response format, returns parsed response"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format=response_format,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            console.print(f"Error from GPT: {e}")
            return None

    def get_embeddings(self, query, model=EMBEDDING_MODEL):
        query_embedding_response = self.client.embeddings.create(
            model=model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding

        return query_embedding
