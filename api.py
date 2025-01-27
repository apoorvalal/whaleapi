import json
import os
from typing import Dict, Generator, List, Optional, Union

import requests

######################################################################
# Environment variable for API key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Default temperature settings
DEFAULT_TEMP = 1.0
TEMPERATURE_MAP = {
    "General": DEFAULT_TEMP,
    "Programming/Math": 0.0,
    "Data Cleaning/Analysis": 1.0,
    "General Conversation": 1.3,
    "Translation": 1.3,
    "Creative Writing": 1.5,
}

# API endpoints
API_USER_BAL = "https://api.deepseek.com/user/balance"
API_CHAT_COM = API_CHAT_FIM = "https://api.deepseek.com/chat/completions"
API_CHAT_MOD = "https://api.deepseek.com/models"

# Default prompts
DEFAULT_SYS_PROMPT = "You are a helpful and concise assistant"
DEFAULT_USER_PROMPT = "Hello"

######################################################################


class WhaleAPI:
    def __init__(self, api_key: Optional[str] = DEEPSEEK_API_KEY):
        """Initialize the DeepSeekAPI with the provided API key."""
        if api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is missing")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def user_balance(self) -> Dict:
        """Fetch the user's balance."""
        response = requests.get(API_USER_BAL, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        prompt: Union[str, List[Dict]] = DEFAULT_USER_PROMPT,
        prompt_sys: str = DEFAULT_SYS_PROMPT,
        stream: bool = False,
        model: str = "deepseek-chat",
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a chat completion."""
        payload = {
            "model": model,
            "messages": [
                {"content": prompt_sys, "role": "system"},
                {"content": prompt, "role": "user"},
            ]
            if isinstance(prompt, str)
            else prompt,
            "stream": stream,
            "max_tokens": 2048,
            "temperature": TEMPERATURE_MAP["General"],
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 1,
            **kwargs,
        }
        response = self._post_request(API_CHAT_COM, payload, stream)
        if stream:
            return self._completion_impl(response, "chat")
        else:
            return response.json()["choices"][0].get("message", {}).get("content", "")

    def fim_completion(
        self,
        prompt: str = DEFAULT_USER_PROMPT,
        stream: bool = False,
        model: str = "deepseek-chat",
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a fill-in-the-middle (FIM) completion."""
        payload = {
            "model": model,
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": 1024,
            "temperature": TEMPERATURE_MAP["Programming/Math"],
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_p": 1,
            **kwargs,
        }

        response = self._post_request(API_CHAT_FIM, payload, stream)
        if stream:
            return self._completion_impl(response, "fim")
        else:
            return response.json()["choices"][0].get("message", {}).get("content", "")

    def get_models(self) -> List[str]:
        """Fetch available models."""
        response = requests.get(API_CHAT_MOD, headers=self.headers)
        response.raise_for_status()
        return [model["id"] for model in response.json()["data"]]

    ######################################################################
    def _post_request(
        self, api_url: str, payload: Dict, stream: bool
    ) -> requests.Response:
        """Internal method to handle POST requests."""
        response = requests.post(
            api_url, headers=self.headers, json=payload, stream=stream
        )
        if response.status_code >= 300:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")
        return response

    # silly gimmick, for completeness
    def _completion_impl(
        self, response: requests.Response, type_: str = "chat"
    ) -> Generator[str, None, None]:
        """Internal method to handle streaming completions."""
        for line in response.iter_lines():
            if line:  # Process non-empty lines
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    line = line[6:].strip()
                if not line or line == "[DONE]":
                    break
                decoded_line = json.loads(line)
                if "choices" in decoded_line:
                    finish_reason = decoded_line["choices"][0].get(
                        "finish_reason", None
                    )
                    if type_ == "chat":
                        delta_content = (
                            decoded_line["choices"][0]
                            .get("delta", {})
                            .get("content", "")
                        )
                    elif type_ == "fim":
                        delta_content = decoded_line["choices"][0].get("text", "")
                    if delta_content:
                        yield delta_content
                    if finish_reason == "stop":
                        break
