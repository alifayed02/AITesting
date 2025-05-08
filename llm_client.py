import os
import re
import json
from openai import OpenAI
from typing import Dict, Any, Optional, List
from pprint import pprint

# Configure API keys
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Initialize clients
gpt_client        = OpenAI(api_key=OPENAI_API_KEY)
deepseek_client   = OpenAI(api_key=DEEPSEEK_API_KEY,   base_url="https://api.deepseek.com")
perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")


def force_json(resp: str) -> Dict[str, Any]:
    """
    Extracts the first JSON object from the response text and parses it.
    Raises ValueError if no valid JSON is found.
    """
    match = re.search(r"\{.*\}", resp, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {resp!r}")
    return json.loads(match.group(0))


def _build_messages(
    question:     str,
    numeric_only: bool = False,
    multi_part:   bool = False,
    image_url:    Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Prepends system instructions to force output shape:
      - numeric_only: model returns just the number.
      - multi_part:   model returns a JSON object.
      - default:      one clear sentence.
    If image_url is provided, appends it as a second user message.
    """
    if numeric_only:
        instr = "You are a concise assistant. Answer with exactly the numeric result—no units, no explanation."
    elif multi_part:
        instr = (
            "You are a JSON-only assistant. Your entire output must be a single JSON object, "
            "parseable by json.loads(). Use top-level string keys \"a\", \"b\", \"c\", … exactly, "
            "with string values only. No markdown or extra commentary."
        )
    else:
        instr = "Answer in one clear sentence only—no bullet points, no extra elaboration."

    messages = [
        {"role": "system", "content": instr},
        {"role": "user",   "content": question}
    ]

    if image_url:
        # If it's a local path, you might need to upload or serve it—here we assume a URL is fine.
        messages.append({
            "role": "user",
            "content": f"Here is the supporting diagram: {image_url}"
        })

    return messages


def get_gpt_response(
    question:     str,
    numeric_only: bool = False,
    multi_part:   bool = False,
    image_url:    Optional[str] = None
) -> str:
    completion = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=_build_messages(question, numeric_only, multi_part, image_url),
        temperature=0.0,
        max_tokens=2000
    )
    return completion.choices[0].message.content.strip()


def get_deepseek_response(
    question:     str,
    numeric_only: bool = False,
    multi_part:   bool = False,
    image_url:    Optional[str] = None
) -> str:
    completion = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=_build_messages(question, numeric_only, multi_part, image_url),
        temperature=0.0,
        max_tokens=2000
    )
    return completion.choices[0].message.content.strip()


def get_perplexity_response(
    question:     str,
    numeric_only: bool = False,
    multi_part:   bool = False,
    image_url:    Optional[str] = None
) -> str:
    completion = perplexity_client.chat.completions.create(
        model="sonar",
        messages=_build_messages(question, numeric_only, multi_part, image_url),
        temperature=0.0,
        max_tokens=2000
    )
    return completion.choices[0].message.content.strip()


def get_response(
    question:     str,
    model:        str,
    numeric_only: bool = False,
    multi_part:   bool = False,
    image_url:    Optional[str] = None
) -> Any:
    """
    Dispatch to the right model, forwarding all flags.
    Returns:
      - str for numeric_only or one-sentence text
      - dict for multi_part (parsed JSON via force_json)
    """
    if model == "gpt":
        raw = get_gpt_response(question, numeric_only, multi_part, image_url)
    elif model == "deepseek":
        raw = get_deepseek_response(question, numeric_only, multi_part, image_url)
    elif model == "perplexity":
        raw = get_perplexity_response(question, numeric_only, multi_part, image_url)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    return force_json(raw) if multi_part else raw


def get_all_responses(
    question:      str,
    models:       Optional[List[str]] = None,
    numeric_only:  bool = False,
    multi_part:    bool = False,
    image_url:     Optional[str] = None
) -> Dict[str, Any]:
    """
    Ask multiple models at once, with consistent output formatting,
    optionally including an image.
    """
    if models is None:
        models = ["gpt", "deepseek", "perplexity"]

    responses: Dict[str, Any] = {}
    for m in models:
        try:
            responses[m] = get_response(question, m, numeric_only, multi_part, image_url)
        except Exception as e:
            responses[m] = f"ERROR: {e}"
    return responses
