import json
import re
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

def try_parse_ollama_output(text: str, output_format: type[T]) -> T:
    """
    Extract and repair JSON from Ollama output, even if wrapped in Markdown or with extra text.
    Returns a validated Pydantic model instance.
    """
    # Try to extract JSON from code block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: extract JSON between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            raise ValueError("No JSON block found in Ollama output.")

    # Optionally: repair common JSON issues here (e.g., control characters, trailing commas)
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        # Try to repair or raise
        # Remove trailing commas
        json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            parsed = json.loads(json_str_fixed)
        except Exception:
            raise ValueError(f"Failed to parse JSON: {e}\nRaw: {json_str}")

    # Some models return a list with one dict
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]

    return output_format.model_validate(parsed)