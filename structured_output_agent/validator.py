"""
validator.py: Self-Healing Schema Validation Engine

Core responsibilities:
1. Extract and clean JSON from raw LLM responses (stripping markdown fences).
2. Validate responses against Pydantic models.
3. On validation failure, reflect error messages back into the LLM and retry up to 3 times.
4. Validate tool responses with the identical retry and error logging mechanism.
5. Log all failures to the ValidationFailureLogger.
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_project_root = _pkg_dir.parent
for _p in [str(_project_root), str(_pkg_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from structured_output_agent.llm_client import BaseLLMClient
    from structured_output_agent.logger import ValidationFailureLogger
except ModuleNotFoundError:
    from llm_client import BaseLLMClient
    from logger import ValidationFailureLogger

# Generic type variable bound to Pydantic BaseModel
T = TypeVar("T", bound=BaseModel)


class MaxRetriesExceededError(Exception):
    """Raised when validation fails after all retry attempts are exhausted."""

    def __init__(self, message: str, attempts: int, last_output: str, last_errors: Any):
        super().__init__(message)
        self.attempts = attempts
        self.last_output = last_output
        self.last_errors = last_errors


def extract_json_string(text: str) -> str:
    """
    Extract clean JSON string from raw LLM output.
    
    Handles:
    - Markdown fenced code blocks: ```json ... ``` or ``` ... ```
    - Conversational preambles or postscripts around JSON objects/arrays.
    """
    text = text.strip()

    # 1. Match code blocks ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 2. If no code block, look for outermost curly braces { ... } or brackets [ ... ]
    start_curly = text.find("{")
    end_curly = text.rfind("}")
    if start_curly != -1 and end_curly != -1 and end_curly > start_curly:
        return text[start_curly : end_curly + 1].strip()

    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        return text[start_bracket : end_bracket + 1].strip()

    # Fallback to returning raw text as-is
    return text


def build_extraction_prompt(user_input: str, schema_class: Type[BaseModel]) -> str:
    """
    Generate the initial extraction prompt containing the Pydantic JSON schema.
    """
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
    return (
        f"You are an expert structured data extraction engine.\n\n"
        f"TASK:\n"
        f"Extract information from the following input text into valid JSON matching the schema below.\n\n"
        f"INPUT TEXT:\n"
        f"\"\"\"\n{user_input}\n\"\"\"\n\n"
        f"REQUIRED JSON SCHEMA:\n"
        f"{schema_json}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Return ONLY the valid JSON object.\n"
        f"2. Do NOT include markdown explanations, chit-chat, or notes.\n"
        f"3. Ensure all numbers are raw numeric values (strip any currency symbols like '$' or 'Rs' from numeric amounts).\n"
        f"4. Detect the exact currency used in the text (e.g. 'Rs', 'PKR', 'USD', 'EUR', 'GBP') and place it in the 'currency' field.\n"
        f"5. If the input text is conversational greetings, chatter, or does NOT contain real invoice/purchase data, do NOT invent fake items or dummy $0 amounts.\n"
    )


def build_feedback_prompt(
    original_input: str,
    bad_output: str,
    error_details: Any,
    schema_class: Type[BaseModel],
    attempt: int,
) -> str:
    """
    Construct the self-healing reflection prompt instructing the LLM to fix its errors.
    """
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
    formatted_errors = (
        json.dumps(error_details, indent=2, default=str)
        if not isinstance(error_details, str)
        else error_details
    )

    return (
        f"Your previous output failed validation on Attempt {attempt}.\n\n"
        f"VALIDATION ERRORS:\n"
        f"{formatted_errors}\n\n"
        f"YOUR PREVIOUS (INVALID) OUTPUT WAS:\n"
        f"\"\"\"\n{bad_output}\n\"\"\"\n\n"
        f"ORIGINAL INPUT TEXT:\n"
        f"\"\"\"\n{original_input}\n\"\"\"\n\n"
        f"TARGET JSON SCHEMA:\n"
        f"{schema_json}\n\n"
        f"INSTRUCTIONS:\n"
        f"Please correct the errors above and return ONLY the valid JSON object strictly adhering to the schema."
    )


def validate_with_retry(
    llm_client: BaseLLMClient,
    user_input: str,
    schema_class: Type[T],
    max_retries: int = 3,
    logger: Optional[ValidationFailureLogger] = None,
    system_prompt: Optional[str] = None,
    image_url: Optional[str] = None,
) -> T:
    """
    Call the LLM and validate its output against a Pydantic schema with automatic retries.

    If validation fails:
    1. Logs the failure (input, bad output, error, attempt number) via logger.
    2. Sends the exact Pydantic validation error back to the LLM for self-correction.
    3. Retries up to max_retries times.

    Raises:
        MaxRetriesExceededError: If validation fails after all retry attempts.
    """
    current_prompt = build_extraction_prompt(user_input, schema_class)
    last_raw_output = ""
    last_error_details: Any = None

    for attempt in range(1, max_retries + 1):
        # 1. Ask the LLM
        raw_output = llm_client.generate(
            current_prompt,
            system_prompt=system_prompt,
            image_url=image_url,
        )
        last_raw_output = raw_output

        # 2. Clean and parse JSON
        cleaned_json = extract_json_string(raw_output)

        try:
            parsed_dict = json.loads(cleaned_json)
            validated_obj = schema_class.model_validate(parsed_dict)
            # Success! Return the validated Pydantic model
            return validated_obj

        except (json.JSONDecodeError, ValidationError) as err:
            # Capture error details cleanly without un-serializable exception objects
            if isinstance(err, ValidationError):
                cleaned_errors = []
                for e in err.errors():
                    entry = dict(e)
                    if "ctx" in entry and isinstance(entry["ctx"], dict):
                        entry["ctx"] = {k: str(v) for k, v in entry["ctx"].items()}
                    cleaned_errors.append(entry)
                error_details = cleaned_errors
            else:
                error_details = f"Invalid JSON syntax: {str(err)}"

            last_error_details = error_details

            # 3. Log the failure event
            if logger:
                logger.log_failure(
                    input_data=current_prompt,
                    bad_output=raw_output,
                    error_details=error_details,
                    attempt=attempt,
                    schema_name=schema_class.__name__,
                )

            # If we've exhausted all attempts, stop
            if attempt == max_retries:
                break

            # 4. Build self-healing feedback prompt for the next attempt
            current_prompt = build_feedback_prompt(
                original_input=user_input,
                bad_output=raw_output,
                error_details=error_details,
                schema_class=schema_class,
                attempt=attempt,
            )

    raise MaxRetriesExceededError(
        f"Validation failed after {max_retries} attempts for schema '{schema_class.__name__}'.",
        attempts=max_retries,
        last_output=last_raw_output,
        last_errors=last_error_details,
    )


def validate_tool_response(
    tool_fn: Callable[..., Any],
    tool_kwargs: Dict[str, Any],
    schema_class: Type[T],
    max_retries: int = 3,
    logger: Optional[ValidationFailureLogger] = None,
) -> T:
    """
    Execute a tool function and validate its response against a Pydantic schema with retry support.
    
    Fulfills Requirement 4: 'Also validate tool responses the same way'
    
    If the tool response fails schema validation, records the failure and retries tool execution.
    """
    last_raw_result: Any = None
    last_error_details: Any = None

    for attempt in range(1, max_retries + 1):
        raw_result = tool_fn(**tool_kwargs)
        last_raw_result = raw_result

        try:
            if isinstance(raw_result, str):
                cleaned = extract_json_string(raw_result)
                validated_obj = schema_class.model_validate_json(cleaned)
            elif isinstance(raw_result, dict):
                validated_obj = schema_class.model_validate(raw_result)
            elif isinstance(raw_result, schema_class):
                validated_obj = raw_result
            else:
                validated_obj = schema_class.model_validate(raw_result)

            return validated_obj

        except (ValidationError, json.JSONDecodeError, Exception) as err:
            if isinstance(err, ValidationError):
                cleaned_errors = []
                for e in err.errors():
                    entry = dict(e)
                    if "ctx" in entry and isinstance(entry["ctx"], dict):
                        entry["ctx"] = {k: str(v) for k, v in entry["ctx"].items()}
                    cleaned_errors.append(entry)
                error_details = cleaned_errors
            else:
                error_details = str(err)

            last_error_details = error_details

            if logger:
                logger.log_failure(
                    input_data=json.dumps(tool_kwargs, default=str),
                    bad_output=str(raw_result),
                    error_details=error_details,
                    attempt=attempt,
                    schema_name=schema_class.__name__,
                )

            if attempt == max_retries:
                break

    raise MaxRetriesExceededError(
        f"Tool response validation failed after {max_retries} attempts for '{schema_class.__name__}'.",
        attempts=max_retries,
        last_output=str(last_raw_result),
        last_errors=last_error_details,
    )
