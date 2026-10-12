"""
logger.py: Validation Failure Logger for Structured Output Agent

Logs every validation failure event (input, bad output, error details, attempt number)
to a structured JSON Lines (.jsonl) file for debugging, auditing, and observability.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "validation_failures.jsonl"


class ValidationFailureLogger:
    """Records validation failures to a JSON Lines log file."""

    def __init__(self, log_file: Optional[Union[str, Path]] = None):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to the .jsonl file where failures will be recorded.
        """
        if log_file is None or log_file == "validation_failures.jsonl":
            self.log_file = DEFAULT_LOG_FILE
        else:
            self.log_file = Path(log_file)

        # Ensure the directory exists if a custom path is provided
        if self.log_file.parent and not self.log_file.parent.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(
        self,
        input_data: str,
        bad_output: str,
        error_details: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        attempt: int,
        schema_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Append a single validation failure entry to the log file.

        Args:
            input_data: The input text/prompt sent to the LLM or tool.
            bad_output: The raw output that failed schema validation.
            error_details: The validation error details (e.g. from Pydantic or JSON parser).
            attempt: The attempt number (1, 2, 3...).
            schema_name: The name of the target Pydantic schema (optional).

        Returns:
            The structured dictionary record that was logged.
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
            "schema_name": schema_name or "Unknown",
            "input_data": input_data,
            "bad_output": bad_output,
            "error_details": error_details,
        }

        # Open in append mode with UTF-8 encoding
        with open(self.log_file, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        return record

    def read_records(self) -> List[Dict[str, Any]]:
        """
        Read all logged failure records from the log file.
        Useful for inspecting failure logs or asserting them in tests.
        """
        if not self.log_file.exists():
            return []

        records = []
        with open(self.log_file, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def clear(self) -> None:
        """Clear the log file (convenient for test setup/teardown)."""
        if self.log_file.exists():
            self.log_file.unlink()
