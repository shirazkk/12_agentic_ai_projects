"""
test_agent.py: Pytest Suite for Structured Output Agent

Tests required scenarios:
1. Good input: Valid response on attempt 1.
2. Bad input / retry exhausted: Invalid response persists for 3 attempts, raises MaxRetriesExceededError.
3. Retry success: Invalid on attempt 1, self-corrected on attempt 2 after reflection.
4. Tool response validation: Validating tool outputs against schema, including retry behavior.
5. Failure logging: Ensuring validation failures are accurately recorded to JSONL audit file.
6. Markdown unwrapping: Handling code block fences around JSON outputs.
"""

import json
from pathlib import Path
import pytest

from structured_output_agent.agent import (
    StructuredOutputAgent,
    calculate_tax,
    verify_vendor,
)
from structured_output_agent.llm_client import MockLLMClient
from structured_output_agent.logger import ValidationFailureLogger
from structured_output_agent.schemas import (
    Invoice,
    TaxCalculationResult,
    VendorVerificationResult,
)
from structured_output_agent.validator import MaxRetriesExceededError


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary log file path for each test."""
    return tmp_path / "test_validation_failures.jsonl"


@pytest.fixture
def failure_logger(temp_log_file: Path) -> ValidationFailureLogger:
    """Fixture providing a fresh ValidationFailureLogger instance."""
    return ValidationFailureLogger(log_file=temp_log_file)


# Sample valid invoice JSON
VALID_INVOICE_JSON = json.dumps({
    "vendor": "Acme Supplies",
    "invoice_date": "2026-10-10",
    "line_items": [
        {
            "description": "Wireless Keyboard",
            "quantity": 2.0,
            "unit_price": 25.0,
            "total_price": 50.0,
        }
    ],
    "subtotal": 50.0,
    "tax": 4.0,
    "total_amount": 54.0,
})

# Sample invalid invoice JSON (violates quantity > 0 constraint)
INVALID_QUANTITY_JSON = json.dumps({
    "vendor": "Acme Supplies",
    "invoice_date": "2026-10-10",
    "line_items": [
        {
            "description": "Wireless Keyboard",
            "quantity": -5.0,  # Invalid: must be gt=0
            "unit_price": 25.0,
            "total_price": 50.0,
        }
    ],
    "subtotal": 50.0,
    "tax": 4.0,
    "total_amount": 54.0,
})

# Sample completely unparseable junk text
CORRUPTED_TEXT = "I am sorry, but I cannot extract the invoice from this text."


def test_good_input(failure_logger: ValidationFailureLogger):
    """Test 1: LLM returns valid schema on attempt 1."""
    mock_client = MockLLMClient(responses=[VALID_INVOICE_JSON])
    agent = StructuredOutputAgent(llm_client=mock_client, logger=failure_logger)

    invoice = agent.extract_invoice("Receipt from Acme Supplies on 2026-10-10 for 2 keyboards")

    # Verify extracted fields
    assert isinstance(invoice, Invoice)
    assert invoice.vendor == "Acme Supplies"
    assert invoice.invoice_date == "2026-10-10"
    assert len(invoice.line_items) == 1
    assert invoice.line_items[0].quantity == 2.0
    assert invoice.total_amount == 54.0

    # Verify LLM was called only once
    assert len(mock_client.call_history) == 1

    # Verify no failures were logged
    assert len(failure_logger.read_records()) == 0


def test_retry_success(failure_logger: ValidationFailureLogger):
    """Test 2: LLM fails on attempt 1, receives error feedback, and fixes it on attempt 2."""
    # Attempt 1: invalid quantity (-5), Attempt 2: valid invoice
    mock_client = MockLLMClient(responses=[INVALID_QUANTITY_JSON, VALID_INVOICE_JSON])
    agent = StructuredOutputAgent(llm_client=mock_client, logger=failure_logger, max_retries=3)

    invoice = agent.extract_invoice("Messy invoice text with 2 keyboards")

    # Assert successful validation on attempt 2
    assert isinstance(invoice, Invoice)
    assert invoice.vendor == "Acme Supplies"
    assert invoice.line_items[0].quantity == 2.0

    # Assert LLM was called twice
    assert len(mock_client.call_history) == 2

    # Assert Attempt 2 prompt contained validation feedback
    attempt_2_prompt = mock_client.call_history[1]["prompt"]
    assert "VALIDATION ERRORS" in attempt_2_prompt
    assert "Attempt 1" in attempt_2_prompt
    assert "greater than 0" in attempt_2_prompt

    # Assert exactly 1 failure was logged for attempt 1
    logged_records = failure_logger.read_records()
    assert len(logged_records) == 1
    assert logged_records[0]["attempt"] == 1
    assert logged_records[0]["schema_name"] == "Invoice"


def test_bad_input_retry_exhausted(failure_logger: ValidationFailureLogger):
    """Test 3: LLM fails 3 times consecutively, exhausts retries, and raises MaxRetriesExceededError."""
    # 3 consecutive corrupted responses
    mock_client = MockLLMClient(responses=[CORRUPTED_TEXT, CORRUPTED_TEXT, CORRUPTED_TEXT])
    agent = StructuredOutputAgent(llm_client=mock_client, logger=failure_logger, max_retries=3)

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        agent.extract_invoice("Unparseable garbage input")

    # Verify exception metadata
    assert exc_info.value.attempts == 3
    assert exc_info.value.last_output == CORRUPTED_TEXT
    assert "after 3 attempts" in str(exc_info.value)

    # Assert all 3 calls were made
    assert len(mock_client.call_history) == 3

    # Assert 3 failures were logged in audit file
    records = failure_logger.read_records()
    assert len(records) == 3
    assert [r["attempt"] for r in records] == [1, 2, 3]


def test_markdown_fenced_json_extraction(failure_logger: ValidationFailureLogger):
    """Test 4: LLM wraps JSON in markdown fences and conversational preambles."""
    fenced_response = f"""
    Certainly! Here is the extracted invoice according to your schema:
    ```json
    {VALID_INVOICE_JSON}
    ```
    Hope this helps!
    """
    mock_client = MockLLMClient(responses=[fenced_response])
    agent = StructuredOutputAgent(llm_client=mock_client, logger=failure_logger)

    invoice = agent.extract_invoice("Raw invoice text")

    assert invoice.vendor == "Acme Supplies"
    assert invoice.total_amount == 54.0
    assert len(failure_logger.read_records()) == 0


def test_tool_response_validation_success(failure_logger: ValidationFailureLogger):
    """Test 5: Tool output schema validation succeeds."""
    agent = StructuredOutputAgent(logger=failure_logger)

    tax_result = agent.execute_tool(
        tool_fn=calculate_tax,
        tool_kwargs={"subtotal": 100.0, "tax_rate": 0.08},
        schema_class=TaxCalculationResult,
    )

    assert isinstance(tax_result, TaxCalculationResult)
    assert tax_result.subtotal == 100.0
    assert tax_result.tax_rate == 0.08
    assert tax_result.calculated_tax == 8.0
    assert tax_result.currency == "USD"


def test_tool_response_validation_failure_and_retry(failure_logger: ValidationFailureLogger):
    """Test 6: Tool returns bad schema on attempt 1, succeeds on attempt 2."""
    call_count = {"count": 0}

    def flaky_tool(subtotal: float) -> dict:
        call_count["count"] += 1
        if call_count["count"] == 1:
            # Bad output: missing required 'tax_rate'
            return {"subtotal": subtotal, "calculated_tax": 8.0}
        # Good output on attempt 2
        return {"subtotal": subtotal, "tax_rate": 0.08, "calculated_tax": 8.0, "currency": "USD"}

    agent = StructuredOutputAgent(logger=failure_logger, max_retries=3)

    result = agent.execute_tool(
        tool_fn=flaky_tool,
        tool_kwargs={"subtotal": 100.0},
        schema_class=TaxCalculationResult,
    )

    assert result.calculated_tax == 8.0
    assert call_count["count"] == 2

    # Verify 1 failure was logged for attempt 1
    records = failure_logger.read_records()
    assert len(records) == 1
    assert records[0]["attempt"] == 1
    assert records[0]["schema_name"] == "TaxCalculationResult"


def test_tool_response_validation_exhausted(failure_logger: ValidationFailureLogger):
    """Test 7: Tool consistently returns invalid schema and exhausts retries."""
    def broken_tool() -> dict:
        return {"unexpected_key": "invalid_data"}

    agent = StructuredOutputAgent(logger=failure_logger, max_retries=3)

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        agent.execute_tool(
            tool_fn=broken_tool,
            tool_kwargs={},
            schema_class=TaxCalculationResult,
        )

    assert exc_info.value.attempts == 3
    assert len(failure_logger.read_records()) == 3


def test_pakistani_currency_extraction(failure_logger: ValidationFailureLogger):
    """Test 8: LLM correctly extracts Pakistani Rupee (Rs / PKR) currency."""
    pkr_invoice_json = json.dumps({
        "vendor": "Imtiaz Super Market",
        "invoice_date": "2026-10-12",
        "currency": "Rs",
        "line_items": [
            {
                "description": "Shan Biryani Masala",
                "quantity": 2.0,
                "unit_price": 140.0,
                "total_price": 280.0,
            }
        ],
        "subtotal": 280.0,
        "tax": 14.0,
        "total_amount": 294.0,
    })
    mock_client = MockLLMClient(responses=[pkr_invoice_json])
    agent = StructuredOutputAgent(llm_client=mock_client, logger=failure_logger)

    invoice = agent.extract_invoice("Bill from Imtiaz Super Market for Rs 294")

    assert invoice.vendor == "Imtiaz Super Market"
    assert invoice.currency == "Rs"
    assert invoice.total_amount == 294.0
    assert invoice.line_items[0].unit_price == 140.0

