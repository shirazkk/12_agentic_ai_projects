"""
agent.py: High-Level Structured Output Agent

Orchestrates:
1. Extraction of structured Invoice data from unstructured messy text.
2. Tool execution with output schema validation.
3. Integration with OpenRouter/Mock LLM clients and ValidationFailureLogger.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

# Automatically ensure both project root and package directory are in sys.path
_current_file = Path(__file__).resolve()
_pkg_dir = _current_file.parent
_project_root = _pkg_dir.parent

for _p in [str(_project_root), str(_pkg_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from structured_output_agent.llm_client import BaseLLMClient, create_llm_client
    from structured_output_agent.logger import ValidationFailureLogger
    from structured_output_agent.schemas import (
        Invoice,
        TaxCalculationResult,
        VendorVerificationResult,
    )
    from structured_output_agent.validator import (
        MaxRetriesExceededError,
        validate_tool_response,
        validate_with_retry,
    )
except ModuleNotFoundError:
    from llm_client import BaseLLMClient, create_llm_client
    from logger import ValidationFailureLogger
    from schemas import (
        Invoice,
        TaxCalculationResult,
        VendorVerificationResult,
    )
    from validator import (
        MaxRetriesExceededError,
        validate_tool_response,
        validate_with_retry,
    )

T = TypeVar("T", bound=BaseModel)


# --- Built-in Sample Tools for Demonstration and Testing ---

def calculate_tax(subtotal: float, tax_rate: float, currency: str = "USD") -> Dict[str, Any]:
    """
    Sample tool that computes invoice tax.
    Outputs data matching the TaxCalculationResult schema.
    """
    calculated_tax = round(subtotal * tax_rate, 2)
    return {
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "calculated_tax": calculated_tax,
        "currency": currency,
    }


def verify_vendor(vendor_name: str) -> Dict[str, Any]:
    """
    Sample tool that checks vendor risk registry.
    Outputs data matching the VendorVerificationResult schema.
    """
    known_vendors = {"Acme Supplies", "Staples", "Office Depot", "TechCorp Inc"}
    is_verified = vendor_name.strip() in known_vendors
    return {
        "vendor_name": vendor_name,
        "is_verified": is_verified,
        "tax_id": "TAX-998877" if is_verified else None,
        "risk_score": 0.05 if is_verified else 0.85,
    }


class StructuredOutputAgent:
    """
    Agent capable of extracting validated Pydantic data from messy text
    and executing tools with schema verification and self-healing retries.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        logger: Optional[ValidationFailureLogger] = None,
        max_retries: int = 3,
        log_file: str = "validation_failures.jsonl",
    ):
        """
        Initialize the agent.

        Args:
            llm_client: LLM client (defaults to OpenRouterLLMClient via .env).
            logger: Failure logger (defaults to ValidationFailureLogger).
            max_retries: Maximum validation retry attempts before raising an error.
            log_file: File path for failure logs if logger is not explicitly passed.
        """
        self.logger = logger or ValidationFailureLogger(log_file=log_file)
        self.max_retries = max_retries
        self.llm_client = llm_client or create_llm_client(use_mock=False)

    def extract_invoice(self, messy_text: str = "", image_url: Optional[str] = None) -> Invoice:
        """
        Extract a validated Invoice from messy, unstructured text or an image.

        Args:
            messy_text: Raw invoice receipt, email text, or notes.
            image_url: Optional base64 or public image data URL for vision models.

        Returns:
            Validated Invoice Pydantic instance.

        Raises:
            MaxRetriesExceededError: If validation fails after max_retries attempts.
        """
        return self.extract_structured_data(
            messy_text=messy_text,
            schema_class=Invoice,
            image_url=image_url,
        )

    def extract_structured_data(
        self,
        messy_text: str,
        schema_class: Type[T],
        image_url: Optional[str] = None,
    ) -> T:
        """
        Generic method to extract any Pydantic schema from messy text or image.

        Args:
            messy_text: Unstructured input text.
            schema_class: The Pydantic model class to validate against.
            image_url: Optional base64 or public image data URL.

        Returns:
            Instance of schema_class.
        """
        return validate_with_retry(
            llm_client=self.llm_client,
            user_input=messy_text,
            schema_class=schema_class,
            max_retries=self.max_retries,
            logger=self.logger,
            image_url=image_url,
        )

    def execute_tool(
        self,
        tool_fn: Callable[..., Any],
        tool_kwargs: Dict[str, Any],
        schema_class: Type[T],
    ) -> T:
        """
        Execute a tool function and validate its output against schema_class.

        Args:
            tool_fn: The callable tool function.
            tool_kwargs: Arguments for the tool.
            schema_class: Expected output schema.

        Returns:
            Validated instance of schema_class.
        """
        return validate_tool_response(
            tool_fn=tool_fn,
            tool_kwargs=tool_kwargs,
            schema_class=schema_class,
            max_retries=self.max_retries,
            logger=self.logger,
        )


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    # Ensure .env is loaded regardless of execution directory
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    # Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Structured Output Agent Demo ===")

    sample_messy_invoice = """
    Hey team, received bill from Acme Supplies dated 2026-10-10.
    Items bought:
    - 5x Wireless Keyboard at $20.00 each (total $100.00)
    - 2x Monitor Stands at $15.50 each (total $31.00)
    Subtotal came out to $131.00.
    Taxes added were $10.48.
    Total payable: $141.48.
    """

    # Check if OPENROUTER_API_KEY is configured
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or "your_openrouter_api_key_here" in api_key:
        print("\n[NOTE] No live OPENROUTER_API_KEY detected in .env.")
        print("Running demonstration with simulated MockLLMClient...\n")
        from structured_output_agent.llm_client import MockLLMClient

        mock_json_response = """
        {
            "vendor": "Acme Supplies",
            "invoice_date": "2026-10-10",
            "line_items": [
                {"description": "Wireless Keyboard", "quantity": 5, "unit_price": 20.0, "total_price": 100.0},
                {"description": "Monitor Stand", "quantity": 2, "unit_price": 15.5, "total_price": 31.0}
            ],
            "subtotal": 131.0,
            "tax": 10.48,
            "total_amount": 141.48
        }
        """
        client = MockLLMClient(responses=[mock_json_response])
        agent = StructuredOutputAgent(llm_client=client)
    else:
        print(f"\nConnecting to OpenRouter (Model: {os.getenv('OPENROUTER_MODEL', 'default')})...\n")
        agent = StructuredOutputAgent()

    print("Extracting invoice...")
    try:
        invoice = agent.extract_invoice(sample_messy_invoice)
        curr = invoice.currency or "$"
        print("\n[SUCCESS] Successfully extracted and validated Invoice!")
        print(f"Vendor: {invoice.vendor}")
        print(f"Date: {invoice.invoice_date}")
        print(f"Currency: {curr}")
        print(f"Line Items: {len(invoice.line_items)}")
        for item in invoice.line_items:
            print(f"  - {item.description}: {item.quantity} @ {curr} {item.unit_price} = {curr} {item.total_price}")
        print(f"Subtotal: {curr} {invoice.subtotal:.2f}")
        print(f"Tax: {curr} {invoice.tax:.2f}")
        print(f"Total: {curr} {invoice.total_amount:.2f}")

        # Demonstrate tool validation
        print("\nExecuting and validating tax calculation tool...")
        tax_res = agent.execute_tool(
            tool_fn=calculate_tax,
            tool_kwargs={"subtotal": invoice.subtotal, "tax_rate": 0.08},
            schema_class=TaxCalculationResult,
        )
        print(f"Validated Tax Result: ${tax_res.calculated_tax} (Rate: {tax_res.tax_rate * 100}%)")

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
