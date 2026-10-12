"""
cli.py: Interactive Terminal Prompt for Structured Output Agent

Allows entering or pasting invoice text directly in the terminal,
or passing a file path to extract.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Automatically ensure both project root and package directory are in sys.path
_current_file = Path(__file__).resolve()
_pkg_dir = _current_file.parent
_project_root = _pkg_dir.parent

for _p in [str(_project_root), str(_pkg_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv()
load_dotenv(_pkg_dir / ".env")
load_dotenv(_project_root / ".env")

try:
    from structured_output_agent.agent import StructuredOutputAgent
except ModuleNotFoundError:
    from agent import StructuredOutputAgent


def main():
    print("=" * 60)
    print("🧾 Structured Output Agent - Interactive Terminal CLI")
    print("=" * 60)

    agent = StructuredOutputAgent()
    print(f"Active Model: {agent.llm_client.model if hasattr(agent.llm_client, 'model') else 'Mock'}")
    print("\nInstructions: Paste your invoice text below.")
    print("(Press Enter, then Ctrl+Z and Enter on Windows to submit):\n")

    try:
        lines = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nExiting.")
        return

    if not lines.strip():
        print("[ERROR] No input provided.")
        return

    print("\nExtracting and validating against Pydantic schema...")
        invoice = agent.extract_invoice(lines)
        curr = invoice.currency or "$"
        print("\n[SUCCESS] Extracted Invoice:")
        print(f"  Vendor:       {invoice.vendor}")
        print(f"  Date:         {invoice.invoice_date}")
        print(f"  Currency:     {curr}")
        print(f"  Line Items:   {len(invoice.line_items)}")
        for idx, item in enumerate(invoice.line_items, 1):
            print(f"    {idx}. {item.description} (Qty: {item.quantity}) - {curr} {item.unit_price:.2f} each = {curr} {item.total_price:.2f}")
        print(f"  Subtotal:     {curr} {invoice.subtotal:.2f}")
        print(f"  Tax:          {curr} {invoice.tax:.2f}")
        print(f"  Total Amount: {curr} {invoice.total_amount:.2f}")
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")


if __name__ == "__main__":
    main()
