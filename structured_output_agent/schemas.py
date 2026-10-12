"""
schemas.py: Data Contracts for Structured Output Agent

Defines Pydantic schemas for:
1. Invoice extraction (LineItem, Invoice)
2. Tool response validation (TaxCalculationResult, VendorVerificationResult)
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class LineItem(BaseModel):
    """Represents a single purchased item or service on an invoice."""
    description: str = Field(..., description="Name or description of the item/service")
    quantity: float = Field(..., gt=0, description="Quantity purchased, must be greater than 0")
    unit_price: float = Field(..., ge=0, description="Unit price per item, must be non-negative")
    total_price: float = Field(..., ge=0, description="Total price for this line item")

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Line item description cannot be empty or blank")
        return value.strip()


class Invoice(BaseModel):
    """Represents a complete validated invoice extracted from messy text."""
    is_invoice: bool = Field(
        default=True,
        description="Set to True ONLY if the text describes a real purchase, bill, or invoice. Set to False for greetings, chatter, questions, or non-invoice text.",
    )
    vendor: str = Field(default="", description="Name of the seller, vendor, or store")
    invoice_date: str = Field(default="", description="Invoice date in YYYY-MM-DD or readable date format")
    line_items: List[LineItem] = Field(default_factory=list, description="List of line items on the invoice")
    currency: str = Field(
        default="USD",
        description="Currency symbol or code as stated in the invoice (e.g. 'Rs', 'PKR', 'USD', '$', 'EUR', 'GBP').",
    )
    subtotal: float = Field(default=0.0, description="Sum of line item totals before tax")
    tax: float = Field(default=0.0, description="Tax amount applied to invoice")
    total_amount: float = Field(default=0.0, description="Final payable amount")

    @model_validator(mode="after")
    def validate_is_invoice_and_fields(self):
        if not self.is_invoice:
            raise ValueError("The input text does not contain invoice or receipt data (e.g. greeting or irrelevant text).")
        if not self.vendor or not self.vendor.strip():
            raise ValueError("Vendor name cannot be empty on a valid invoice.")
        if not self.invoice_date or not self.invoice_date.strip() or len(self.invoice_date.strip()) < 4:
            raise ValueError("Invoice date is missing or invalid on a valid invoice.")
        if len(self.line_items) == 0:
            raise ValueError("Invoice must contain at least one line item.")
        if self.total_amount <= 0:
            raise ValueError("Total amount must be greater than 0 on a valid invoice.")
        return self


# --- Tool Response Schemas ---
# Used to validate structured outputs produced by tools called during agent workflows.

class TaxCalculationResult(BaseModel):
    """Schema for validating the output of a tax calculation tool."""
    subtotal: float = Field(..., ge=0, description="Pre-tax amount")
    tax_rate: float = Field(..., ge=0, le=1.0, description="Tax rate decimal (e.g. 0.08 for 8%)")
    calculated_tax: float = Field(..., ge=0, description="Computed tax amount")
    currency: str = Field(default="USD", description="Currency code (e.g. USD, EUR, PKR)")


class VendorVerificationResult(BaseModel):
    """Schema for validating vendor verification tool results."""
    vendor_name: str = Field(..., description="Name of the vendor verified")
    is_verified: bool = Field(..., description="True if vendor exists in registry, else False")
    tax_id: Optional[str] = Field(default=None, description="Official Tax ID or VAT number if available")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk assessment score between 0.0 and 1.0")
