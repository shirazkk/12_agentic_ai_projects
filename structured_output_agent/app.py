"""
app.py: Streamlit Web UI for Structured Output Agent

Features:
- Prompt box to enter or paste messy invoice text.
- File and image uploader (supports .txt, .png, .jpg, .jpeg, .webp).
- OpenRouter model selector (switch between Nemotron, Gemini Vision, Llama Vision, etc.).
- Formatted metrics cards and table for validated Pydantic data.
- Live inspection of self-healing retry logs.
- Interactive tool validation testing.
"""

import base64
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import streamlit as st

# Automatically ensure both project root and package directory are in sys.path
_current_file = Path(__file__).resolve()
_pkg_dir = _current_file.parent
_project_root = _pkg_dir.parent

for _p in [str(_project_root), str(_pkg_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load environment variables
load_dotenv()
_pkg_env = _pkg_dir / ".env"
if _pkg_env.exists():
    load_dotenv(dotenv_path=_pkg_env)
_root_env = _project_root / ".env"
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env)

try:
    from structured_output_agent.agent import (
        StructuredOutputAgent,
        calculate_tax,
        verify_vendor,
    )
    from structured_output_agent.llm_client import OpenRouterLLMClient, MockLLMClient
    from structured_output_agent.logger import ValidationFailureLogger
    from structured_output_agent.schemas import (
        Invoice,
        TaxCalculationResult,
        VendorVerificationResult,
    )
    from structured_output_agent.validator import MaxRetriesExceededError
except ModuleNotFoundError:
    from agent import (
        StructuredOutputAgent,
        calculate_tax,
        verify_vendor,
    )
    from llm_client import OpenRouterLLMClient, MockLLMClient
    from logger import ValidationFailureLogger
    from schemas import (
        Invoice,
        TaxCalculationResult,
        VendorVerificationResult,
    )
    from validator import MaxRetriesExceededError


# --- Page Configuration ---
st.set_page_config(
    page_title="Structured Output Agent",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Structured Output Agent")
st.caption(
    "Extract strictly validated Pydantic data from messy text and receipt images "
    "using LLMs with an automatic Self-Correction & Reflection loop."
)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Model Configuration")

env_key = os.getenv("OPENROUTER_API_KEY", "")
api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=env_key,
    type="password",
    help="Loaded automatically from .env if present.",
)

default_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
model_options = [
    default_model,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # Active Free Multimodal/Vision Model
    "nvidia/nemotron-3-super-120b-a12b:free",              # Text Model
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-4o-mini",
]
# Remove duplicates while preserving order
seen = set()
unique_models = [m for m in model_options if not (m in seen or seen.add(m))]

selected_model = st.sidebar.selectbox(
    "Active Model",
    options=unique_models,
    index=0,
    help="Select or switch models. For receipt photos, choose 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'.",
)

custom_model = st.sidebar.text_input(
    "Or Enter Custom Model ID",
    value="",
    placeholder="e.g. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
)
final_model = custom_model.strip() if custom_model.strip() else selected_model

max_retries = st.sidebar.slider("Max Retries", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **Vision Tip:**\n"
    "`nvidia/nemotron-3-super` is text-only.\n"
    "When uploading receipt photos, select the active free vision model:\n"
    "👉 **`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`**"
)

# --- Sample Presets ---
sample_1 = """Hey Alex, here's the bill from Staples on 2026-10-12:
- 10x Ballpoint Pens @ $1.50 each ($15.00)
- 4x Spiral Notebooks @ $3.25 each ($13.00)
- 1x Heavy Duty Stapler @ $12.00 ($12.00)
Subtotal was $40.00.
Sales Tax came out to $3.20.
Total bill charged: $43.20.
"""

sample_2 = """Bistro Bella Italian Restaurant
Date: 2026-10-11
Table 4 - Server: Marco
1x Margherita Pizza ($18.50)
2x Sparkling Water ($4.00 each, $8.00)
1x Tiramisu Dessert ($9.50)
Food Subtotal: $36.00
Tax: $2.88
Total Paid: $38.88
"""

sample_pakistan = """Imtiaz Super Market Karachi
Date: 2026-10-12
Invoice # 98452
Items:
- 2x Shan Biryani Masala @ Rs. 140 each (Rs. 280)
- 5x Basmati Rice 1kg @ Rs. 350 each (Rs. 1750)
- 1x Tapal Danedar Tea 450g @ Rs. 850 (Rs. 850)
Subtotal: Rs. 2880
Sales Tax: Rs. 144
Total Amount: Rs. 3024
"""

sample_broken = """Bill from Tech Repair dated 2026-10-10.
Screen Replacement: quantity -2, unit price 50.
Subtotal 100, Tax 10, Total 110.
"""

# --- Input Area ---
tab_text, tab_image, tab_tools, tab_logs = st.tabs([
    "✏️ Text Prompt Box",
    "🖼️ Upload Receipt / Image",
    "🛠️ Test Tool Validation",
    "📜 Failure Audit Logs",
])

with tab_text:
    st.subheader("Messy Invoice Text Input")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    if col_btn1.button("📋 Sample: US Office ($)"):
        st.session_state["prompt_input"] = sample_1
    if col_btn2.button("🍕 Sample: Restaurant ($)"):
        st.session_state["prompt_input"] = sample_2
    if col_btn3.button("🇵🇰 Sample: Pakistan (Rs / PKR)"):
        st.session_state["prompt_input"] = sample_pakistan
    if col_btn4.button("⚠️ Sample: Faulty Input"):
        st.session_state["prompt_input"] = sample_broken

    user_prompt = st.text_area(
        "Enter or paste unstructured invoice text below:",
        value=st.session_state.get("prompt_input", sample_pakistan),
        height=180,
    )

    run_text_btn = st.button("🚀 Extract & Validate Invoice", key="btn_text", type="primary")

with tab_image:
    st.subheader("Upload Receipt File or Photo")
    uploaded_file = st.file_uploader(
        "Choose an image or text file",
        type=["png", "jpg", "jpeg", "webp", "txt"],
    )

    image_data_url: Optional[str] = None
    image_prompt_text = ""

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext in ["png", "jpg", "jpeg", "webp"]:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", width=400)
            bytes_data = uploaded_file.getvalue()
            b64_str = base64.b64encode(bytes_data).decode("utf-8")
            mime_type = f"image/{'jpeg' if file_ext in ['jpg', 'jpeg'] else file_ext}"
            image_data_url = f"data:{mime_type};base64,{b64_str}"
            image_prompt_text = st.text_input(
                "Additional instructions (optional):",
                value="Extract all invoice details from this receipt image.",
            )
            if "omni" not in final_model.lower() and "vision" not in final_model.lower() and "gemma" not in final_model.lower():
                st.warning(
                    f"⚠️ Current model '{final_model}' is text-only. "
                    "For photos of receipts, switch the model in the sidebar to the active free vision model:\n"
                    "👉 **`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`**"
                )
        elif file_ext == "txt":
            file_text = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File Content Preview:", value=file_text, height=150)
            user_prompt = file_text

    run_image_btn = st.button("🚀 Extract from Uploaded File", key="btn_image", type="primary")

# --- Execution Logic ---
should_run = run_text_btn or run_image_btn
active_text = user_prompt if run_text_btn else (image_prompt_text or "Extract invoice details")
active_image = None if run_text_btn else image_data_url

if should_run:
    if not api_key:
        st.error("❌ OpenRouter API Key is required. Please enter it in the sidebar or save it in your .env file.")
    elif not active_text.strip() and not active_image:
        st.error("❌ Please provide either text or upload a receipt image.")
    else:
        with st.spinner(f"Extracting with {final_model} (Validating schema & retrying on errors)..."):
            try:
                # Initialize client and agent
                client = OpenRouterLLMClient(
                    api_key=api_key,
                    model=final_model,
                    temperature=temperature,
                )
                logger = ValidationFailureLogger("validation_failures.jsonl")
                agent = StructuredOutputAgent(
                    llm_client=client,
                    logger=logger,
                    max_retries=max_retries,
                )

                # Extract and validate
                invoice: Invoice = agent.extract_invoice(
                    messy_text=active_text,
                    image_url=active_image,
                )

                st.success("✅ Extraction & Pydantic Validation Successful!")

                # --- Dynamic Currency Detection ---
                curr = invoice.currency or "USD"

                # --- Results Display ---
                col1, col2, col3 = st.columns(3)
                col1.metric("🏢 Vendor", invoice.vendor)
                col2.metric("📅 Invoice Date", invoice.invoice_date)
                col3.metric(f"💰 Total ({curr})", f"{curr} {invoice.total_amount:.2f}")

                col4, col5, col6 = st.columns(3)
                col4.metric(f"Subtotal ({curr})", f"{curr} {invoice.subtotal:.2f}")
                col5.metric(f"Tax ({curr})", f"{curr} {invoice.tax:.2f}")
                col6.metric("Line Items Count", len(invoice.line_items))

                st.markdown("### 📋 Line Items")
                items_data = [
                    {
                        "Description": item.description,
                        "Quantity": item.quantity,
                        f"Unit Price ({curr})": f"{curr} {item.unit_price:.2f}",
                        f"Total Price ({curr})": f"{curr} {item.total_price:.2f}",
                    }
                    for item in invoice.line_items
                ]
                st.table(items_data)

                with st.expander("🔍 View Raw Validated JSON (Pydantic Model)"):
                    st.json(invoice.model_dump())

            except MaxRetriesExceededError as e:
                st.error(f"❌ Max Retries Exceeded ({e.attempts} attempts failed).")
                st.markdown(f"**Last Error Details:**\n`{e.last_errors}`")
                st.markdown(f"**Last Model Output:**\n```\n{e.last_output}\n```")
                st.info("💡 Check the 'Failure Audit Logs' tab to inspect every failed attempt.")
            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")

# --- Tools Tab ---
with tab_tools:
    st.subheader("Test Validated Tool Execution (Requirement 4)")
    st.write(
        "Demonstrates executing internal agent tools where the tool's output is "
        "enforced and guaranteed by a Pydantic schema."
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Tool 1: Tax Calculator")
        tool_subtotal = st.number_input("Pre-tax Subtotal ($)", min_value=0.0, value=100.0, step=10.0)
        tool_rate = st.slider("Tax Rate", min_value=0.0, max_value=0.30, value=0.08, step=0.01)

        if st.button("Run Tax Tool"):
            agent = StructuredOutputAgent()
            res = agent.execute_tool(
                tool_fn=calculate_tax,
                tool_kwargs={"subtotal": tool_subtotal, "tax_rate": tool_rate},
                schema_class=TaxCalculationResult,
            )
            st.success(f"Validated Tool Output: **${res.calculated_tax:.2f}** tax on ${res.subtotal:.2f}")
            st.json(res.model_dump())

    with col_t2:
        st.markdown("#### Tool 2: Vendor Verifier")
        vendor_name_input = st.text_input("Vendor Name to Verify", value="Acme Supplies")

        if st.button("Run Vendor Tool"):
            agent = StructuredOutputAgent()
            v_res = agent.execute_tool(
                tool_fn=verify_vendor,
                tool_kwargs={"vendor_name": vendor_name_input},
                schema_class=VendorVerificationResult,
            )
            st.success(f"Vendor '{v_res.vendor_name}' Verified: **{v_res.is_verified}** (Risk: {v_res.risk_score})")
            st.json(v_res.model_dump())

# --- Audit Logs Tab ---
with tab_logs:
    st.subheader("📜 Validation Failure Audit Trail (Requirement 5)")
    st.write("Displays entries recorded in `validation_failures.jsonl` whenever schema validation fails.")

    logger = ValidationFailureLogger("validation_failures.jsonl")
    records = logger.read_records()

    col_log1, col_log2 = st.columns([4, 1])
    col_log1.write(f"Total Failures Recorded: **{len(records)}**")
    if col_log2.button("🗑️ Clear Log File"):
        logger.clear()
        st.rerun()

    if records:
        for idx, rec in enumerate(reversed(records)):
            with st.expander(f"Failure #{len(records) - idx}: Attempt {rec.get('attempt')} at {rec.get('timestamp')}"):
                st.markdown(f"**Target Schema:** `{rec.get('schema_name')}`")
                st.markdown("**Error Details:**")
                st.json(rec.get("error_details"))
                st.markdown("**Bad Raw Output:**")
                st.code(rec.get("bad_output") or "None")
                st.markdown("**Input Prompt / Context:**")
                st.text(rec.get("input_data", "")[:400] + ("..." if len(rec.get("input_data", "")) > 400 else ""))
    else:
        st.info("No validation failures recorded yet. The agent's outputs have all passed validation!")
