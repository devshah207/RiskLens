"""Clearly attributed recorded examples. Never masquerade as fresh file evaluations."""
from __future__ import annotations

import math
import pandas as pd

PUBLISHED_REPORT = "docs/phase7/README.md (exposed development benchmark; published 2026-11-02)"
EXAMPLES = (
    {
        "id": "ril_sales", "ticker": "RIL", "company": "Reliance Industries",
        "question": "What was Reliance's FY2025 consolidated Value of Sales in crore?",
        "answer": "₹9,60,355 crore (historical known-case PDF reference)",
        "status": "needs_review", "route": "recorded_pdf_development_case",
        "verification_status": "documented_reference_number; programmatic attribution not certified",
        "citations": [{"source_file": "Reliance FY2025 annual report", "ticker": "RIL",
                       "note": "Page not provided by the handover; not asserted here."}],
    },
    {
        "id": "hdfc_ratio", "ticker": "HDFCBANK", "company": "HDFC Bank",
        "question": "What was HDFC Bank's net interest income as percent of working funds?",
        "answer": "3.53% (historical known-case PDF reference)",
        "status": "needs_review", "route": "recorded_pdf_development_case",
        "verification_status": "historical matched reference; requires contextual review",
        "citations": [{"source_file": "HDFC Bank annual report", "ticker": "HDFCBANK"}],
    },
    {
        "id": "infy_expenses", "ticker": "INFY", "company": "Infosys",
        "question": "What were Infosys's FY2025 standalone selling and marketing expenses in crore?",
        "answer": "₹6,282 crore (standalone; historical known-case reference)",
        "status": "needs_review", "route": "recorded_pdf_development_case",
        "verification_status": "historical reference; do not confuse with consolidated ₹7,588 crore",
        "citations": [{"source_file": "Infosys FY2025 annual report", "ticker": "INFY"}],
    },
    {
        "id": "infy_source_supported", "ticker": "INFY", "company": "Infosys",
        "question": "What were Infosys FY2024 standalone net current assets in crore?",
        "answer": "₹43,866 crore — recorded structured PDF-source answer",
        "status": "answered", "route": "recorded_original_pdf_structured_fact",
        "verification_status": "source page checked in historical review; not independently human-certified",
        "citations": [{"source_file": "Infosys annual report", "ticker": "INFY",
                       "scope": "standalone", "period": "FY2024", "unit": "crore",
                       "note": "Source page checked historically; physical page number unavailable here."}],
    },
    {
        "id": "ril_refusal", "ticker": "RIL", "company": "Reliance Industries",
        "question": "What was Reliance FY2024 consolidated income from services?",
        "answer": "The Phase 7 system incorrectly refused this answerable question. The source document contained ₹1,16,476 crore.",
        "status": "abstained", "route": "historical_incorrect_refusal",
        "verification_status": "documented_error; no new model response",
        "citations": [{"source_file": "Reliance annual report", "ticker": "RIL",
                       "pdf_page": 100, "note": "Original PDF physical page, per handover"}],
    },
    {
        "id": "unsupported", "ticker": "ICICIBANK", "company": "ICICI Bank",
        "question": "What were ICICI Bank audited FY2033 net profits?",
        "answer": "FY2033 audited figures are outside the FY2025 historical source collection.",
        "status": "abstained", "route": "recorded_historical_period_guard",
        "verification_status": "unsupported_future_year_refusal",
        "citations": [],
    },
)


def recorded_question(example_id: str) -> dict:
    for ex in EXAMPLES:
        if ex["id"] == example_id:
            return {**ex, "provider": "recorded_demo_only", "recorded_source": PUBLISHED_REPORT}
    raise KeyError("Unknown recorded demo question")


def synthetic_timeline(n: int = 40) -> pd.DataFrame:
    """Fictional session numbers, NOT simulated holdout dates or real predictions."""
    if not 2 <= n <= 100:
        raise ValueError("Synthetic example size must be 2..100")
    return pd.DataFrame({
        "fictional_session": list(range(1, n + 1)),
        "fictional_xgboost_probability": [round(.07 + .025*math.sin(i/4), 4) for i in range(n)],
        "fictional_lstm_probability": [round(.06 + .028*math.cos(i/5), 4) for i in range(n)],
    })
