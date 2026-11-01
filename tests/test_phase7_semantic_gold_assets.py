"""Research QA labels must be explicit and never disguised as human-certified."""
from __future__ import annotations

import json
from pathlib import Path


def test_retroactive_reference_corrections_are_explicitly_labelled():
    p = Path(__file__).resolve().parents[1] / "docs/phase7_retrospective_reference_addenda.json"
    data = json.loads(p.read_text())
    assert data["status"].startswith("RETROSPECTIVE")
    assert len(data["addenda"]) == 4
    assert {row["question_id"] for row in data["addenda"]} == {
        "HDFC03", "INFY01", "ICICI02", "ICICI03"
    }
    assert all(len(row["source_sha256"]) == 64 for row in data["addenda"])


def test_new_challenge_labels_are_provisional_and_predeclared():
    p = Path(__file__).resolve().parents[1] / "docs/phase7_new_challenge_set_v1.json"
    data = json.loads(p.read_text())
    cases = data["questions"]
    assert len(cases) == 15
    assert sum(q["answerable"] for q in cases) == 13
    assert len({q["id"] for q in cases}) == len(cases)
    assert "NOT_HUMAN_CERTIFIED" in data["status"]
    assert all("source_sha256" in q for q in cases if q["answerable"])
