# Phase 7 — XBRL-first financial-report RAG (research prototype)

**Status: implemented and evaluated on an exposed, 34-question development benchmark. Not a production-grade or blind-validated financial advisor.**

This report is written in plain language so recruiters and other developers can understand exactly what works and what is still limited.

## What the system does

RiskLens combines two different types of financial information:

1. **XBRL financial filings** — XML files from the National Stock Exchange (NSE), in which reported financial facts have tags, dates, units and accounting context. The program looks for an **exact** financial item, company, year and standalone/consolidated reporting type. If a matching value exists, it can answer using that value without needing Groq.
2. **PDF annual reports** — if there is no exact XBRL match, RiskLens searches the company's uploaded annual-report PDF. It uses lexical search (BM25) and semantic similarity (MiniLM embeddings) to find relevant passages and tables, and Groq to help answer from the retrieved passages. The system can return an answer, return a tentative number marked `needs_review`, or refuse when its available evidence is insufficient.

The current code does not treat one similar-sounding financial item as another. For example, *Revenue from Operations* must not automatically answer a question about *Value of Sales*.

### Data and coverage

- Four FY2025 companies: Reliance Industries, HDFC Bank, Infosys and ICICI Bank.
- Five PDF files in the research corpus (separate ICICI reports).
- Eight XBRL XML files from the NSE: standalone and consolidated FY2025 files for each company.
- The original 34-question benchmark contains 20 numerical questions, eight explanations and six intentionally unsupported future-year questions.
- The original benchmark was repeatedly used in development. **It is not an independent blind evaluation**.

**The PDFs, original XBRL files, local SQLite index, Groq account credentials and API-call logs are NOT in this GitHub branch.** To rerun the system, obtain documents from their original sources and configure your own local environment.

## Measured outcomes through 2 November 2026

| Outcome | Questions | What that means |
|---|---:|---|
| Supported numerical or written answers | **18** | Five from the earlier XBRL/stored-fact routes and 13 from the later live PDF/Groq route. An assistant compared these with the source PDFs. |
| Correct refusals | **6** | Questions about unsupported future reporting periods were appropriately refused. |
| Incorrect refusals | **4** | The PDF contained an answer, but the system refused because its source-checking rules were too restrictive. |
| Not yet run end-to-end | **6** | Six numerical questions remained incomplete when the final evaluation was stopped by the local token budget. |
| **Total** | **34** | Every question accounted for. |

**This is not 100% accuracy.** Of 17 completed live questions, 13 returned source-supported answers and four incorrectly refused answerable questions. The 13 answers were judged supported by the assistant, **not independently human-certified**. The published benchmark is not blind, and six questions are untested.

**Four direct XBRL answers:** Three matched their PDF reference values exactly. ICICI Bank's reserves differed by ₹3,000 after unit conversion; the XBRL file's precision information supports treating that difference as rounding. The fifth earlier answer, Infosys's FY2024 standalone net current assets, was read from an existing structured PDF fact and checked against the annual report.

**Four known incorrect refusals:**

- Reliance FY2024 consolidated income from services: ₹1,16,476 crore appears in the report, but the system abstained.
- HDFC FY2024 standalone business per employee: ₹21.49 crore appears in the report, but the system abstained.
- Infosys RISC360/iGRC explanation exists in the report, but the system abstained.
- ICICI five broad risk types exist in the report, but the system abstained.

See [the full 17-case evidence audit](PHASE7_LIVE_AUDIT_17.csv) for individual question IDs, answers, source pages and review labels.

## What we tried and why this is the chosen version

The research tried a PDF-number ledger and independent checker, more detailed page geometry, font-based parent headings, and table-structure experiments. Several tests found that these systems could copy the right number but assign the wrong period or unit. Rather than continually add special rules, the final approach uses XBRL when the exact fact is tagged, and keeps the existing PDF/Groq route for details not covered by XBRL.

Camelot recovered several source-table layouts in separate offline experiments; that does **not** mean the complete production RAG was evaluated with Camelot. A Groq Vision fallback was implemented in a separate experiment but was **not live-validated**. Neither is claimed as a required or tested component of the final measured pipeline.

## Reproducing the research

The source code includes:

- `src/risklens/research/` — original PDF search and answering modules plus the XBRL reader and XBRL-first adapter.
- `scripts/phase7_xbrl_ask.py` — one-question XBRL-first command.
- `scripts/phase7_xbrl_offline_evaluate.py` — offline routing evaluation.
- `tools/phase7_live_eval/final_23_eval.py` — resumable optional Groq evaluation runner with a **75-second default minimum gap between API requests**, measured-token accounting, usage limits and checkpoints.
- `tests/test_phase7_xbrl_rag_v1.py` — targeted XBRL and integration tests, run locally with the eight real XML filings.

Install the research dependencies in a Python virtual environment (the exact versions can differ across systems):

```bash
pip install -e ".[research-all]" lxml
```

Example, from the repository root, after providing your own filings:

```bash
python scripts/phase7_xbrl_ask.py \
  'What were Infosys consolidated deferred tax liabilities (net) as of March 2025, in crore?' \
  --xbrl-archive "$HOME/Downloads/Archive.zip"
```

The original annual-report PDFs and built search index are needed for PDF fallback. **No real Groq request is made by the offline evaluation.** A live run should be separately authorized and uses the account owner's Groq quota.

The live runner can check the original 34-question CSV, but it does not publish that CSV or local Groq usage logs. Its bundled baseline records the original routing decisions, not the reference answers used by the model. The source dataset must be supplied separately to repeat the historical evaluation.

## Known limitations

1. Special financial disclosures may not have an exact XBRL tag. These questions rely on PDF search and may remain unverified.
2. Some correct PDF answers are refused, so **answer coverage is imperfect**.
3. Some numerical answers remain marked `needs_review` even when the amount matches the report, because source-context verification is incomplete.
4. A correct number is not enough: the year, standalone/consolidated scope, unit and financial item must also match.
5. The explanation route may add unnecessary source citations; retrieved supporting text should be reviewed.
6. The results are from four known companies and a benchmark seen during development. They do not establish generalization to new companies or unseen annual reports.
7. Six numerical benchmark cases were not fully run at the release checkpoint.
8. Current software requires local research dependencies and source documents; it is a **research prototype**, not a hosted website or consumer app.

## What did not change

The Phase 1–6 financial-risk model, XGBoost/LSTM model freeze, selected champion/challenger, sealed 433-row holdout and related governance files are **not touched by this release**. This branch contains only the Phase 7 research source and honest documentation, built separately from the existing remote `main` branch.

## Interview explanation (short and accurate)

> I built RiskLens as an explainable financial-risk research project. Its main component models downside-risk events on NIFTY 50. The supplementary research component answers questions from company financial filings. I integrated structured NSE XBRL data with PDF retrieval and Groq-generated explanations, keeping source references and conservative review statuses. On an exposed 34-question development set, 18 answers were source-supported, six unsupported requests were correctly refused, four answerable requests were incorrectly refused, and six numerical questions remained untested. The main limitation is answer coverage on complex financial disclosures, so I reported the limitation instead of claiming production accuracy.

## Release rule

This is a separate feature/release branch. **Do not automatically merge into `main`**. Treat any later merge as a separate reviewed decision.
