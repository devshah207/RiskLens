# RiskLens Phase 9 — Release acceptance and source integrity

**Reviewed:** 2 November 2026. **Baseline:** `main` at `986eef114a1c1bd7274cb4504e62d24c61327904`, the merged Phase 8 [PR #8](https://github.com/devshah207/RiskLens/pull/8). Earlier Phase 6 and 7 PRs (#6, #7) are also merged.

## Scope of Phase 9

Phase 9 updates only `README.md`, `docs/ROADMAP.md`, and `docs/PHASE9_*.md`. It does not alter the approved premium UI, the original ML/RAG modules, protected evidence hashes, saved predictions, sources, or credentials.

## Evidence and acceptance checks

| Evidence | Release rule and verification | Limit |
|---|---|---|
| Phase 8 GitHub CI | Two completed successful Phase 8 branch workflow runs on final Phase 8 commit `f1330e0` | Code lint/tests do not prove financial accuracy |
| Phase 8 local app suite | Previous premium release build: 45 passing offline Phase 8 tests | Not a full Streamlit browser or live Groq test |
| Owner-local research preflight | Owner's terminal: real XBRL result `answered` with citation; PDF retrieval returned 5 candidate pages | No Groq generation in offline preflight |
| Phase 6 governance | Original Mac freeze verifier had earlier passed 9/9 protected hashes | This assistant cannot independently verify all originals in its container; no reruns of the holdout |
| Published Phase 7 research | Historical exposed 34-case outcomes 18 supported, 6 correct refusals, 4 incorrect refusals, 6 not run | Not blind accuracy or human-certified financial audit |
| Phase 9 documentation | Ensure README architecture/results match originals, no broken relative links, no secrets in commits, docs-only changes | Screenshots are optional follow-up work; no synthetic captures in release |
| GitHub Actions on Phase 9 PR | Require successful lint and test run(s) on the documentation PR before merging | If CI is pending/fails, do not mark remote publication complete |
| Source integrity | Compare changed filenames against `README.md` and `docs/` only | No `src/`, `app/`, `data/`, original PDFs/XML, keys, `.venv`, or frozen artifacts |

## Local verification commands (no retraining)

```bash
cd ~/Downloads/RiskLens
source .venv/bin/activate
PYTHONPATH=.:src python -m pytest -q tests/test_phase8_*.py
ruff check app src tests scripts --select F,E4,E7,E9
python scripts/phase8_rag_preflight.py --pdf-offline
python scripts/verify_phase6_freeze.py
python -m streamlit run app/Home.py
```

The freeze verifier requires original local protected files and may be unavailable in partial checkouts. The PDF offline preflight deliberately does **not** claim a Groq-generated answer.

## Portfolio visual policy

Real screenshots must be captured from a functioning instance of the **premium** interface with private paths, usernames, API keys and other secrets removed. None are included in the Phase 9 documentation release because no verified premium screenshot was supplied. Screenshots are a desirable portfolio enhancement, **not** a prerequisite to publishing truthful documentation and source code.

## Outstanding research (not Phase 9 release blockers)

A prospective blind market test, independently verified broad financial RAG benchmark, improved predeclared alert-threshold policy and fully authenticated live Groq UI demo would be needed before making production-grade claims. These are explicitly not retroactive changes to the frozen Phase 5/6 test.
