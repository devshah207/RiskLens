"""Import a manually downloaded NIFTY 50 historical CSV with traceable provenance.

The importer validates the source's STRUCTURE, not its authenticity. It never
fetches market data, repairs prices, fills dates, or trains a model.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from risklens.data.validation import OHLCValidationError, validate_ohlc


class ImportErrorData(ValueError):
    """Raised when the source cannot be safely imported as NIFTY 50 data."""


# Export formats from different NSE reporting pages use different column names.
COLUMN_ALIASES = {"index date": "date", "closing": "close"}


def _read_source(path: Path) -> pd.DataFrame:
    """Read all CSV columns as text so we do not silently change identifiers."""
    if not path.is_file():
        raise ImportErrorData(f"Source CSV file not found: {path}")
    try:
        return pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            skip_blank_lines=False,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ImportErrorData(f"Cannot read source CSV '{path}': {exc}") from exc


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize known header variants; reject ambiguous duplicate meanings."""
    renamed = frame.copy()
    normalized = [COLUMN_ALIASES.get(str(c).strip().casefold(), str(c).strip().casefold())
                  for c in renamed.columns]
    if len(set(normalized)) != len(normalized):
        raise ImportErrorData(
            "Duplicate or ambiguous column names after normalization. "
            "For example, do not supply both 'Close' and 'Closing'."
        )
    renamed.columns = normalized
    return renamed


def _select_nifty50(frame: pd.DataFrame, assume_nifty50: bool) -> tuple[pd.DataFrame, int, bool]:
    """Filter an explicitly named NIFTY 50 series, or require user attestation."""
    if "index name" not in frame.columns:
        if not assume_nifty50:
            raise ImportErrorData(
                "No 'Index Name' column. Use --assume-nifty50 only after "
                "checking manually that this entire CSV describes NIFTY 50."
            )
        return frame, 0, True

    names = frame["index name"].astype("string").str.strip()
    if names.isna().any() or names.eq("").any():
        raise ImportErrorData("Missing values in 'Index Name' column; fix source first.")
    keep = names.str.casefold().eq("nifty 50")
    if not keep.any():
        raise ImportErrorData("No 'NIFTY 50' records found in 'Index Name'.")
    dropped = int((~keep).sum())
    return frame.loc[keep].drop(columns=["index name"]).copy(), dropped, False


def import_nifty50_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    source_page: str | None = None,
    assume_nifty50: bool = False,
    overwrite: bool = False,
) -> dict:
    """Validate and import NIFTY 50 OHLC data; return a provenance report.

    Creates nifty50_ohlc.csv and nifty50_import_report.json in output_dir.
    The report records the *declared* source page, not verified authenticity.
    """
    source_path = Path(input_csv)
    output_path = Path(output_dir)
    cleaned_csv = output_path / "nifty50_ohlc.csv"
    report_json = output_path / "nifty50_import_report.json"

    if not overwrite and (cleaned_csv.exists() or report_json.exists()):
        raise ImportErrorData(
            "An output file already exists. Specify a different output directory "
            "or pass --overwrite after reviewing the existing outputs."
        )

    raw = _read_source(source_path)
    if raw.empty:
        raise ImportErrorData("Source CSV contains no observations.")
    normalized = _normalize_headers(raw)
    selected, excluded_rows, assumed = _select_nifty50(normalized, assume_nifty50)

    try:
        checked = validate_ohlc(selected)
    except OHLCValidationError as exc:
        raise ImportErrorData(f"OHLC validation failed: {exc}") from exc

    warnings = [
        "The importer checks structure and internal consistency, not NSE authenticity."
    ]
    if assumed:
        warnings.append("NIFTY 50 identity was asserted by the user, not verified in the CSV.")
    if excluded_rows:
        warnings.append(f"Excluded {excluded_rows} non-NIFTY 50 records.")
    if checked.attrs.get("reordered"):
        warnings.append("Source observations were not chronological; sorted by date.")

    report = {
        "instrument": "NIFTY 50 price index (user-identified source)",
        "claimed_source_page": source_page,
        "source_filename": source_path.name,
        "input_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_rows": len(raw),
        "excluded_non_nifty50_rows": excluded_rows,
        "identity_assumed_without_index_column": assumed,
        "rows_imported": len(checked),
        "first_session": checked["date"].min().date().isoformat(),
        "last_session": checked["date"].max().date().isoformat(),
        "ignored_extra_columns": checked.attrs.get("ignored_columns", []),
        "validation": "dates, positive finite prices, duplicate dates, OHLC bounds",
        "warnings": warnings,
    }

    # Only write output after every validation check succeeds.
    output_path.mkdir(parents=True, exist_ok=True)
    exported = checked.copy()
    exported["date"] = exported["date"].dt.strftime("%Y-%m-%d")
    exported.to_csv(cleaned_csv, index=False)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
