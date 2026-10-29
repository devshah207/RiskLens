"""Combine yearly NIFTY 50 CSV exports and audit historical downside events.

The source CSVs remain unchanged. Risk-event frequency is computed ONLY for the
pre-declared development period; the final holdout outcomes remain unexamined.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from risklens.data.ingestion import (
    ImportErrorData,
    _normalize_headers,
    _read_source,
    _select_nifty50,
)
from risklens.data.validation import OHLCValidationError, validate_ohlc


OUTPUT_NAMES = (
    "nifty50_ohlc.csv",
    "nifty50_batch_import_report.json",
    "nifty50_event_frequency_development.json",
)


def _csv_paths(input_dir: Path) -> list[Path]:
    """Find CSV files directly inside the requested annual-export folder."""
    if not input_dir.is_dir():
        raise ImportErrorData(f"Input folder does not exist: {input_dir}")
    paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )
    if not paths:
        raise ImportErrorData(f"No CSV files found directly inside: {input_dir}")
    return paths


def _parse_cutoff(value: str) -> pd.Timestamp:
    """Require an unambiguous, predeclared YYYY-MM-DD development cutoff."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ImportErrorData(
            "--holdout-start must be a valid YYYY-MM-DD date"
        ) from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ImportErrorData("--holdout-start must use exactly YYYY-MM-DD")
    return pd.Timestamp(parsed)


def _read_one_file(path: Path, assume_nifty50: bool) -> tuple[pd.DataFrame, dict]:
    """Reuse the existing single-file reader, index filter and OHLC validator."""
    try:
        raw = _read_source(path)
        if raw.empty:
            raise ImportErrorData("CSV contains no observations")
        normalized = _normalize_headers(raw)
        selected, excluded, assumed = _select_nifty50(normalized, assume_nifty50)
        checked = validate_ohlc(selected)
    except (ImportErrorData, OHLCValidationError) as exc:
        raise ImportErrorData(f"{path.name}: {exc}") from exc

    metadata = {
        "source_filename": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "source_rows": int(len(raw)),
        "non_nifty50_rows_excluded": excluded,
        "valid_nifty50_rows": int(len(checked)),
        "first_session": checked["date"].min().date().isoformat(),
        "last_session": checked["date"].max().date().isoformat(),
        "identity_assumed_without_index_name": assumed,
        "reordered_within_file": bool(checked.attrs.get("reordered", False)),
        "ignored_extra_columns": checked.attrs.get("ignored_columns", []),
    }
    return checked, metadata


def _combine_and_check(
    frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, int, list[str]]:
    """Reject conflicting overlaps; explicitly count identical repeated dates."""
    combined = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable")
    duplicate_rows = combined[combined.duplicated("date", keep=False)]
    exact_repeats = 0
    repeated_dates = []
    for date, group in duplicate_rows.groupby("date", sort=True):
        numeric = group[["open", "high", "low", "close"]]
        if (numeric != numeric.iloc[0]).any().any():
            raise ImportErrorData(
                "Conflicting OHLC values across CSVs for trading date "
                f"{date.date().isoformat()}; examine the originals. No output was saved."
            )
        exact_repeats += len(group) - 1
        repeated_dates.append(date.date().isoformat())
    if exact_repeats:
        combined = combined.drop_duplicates("date", keep="first")
    combined = combined.reset_index(drop=True)
    # Validate the merged result too. It should retain unique, sorted dates.
    try:
        checked = validate_ohlc(combined)
    except OHLCValidationError as exc:
        raise ImportErrorData(f"Merged dataset failed validation: {exc}") from exc
    return checked, exact_repeats, repeated_dates


def _date_gap_report(dates: pd.Series, warning_days: int = 7) -> list[dict]:
    """Flag unusually long *calendar* gaps, without assuming trading holidays."""
    gaps = dates.diff().dt.days
    flagged = []
    for i in range(1, len(dates)):
        if int(gaps.iloc[i]) > warning_days:
            flagged.append(
                {
                    "previous_session": dates.iloc[i - 1].date().isoformat(),
                    "next_session": dates.iloc[i].date().isoformat(),
                    "calendar_days_between": int(gaps.iloc[i]),
                }
            )
    return flagged


def _development_event_frequency(
    data: pd.DataFrame, holdout_start: pd.Timestamp, threshold: float
) -> dict:
    """Count next-session downside labels only if BOTH dates precede holdout."""
    dev = data.loc[data["date"] < holdout_start].reset_index(drop=True)
    if len(dev) < 2:
        raise ImportErrorData(
            "Need at least two trading sessions before --holdout-start to count "
            "development downside events. Choose the cutoff before model selection."
        )
    # Drop the final development row: its next observed close may be in holdout.
    current = dev.iloc[:-1].copy()
    next_closes = dev["close"].iloc[1:].to_numpy()
    next_returns = (next_closes / current["close"].to_numpy()) - 1.0
    events = next_returns <= -threshold
    counts = []
    for year in sorted(current["date"].dt.year.unique()):
        year_mask = (current["date"].dt.year == year).to_numpy()
        total = int(year_mask.sum())
        positive = int(events[year_mask].sum())
        counts.append(
            {
                "prediction_year": int(year),
                "eligible_prediction_days": total,
                "downside_event_days": positive,
                "event_rate": round(positive / total, 6),
            }
        )
    positives = int(events.sum())
    total = len(current)
    return {
        "scope": "Development observations ONLY; holdout labels intentionally unexamined",
        "holdout_starts_on": holdout_start.date().isoformat(),
        "threshold": f"next observed trading-session close-to-close return <= {-threshold:.4f}",
        "prediction_timestamp": "after trading session t close",
        "eligible_prediction_days": total,
        "downside_event_days": positives,
        "non_event_days": total - positives,
        "event_rate": round(positives / total, 6),
        "by_prediction_year": counts,
        "last_development_session_not_labeled": dev["date"].iloc[-1].date().isoformat(),
        "method": "close[t+1]/close[t]-1, consecutive observed trading sessions within development period",
        "warning": "This is descriptive event frequency, NOT a trained model score or evidence of predictability.",
    }


def import_nifty50_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    assume_nifty50: bool = False,
    holdout_start: str = "2025-01-01",
    downside_threshold: float = 0.01,
    overwrite: bool = False,
    source_page: str | None = None,
) -> tuple[dict, dict]:
    """Merge annual exports, write standardized OHLC and two JSON audit reports.

    Nothing is written until every source passes validation and all overlaps
    are reconciled. Existing output files require explicit ``overwrite=True``.
    """
    if not (0 < downside_threshold < 1):
        raise ImportErrorData("Downside threshold must be between 0 and 1 (exclusive)")
    cutoff = _parse_cutoff(holdout_start)
    root = Path(input_dir)
    destination = Path(output_dir)
    paths = _csv_paths(root)
    targets = [destination / name for name in OUTPUT_NAMES]
    if not overwrite and any(target.exists() for target in targets):
        raise ImportErrorData(
            "Batch output already exists. Choose another output folder, "
            "or use --overwrite after reviewing existing files."
        )

    frames = []
    per_file = []
    for path in paths:
        frame, metadata = _read_one_file(path, assume_nifty50)
        frames.append(frame)
        per_file.append(metadata)

    combined, duplicate_count, repeated_dates = _combine_and_check(frames)
    gap_warnings = _date_gap_report(combined["date"])
    event_report = _development_event_frequency(combined, cutoff, downside_threshold)

    years = combined["date"].dt.year.value_counts().sort_index()
    warnings = [
        "CSV validation checks structure and internal consistency, not NSE authenticity.",
        "Gaps are calendar-day flags, not confirmed missing trading sessions; check NSE holidays.",
        "Event-frequency outcomes after the holdout cutoff have not been inspected.",
    ]
    if any(m["identity_assumed_without_index_name"] for m in per_file):
        warnings.append(
            "At least one file has no index-name column; NIFTY 50 identity was user-asserted."
        )
    if duplicate_count:
        warnings.append(
            f"Removed {duplicate_count} exact cross-file overlap rows, recorded below."
        )
    if gap_warnings:
        warnings.append(
            f"Review {len(gap_warnings)} calendar gaps longer than seven days."
        )
    if event_report["downside_event_days"] < 50:
        warnings.append(
            "Fewer than 50 development downside events; statistical estimates may be unstable."
        )

    audit = {
        "instrument": "NIFTY 50 price index (claimed source identity)",
        "source_page_claimed": source_page,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "number_of_files": len(paths),
        "input_files": per_file,
        "total_source_nifty_rows_including_overlaps": sum(
            m["valid_nifty50_rows"] for m in per_file
        ),
        "identical_cross_file_overlap_rows_removed": duplicate_count,
        "cross_file_overlap_dates": repeated_dates,
        "final_unique_trading_sessions": int(len(combined)),
        "first_session": combined["date"].min().date().isoformat(),
        "last_session": combined["date"].max().date().isoformat(),
        "observations_per_calendar_year": {str(k): int(v) for k, v in years.items()},
        "unusually_long_calendar_gaps": gap_warnings,
        "held_out_from_event_analysis_start": cutoff.date().isoformat(),
        "warnings": warnings,
    }

    destination.mkdir(parents=True, exist_ok=True)
    export = combined.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    # Stage all three files, then replace targets. This avoids creating partial
    # output if validation fails, though it is not a database transaction.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=destination) as temp:
        stage = Path(temp)
        export.to_csv(stage / OUTPUT_NAMES[0], index=False)
        (stage / OUTPUT_NAMES[1]).write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        (stage / OUTPUT_NAMES[2]).write_text(
            json.dumps(event_report, indent=2) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    return audit, event_report
