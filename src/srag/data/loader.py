"""Loads and cleans SRAG cases from the raw Open DATASUS CSVs."""
import re
from datetime import date
from pathlib import Path

import pandas as pd

from srag.data.schema import ALLOWED_COLUMNS, CATEGORICAL_COLUMNS, DATE_COLUMNS, EARLIEST_VALID_DATE, VALID_CODES

# Open DATASUS names each snapshot INFLUD<yy>-<dd>-<mm>-<yyyy>.csv (year of the data, then extraction date).
_SNAPSHOT_NAME = re.compile(r"^(INFLUD\d{2})-(\d{2})-(\d{2})-(\d{4})\.csv$", re.IGNORECASE)


def select_latest_per_year(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Keep only the newest snapshot of each year; older ones would count the same cases twice.

    Returns (selected, ignored). Nothing is deleted. Files whose name does not follow the pattern
    are kept as they are.
    """
    newest: dict[str, tuple[date, Path]] = {}
    ignored: list[Path] = []
    for path in sorted(paths):
        key, snapshot = _snapshot_key(path)
        if key not in newest:
            newest[key] = (snapshot, path)
        elif snapshot > newest[key][0]:
            ignored.append(newest[key][1])
            newest[key] = (snapshot, path)
        else:
            ignored.append(path)
    return sorted(path for _, path in newest.values()), sorted(ignored)


def _snapshot_key(path: Path) -> tuple[str, date]:
    match = _SNAPSHOT_NAME.match(path.name)
    if match:
        year_key, day, month, year = match.groups()
        try:
            return year_key.upper(), date(int(year), int(month), int(day))
        except ValueError:
            pass  # impossible date in the name: treat the file as unrecognized
    return path.name, date.min


def load_cases(paths: list[Path], today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per case with only allowlisted columns, cleaned."""
    today = today or pd.Timestamp.today().normalize()
    frames = [
        pd.read_csv(path, sep=";", usecols=ALLOWED_COLUMNS, dtype=str, encoding="latin-1") for path in paths
    ]
    cases = pd.concat(frames, ignore_index=True)

    for column in DATE_COLUMNS:
        cases[column] = _parse_date(cases[column], today)
    for column in CATEGORICAL_COLUMNS:
        cases[column] = _clean_code(cases[column], VALID_CODES[column])

    cases = cases.dropna(subset=["DT_SIN_PRI"])
    cases["HOSPITAL"] = _uti_implies_hospital(cases)
    return cases


def _parse_date(series: pd.Series, today: pd.Timestamp) -> pd.Series:
    # Raw values look like 2026-01-05T00:00:00.000Z; keep only the date part.
    parsed = pd.to_datetime(series.str[:10], format="%Y-%m-%d", errors="coerce")
    return parsed.where((parsed >= EARLIEST_VALID_DATE) & (parsed <= today))


def _clean_code(series: pd.Series, valid: set[str]) -> pd.Series:
    # Some columns arrive as floats in text ("6.0000000000"); normalize before validating.
    normalized = series.str.split(".").str[0]
    return normalized.where(normalized.isin(valid))


def _uti_implies_hospital(cases: pd.DataFrame) -> pd.Series:
    return cases["HOSPITAL"].mask(cases["UTI"] == "1", "1")
