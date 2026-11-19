"""The four report metrics, as pure functions over the cleaned cases table.

Every function takes the reference date (`today`) explicitly so results are
reproducible and auditable. All windows end at `today - MATURATION_DAYS`
because recent cases are still being notified (p90 delay is ~12 days).
"""
from dataclasses import dataclass

import pandas as pd

MATURATION_DAYS = 14
LONG_WINDOW_DAYS = 28
SHORT_WINDOW_DAYS = 7


@dataclass(frozen=True)
class CaseGrowth:
    window_days: int
    cutoff: pd.Timestamp
    current: int
    previous: int
    change_pct: float | None


@dataclass(frozen=True)
class Mortality:
    window_days: int
    cutoff: pd.Timestamp
    deaths: int
    closed_cases: int
    rate_pct: float | None
    pending_pct: float


@dataclass(frozen=True)
class IcuShare:
    window_days: int
    cutoff: pd.Timestamp
    icu: int
    hospitalized: int
    share_pct: float | None


@dataclass(frozen=True)
class VaccinationShare:
    window_days: int
    cutoff: pd.Timestamp
    flu_pct: float | None
    covid_pct: float | None


def cutoff_date(today: pd.Timestamp) -> pd.Timestamp:
    return today - pd.Timedelta(days=MATURATION_DAYS)


def case_growth(cases: pd.DataFrame, today: pd.Timestamp, window_days: int = LONG_WINDOW_DAYS) -> CaseGrowth:
    """Cases in the latest window vs. the window immediately before it."""
    cutoff = cutoff_date(today)
    current = len(_window(cases, cutoff, window_days))
    previous = len(_window(cases, cutoff - pd.Timedelta(days=window_days), window_days))
    return CaseGrowth(window_days, cutoff, current, previous, _change_pct(current, previous))


def mortality(cases: pd.DataFrame, today: pd.Timestamp, window_days: int = LONG_WINDOW_DAYS) -> Mortality:
    """SRAG deaths over cases with a known outcome, plus the share still without one.

    EVOLUCAO: 1=cure, 2=SRAG death, 3=death from other causes. Missing or 9 (unknown)
    means no outcome yet and is excluded from the denominator.
    """
    cutoff = cutoff_date(today)
    window = _window(cases, cutoff, window_days)
    closed = window["EVOLUCAO"].isin(["1", "2", "3"])
    deaths = int((window["EVOLUCAO"] == "2").sum())
    pending_pct = round(float((1 - closed.mean()) * 100), 1) if len(window) else 0.0
    return Mortality(window_days, cutoff, deaths, int(closed.sum()), _pct(deaths, int(closed.sum())), pending_pct)


def icu_share(cases: pd.DataFrame, today: pd.Timestamp, window_days: int = LONG_WINDOW_DAYS) -> IcuShare:
    """Hospitalized cases that went to the ICU. A proxy for severity, not bed occupancy
    (the dataset has no bed capacity)."""
    cutoff = cutoff_date(today)
    window = _window(cases, cutoff, window_days)
    hospitalized = window[(window["HOSPITAL"] == "1") & window["UTI"].isin(["1", "2"])]
    icu = int((hospitalized["UTI"] == "1").sum())
    return IcuShare(window_days, cutoff, icu, len(hospitalized), _pct(icu, len(hospitalized)))


def vaccination_share(
    cases: pd.DataFrame, today: pd.Timestamp, window_days: int = LONG_WINDOW_DAYS
) -> VaccinationShare:
    """Share of cases with a recorded vaccination, per vaccine. This is among SRAG cases,
    not population coverage."""
    cutoff = cutoff_date(today)
    window = _window(cases, cutoff, window_days)
    return VaccinationShare(window_days, cutoff, _yes_share(window["VACINA"]), _yes_share(window["VACINA_COV"]))


def _window(cases: pd.DataFrame, end: pd.Timestamp, window_days: int) -> pd.DataFrame:
    start = end - pd.Timedelta(days=window_days - 1)
    return cases[(cases["DT_SIN_PRI"] >= start) & (cases["DT_SIN_PRI"] <= end)]


def _yes_share(answers: pd.Series) -> float | None:
    """Percent answering yes (1) among those who answered yes or no (1 or 2)."""
    answered = answers.isin(["1", "2"])
    return _pct(int((answers == "1").sum()), int(answered.sum()))


def _pct(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 1) if denominator else None


def _change_pct(current: int, previous: int) -> float | None:
    return round((current / previous - 1) * 100, 1) if previous else None
