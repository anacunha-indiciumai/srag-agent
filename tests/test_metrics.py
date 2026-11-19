import pandas as pd
import pytest

from srag.tools import metrics

TODAY = pd.Timestamp("2026-11-17")
CUTOFF = pd.Timestamp("2026-11-03")  # TODAY - 14 days


def make_cases(rows: list[dict]) -> pd.DataFrame:
    defaults = {"EVOLUCAO": "1", "HOSPITAL": "1", "UTI": "2", "VACINA": "2", "VACINA_COV": "2"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def on(date: str, **fields) -> dict:
    return {"DT_SIN_PRI": pd.Timestamp(date), **fields}


def test_case_growth_compares_latest_window_with_the_previous_one():
    cases = make_cases(
        [on("2026-11-01")] * 6  # current 7-day window: 2026-10-28..2026-11-03
        + [on("2026-10-25")] * 4  # previous window: 2026-10-21..2026-10-27
    )
    result = metrics.case_growth(cases, TODAY, window_days=7)
    assert (result.current, result.previous, result.change_pct) == (6, 4, 50.0)
    assert result.cutoff == CUTOFF


def test_case_growth_ignores_cases_after_the_cutoff():
    cases = make_cases([on("2026-11-01")] * 3 + [on("2026-11-10")] * 50)
    assert metrics.case_growth(cases, TODAY, window_days=7).current == 3


def test_case_growth_is_none_when_previous_window_is_empty():
    result = metrics.case_growth(make_cases([on("2026-11-01")]), TODAY, window_days=7)
    assert result.change_pct is None


def test_mortality_excludes_cases_without_outcome_from_denominator():
    cases = make_cases(
        [on("2026-11-01", EVOLUCAO="2")] * 2
        + [on("2026-11-01", EVOLUCAO="1")] * 6
        + [on("2026-11-01", EVOLUCAO="3")] * 2  # death from other causes: closed, not a SRAG death
        + [on("2026-11-01", EVOLUCAO=None)] * 5
        + [on("2026-11-01", EVOLUCAO="9")] * 5
    )
    result = metrics.mortality(cases, TODAY, window_days=7)
    assert (result.deaths, result.closed_cases, result.rate_pct) == (2, 10, 20.0)
    assert result.pending_pct == 50.0


def test_mortality_with_no_closed_cases_has_no_rate():
    result = metrics.mortality(make_cases([on("2026-11-01", EVOLUCAO=None)]), TODAY, window_days=7)
    assert result.rate_pct is None
    assert result.pending_pct == 100.0


def test_icu_share_counts_only_hospitalized_cases_with_known_icu_status():
    cases = make_cases(
        [on("2026-11-01", HOSPITAL="1", UTI="1")] * 2
        + [on("2026-11-01", HOSPITAL="1", UTI="2")] * 6
        + [on("2026-11-01", HOSPITAL="1", UTI="9")] * 4  # unknown ICU status: excluded
        + [on("2026-11-01", HOSPITAL="2", UTI="2")] * 4  # not hospitalized: excluded
    )
    result = metrics.icu_share(cases, TODAY, window_days=7)
    assert (result.icu, result.hospitalized, result.share_pct) == (2, 8, 25.0)


def test_vaccination_share_is_computed_per_vaccine_among_those_who_answered():
    cases = make_cases(
        [on("2026-11-01", VACINA="1", VACINA_COV="1")] * 1
        + [on("2026-11-01", VACINA="2", VACINA_COV="1")] * 3
        + [on("2026-11-01", VACINA="9", VACINA_COV="2")] * 4  # unknown: excluded from flu
    )
    result = metrics.vaccination_share(cases, TODAY, window_days=7)
    assert result.flu_pct == 25.0
    assert result.covid_pct == 50.0


@pytest.mark.parametrize("function", [metrics.case_growth, metrics.mortality, metrics.icu_share])
def test_metrics_do_not_fail_on_empty_windows(function):
    empty = make_cases([on("2020-01-01")])
    function(empty, TODAY, window_days=7)
