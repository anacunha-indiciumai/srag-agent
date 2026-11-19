import pandas as pd

from srag.tools import charts

TODAY = pd.Timestamp("2026-11-18")  # cutoff = 2026-11-04


def make_cases(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"DT_SIN_PRI": pd.to_datetime(dates)})


def test_daily_series_covers_exactly_the_last_30_days_filling_gaps_with_zero():
    series = charts.daily_series(make_cases(["2026-11-10", "2026-11-10", "2026-11-12"]), TODAY)
    assert len(series) == 30
    assert series["period"].iloc[0] == pd.Timestamp("2026-10-20")
    assert series["period"].iloc[-1] == TODAY
    cases_by_day = dict(zip(series["period"], series["cases"]))
    assert cases_by_day[pd.Timestamp("2026-11-10")] == 2
    assert cases_by_day[pd.Timestamp("2026-11-11")] == 0


def test_daily_series_marks_the_last_14_days_as_incomplete():
    series = charts.daily_series(make_cases(["2026-11-10"]), TODAY)
    assert series["complete"].sum() == 16  # 2026-10-20..2026-11-04
    assert series["complete"].iloc[15] and not series["complete"].iloc[16]


def test_daily_series_ignores_cases_outside_the_window():
    series = charts.daily_series(make_cases(["2026-01-01"]), TODAY)
    assert series["cases"].sum() == 0


def test_monthly_series_covers_the_last_12_months_and_marks_the_current_one_incomplete():
    series = charts.monthly_series(make_cases(["2025-12-15", "2026-10-20", "2026-11-02"]), TODAY)
    assert len(series) == 12
    assert str(series["period"].iloc[0]) == "2025-12"
    assert str(series["period"].iloc[-1]) == "2026-11"
    assert list(series["cases"][[0, 10, 11]]) == [1, 1, 1]
    assert series["complete"].iloc[10] and not series["complete"].iloc[11]


def test_charts_are_written_as_png_files(tmp_path):
    cases = make_cases(["2026-11-01", "2026-10-01"])
    daily = charts.daily_cases_chart(cases, TODAY, tmp_path / "daily.png")
    monthly = charts.monthly_cases_chart(cases, TODAY, tmp_path / "monthly.png")
    for path in (daily, monthly):
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
