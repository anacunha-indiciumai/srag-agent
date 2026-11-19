from pathlib import Path

from srag.data.loader import select_latest_per_year


def paths(*names: str) -> list[Path]:
    return [Path("data/raw") / name for name in names]


def names(files: list[Path]) -> list[str]:
    return [file.name for file in files]


def test_different_years_are_all_kept():
    selected, ignored = select_latest_per_year(paths("INFLUD25-16-11-2026.csv", "INFLUD26-16-11-2026.csv"))
    assert names(selected) == ["INFLUD25-16-11-2026.csv", "INFLUD26-16-11-2026.csv"]
    assert ignored == []


def test_only_the_newest_snapshot_of_a_year_is_kept():
    selected, ignored = select_latest_per_year(paths("INFLUD25-02-10-2026.csv", "INFLUD25-16-11-2026.csv"))
    assert names(selected) == ["INFLUD25-16-11-2026.csv"]
    assert names(ignored) == ["INFLUD25-02-10-2026.csv"]


def test_snapshots_are_compared_by_date_not_by_alphabetical_order():
    # 03-01-2027 is later than 16-11-2026 although "03" sorts before "16".
    selected, _ = select_latest_per_year(paths("INFLUD25-16-11-2026.csv", "INFLUD25-03-01-2027.csv"))
    assert names(selected) == ["INFLUD25-03-01-2027.csv"]


def test_the_result_does_not_depend_on_the_order_of_the_input():
    forward = select_latest_per_year(paths("INFLUD25-02-10-2026.csv", "INFLUD25-16-11-2026.csv"))
    backward = select_latest_per_year(paths("INFLUD25-16-11-2026.csv", "INFLUD25-02-10-2026.csv"))
    assert forward == backward


def test_files_with_an_unrecognized_name_are_kept():
    selected, ignored = select_latest_per_year(paths("INFLUD25-16-11-2026.csv", "INFLUD-teste.csv", "INFLUD25-99-99-2026.csv"))
    assert names(selected) == ["INFLUD-teste.csv", "INFLUD25-16-11-2026.csv", "INFLUD25-99-99-2026.csv"]
    assert ignored == []


def test_the_allowlist_holds_only_the_six_columns_the_pipeline_uses():
    from srag.data.schema import ALLOWED_COLUMNS

    assert sorted(ALLOWED_COLUMNS) == ["DT_SIN_PRI", "EVOLUCAO", "HOSPITAL", "UTI", "VACINA", "VACINA_COV"]
