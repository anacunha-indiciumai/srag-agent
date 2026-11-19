import hashlib
import json

import pandas as pd
import pytest

from srag import main as entrypoint


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "INFLUD99.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(entrypoint, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(entrypoint, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(entrypoint, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path


def audit_events(workspace) -> list[dict]:
    lines = (workspace / "logs" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize("failure, status", [(RuntimeError("boom"), "error"), (KeyboardInterrupt(), "interrupted")])
def test_a_run_that_fails_or_is_interrupted_still_records_how_it_ended(workspace, monkeypatch, failure, status):
    def failing_load(*args, **kwargs):
        raise failure

    monkeypatch.setattr(entrypoint, "load_cases", failing_load)
    with pytest.raises(type(failure)):
        entrypoint.main()
    events = audit_events(workspace)
    start, end = events[0], events[-1]
    assert (start["event"], end["event"], end["status"]) == ("run.start", "run.end", status)
    assert end["error"] == type(failure).__name__
    assert "boom" not in json.dumps(end)  # only the exception type is logged, never its message


def test_today_follows_brazil_time_not_the_server_clock():
    evening_in_brazil = pd.Timestamp("2026-11-20 01:30", tz="UTC")  # still the 19th, 22:30 in Brasília
    assert entrypoint.today_in_brazil(evening_in_brazil) == pd.Timestamp("2026-11-19")
    assert entrypoint.today_in_brazil(pd.Timestamp("2026-11-20 03:00", tz="UTC")) == pd.Timestamp("2026-11-20")


def test_a_failure_to_write_run_end_does_not_hide_the_original_error(workspace, monkeypatch):
    record = entrypoint.AuditLog.record

    def record_unless_run_end(self, event, *args, **kwargs):
        if event == "run.end":
            raise OSError("disk full")
        record(self, event, *args, **kwargs)

    def failing_load(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(entrypoint.AuditLog, "record", record_unless_run_end)
    monkeypatch.setattr(entrypoint, "load_cases", failing_load)
    with pytest.raises(RuntimeError, match="boom"):
        entrypoint.main()


def test_an_invalid_llm_setting_still_leaves_a_run_end(workspace, monkeypatch):
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "nao-existe")
    with pytest.raises(ValueError):
        entrypoint.main()
    events = audit_events(workspace)
    assert [e["event"] for e in events] == ["run.start", "run.end"]
    assert events[-1]["status"] == "error"


def test_running_without_data_files_stops_before_creating_any_log(workspace, monkeypatch):
    (workspace / "raw" / "INFLUD99.csv").unlink()
    with pytest.raises(SystemExit):
        entrypoint.main()
    assert not (workspace / "logs").exists()


def test_run_start_records_the_hash_and_size_of_each_input_file_by_name_only(workspace, monkeypatch):
    (workspace / "raw" / "INFLUD99.csv").write_text("abc", encoding="utf-8")
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "nao-existe")  # stops the run right after run.start
    with pytest.raises(ValueError):
        entrypoint.main()
    start = audit_events(workspace)[0]
    assert start["data_provenance"] == [
        {"file": "INFLUD99.csv", "sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}
    ]
    assert str(workspace) not in json.dumps(start)
