import json
import threading

import pytest

from srag.governance.audit import AuditLog, sha256_file, sha256_text, verify_chain


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_record_is_one_json_line_with_run_id_event_and_status(tmp_path):
    audit = AuditLog(tmp_path / "logs" / "audit.jsonl", run_id="run-1")
    audit.record("step.metrics", duration_ms=12)
    audit.record("guardrail.explanation", "blocked", violations=["x"])
    first, second = read(tmp_path / "logs" / "audit.jsonl")
    assert (first["run_id"], first["event"], first["status"], first["duration_ms"]) == ("run-1", "step.metrics", "ok", 12)
    assert (second["status"], second["violations"]) == ("blocked", ["x"])
    assert "timestamp" in first


def test_events_carry_a_sequence_number_that_restarts_in_each_run(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path, run_id="a")
    first.record("one")
    first.record("two")
    AuditLog(path, run_id="b").record("three")
    assert [(e["run_id"], e["seq"]) for e in read(path)] == [("a", 1), ("a", 2), ("b", 1)]


def test_the_log_is_append_only_across_runs(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path, run_id="a").record("one")
    AuditLog(path, run_id="b").record("two")
    assert [entry["run_id"] for entry in read(path)] == ["a", "b"]


def test_unserializable_values_are_stringified_instead_of_failing(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record("run.start", today=__import__("pandas").Timestamp("2026-11-18"))
    assert "2026-11-18" in read(tmp_path / "audit.jsonl")[0]["today"]


def test_sha256_is_stable():
    assert sha256_text("a") == sha256_text("a") != sha256_text("b")


def write_log(path, count=4):
    audit = AuditLog(path, run_id="a")
    for index in range(count):
        audit.record(f"event.{index}")


def test_an_untouched_chain_is_intact(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_log(path)
    assert read(path)[0]["prev_hash"] == ""
    assert verify_chain(path) == []


def test_an_edited_line_is_detected_at_the_next_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("event.1", "event.X")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_chain(path) == [3]


def test_a_deleted_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_chain(path) == [2]


def test_the_chain_continues_across_audit_log_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_log(path, 2)
    AuditLog(path, run_id="b").record("later")
    assert verify_chain(path) == []
    assert len(read(path)) == 3


@pytest.mark.parametrize("field", ["timestamp", "run_id", "seq", "prev_hash"])
def test_details_cannot_overwrite_the_fields_that_make_the_log_verifiable(tmp_path, field):
    path = tmp_path / "audit.jsonl"
    with pytest.raises(ValueError, match=field):
        AuditLog(path).record("step.x", **{field: "forged"})
    assert not path.exists()


def test_instances_writing_alternately_keep_the_chain_intact(tmp_path):
    path = tmp_path / "audit.jsonl"
    first, second = AuditLog(path, run_id="a"), AuditLog(path, run_id="b")
    for index in range(3):
        first.record(f"a.{index}")
        second.record(f"b.{index}")
    assert verify_chain(path) == []
    assert len(read(path)) == 6


def test_concurrent_runs_never_break_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"

    def run(run_id):
        audit = AuditLog(path, run_id=run_id)
        for index in range(25):
            audit.record(f"event.{index}")

    threads = [threading.Thread(target=run, args=(f"run-{number}",)) for number in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(read(path)) == 8 * 25
    assert verify_chain(path) == []


def test_a_missing_or_empty_file_has_no_broken_links(tmp_path):
    assert verify_chain(tmp_path / "nope.jsonl") == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert verify_chain(empty) == []
    AuditLog(empty).record("first")
    assert verify_chain(empty) == []


def test_sha256_file_matches_the_text_hash_and_streams_in_chunks(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("abcdef" * 100, encoding="utf-8")
    assert sha256_file(path, chunk_size=7) == sha256_text("abcdef" * 100)
