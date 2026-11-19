"""Append-only audit trail (JSON Lines): one event per pipeline step and per agent decision.

Entries carry summaries, hashes and counts, never case-level rows, so the log is safe to keep
and share. Every run has its own `run_id` to reconstruct what happened in that execution.
"""
import fcntl
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


GENESIS_HASH = ""  # prev_hash of the very first line of a log
RESERVED_FIELDS = {"timestamp", "run_id", "seq", "prev_hash"}  # set by the log itself, never by the caller


class AuditLog:
    """Each entry stores the sha256 of the previous line, so editing or deleting a line breaks the chain."""

    def __init__(self, path: Path, run_id: str | None = None):
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, status: str = "ok", **details) -> None:
        reserved = sorted(RESERVED_FIELDS & details.keys())
        if reserved:  # a detail with one of these names would silently replace a field that makes the log verifiable
            raise ValueError(f"Audit details cannot use reserved fields: {', '.join(reserved)}")
        self._sequence += 1
        with self.path.open("a", encoding="utf-8") as file:
            # The lock lasts until the file closes, so the previous hash is read and the new line appended as one step:
            # concurrent runs, or another instance, can no longer link to the same line.
            fcntl.flock(file, fcntl.LOCK_EX)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "run_id": self.run_id,
                "seq": self._sequence,  # orders events within a run even if the system clock jumps
                "event": event,
                "status": status,
                **details,
                "prev_hash": _hash_of_last_line(self.path),
            }
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def verify_chain(path: Path) -> list[int]:
    """1-based line numbers whose prev_hash does not match the line before them (empty = intact)."""
    if not path.exists():
        return []
    broken = []
    expected = GENESIS_HASH
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            recorded = json.loads(line).get("prev_hash")
        except (json.JSONDecodeError, AttributeError):
            recorded = None
        if recorded != expected:
            broken.append(number)
        expected = sha256_text(line)
    return broken


def _hash_of_last_line(path: Path) -> str:
    if not path.exists():
        return GENESIS_HASH
    lines = path.read_text(encoding="utf-8").splitlines()
    return sha256_text(lines[-1]) if lines else GENESIS_HASH


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
