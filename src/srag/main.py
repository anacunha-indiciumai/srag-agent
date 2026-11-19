"""Generates the SRAG report: python -m srag.main"""
import sys
from pathlib import Path

import pandas as pd

from srag.agent.graph import build_graph
from srag.data.loader import load_cases, select_latest_per_year
from srag.governance.audit import AuditLog, sha256_file
from srag.llm.factory import LLMSettings, build_chat_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
BRAZIL_TIMEZONE = "America/Sao_Paulo"  # DATASUS dates are Brazilian calendar days, whatever the server clock says


def today_in_brazil(now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return now.tz_convert(BRAZIL_TIMEZONE).normalize().tz_localize(None)


def main() -> None:
    today = today_in_brazil()
    csv_files, ignored_files = select_latest_per_year(sorted(RAW_DATA_DIR.glob("INFLUD*.csv")))
    if not csv_files:
        raise SystemExit(f"No INFLUD*.csv files found in {RAW_DATA_DIR}")

    audit = AuditLog(LOGS_DIR / "audit.jsonl")
    audit.record("run.start", today=today, data_files=[path.name for path in csv_files],
                 data_provenance=[
                     {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in csv_files
                 ],
                 ignored_older_snapshots=[path.name for path in ignored_files])

    try:
        settings = LLMSettings.from_env()  # inside the try: a bad provider or missing model must leave a run.end
        audit.record("run.config", provider=settings.provider, model=settings.model)
        cases = load_cases(csv_files, today)
        graph = build_graph(cases, build_chat_model(settings), today, REPORTS_DIR, audit)
        result = graph.invoke({})
    except BaseException as error:  # includes Ctrl+C: an unfinished run must still leave an end marker
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        try:
            audit.record("run.end", status, error=type(error).__name__)
        except OSError as log_error:  # e.g. disk full: the original error is the one worth showing
            print(f"Aviso: não foi possível gravar o run.end no log ({type(log_error).__name__})", file=sys.stderr)
        raise

    report_path = Path(result["report_path"]).relative_to(REPORTS_DIR.parent).as_posix()  # no machine path in the log
    audit.record("run.end", report_path=report_path, used_fallback=result["used_fallback"])
    print(f"Relatório gerado: {result['report_path']}")


if __name__ == "__main__":
    main()
