import json
from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from srag.agent import graph as graph_module
from srag.agent import report
from srag.agent.graph import build_graph
from srag.governance import guardrails
from srag.governance.audit import AuditLog
from srag.tools.news import FailedFeed, NewsItem, NewsResult

TODAY = pd.Timestamp("2026-11-18")


class FakeLLM:
    """Stands in for a chat model: records the messages it receives and answers from a script."""

    def __init__(self, *texts):
        self.texts = list(texts) or ["Cenário de teste."]
        self.received: list = []

    def invoke(self, messages):
        self.received.append(list(messages))
        text = self.texts[min(len(self.received), len(self.texts)) - 1]
        return type("Response", (), {"content": text})()


def make_cases() -> pd.DataFrame:
    # 11-10: after the cutoff, enough recent cases to keep the data "fresh" without touching the windows
    dates = ["2026-10-05"] * 10 + ["2026-10-20"] * 7 + ["2026-11-10"] * 10 + ["2026-09-01"]
    return pd.DataFrame({
        "DT_SIN_PRI": pd.to_datetime(dates),
        "EVOLUCAO": ["1"] * 17 + ["2"] + ["1"] * 10,
        "HOSPITAL": ["1"] * len(dates),
        "UTI": ["2"] * 15 + ["1"] * 3 + ["2"] * 9 + ["1"],
        "VACINA": ["2"] * len(dates),
        "VACINA_COV": ["1"] * len(dates),
    })


def fake_news() -> NewsResult:
    item = NewsItem("SRAG em alta", "Agência Saúde", datetime(2026, 11, 16, tzinfo=timezone.utc), "https://example.com/a")
    return NewsResult([item], dropped_suspicious=0)


def run(tmp_path, llm=None, news_fetcher=fake_news, cases=None, sleep=lambda seconds: None):
    llm = llm or FakeLLM()
    audit = AuditLog(tmp_path / "audit.jsonl", run_id="test-run")
    graph = build_graph(make_cases() if cases is None else cases, llm, TODAY, tmp_path, audit, news_fetcher, sleep)
    return graph.invoke({}), llm


def audit_events(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def read_report(tmp_path) -> str:
    return (tmp_path / "relatorio_2026-11-18.md").read_text(encoding="utf-8")


def test_pipeline_writes_the_report_and_both_charts(tmp_path):
    result, _ = run(tmp_path)
    assert result["report_path"] == str(tmp_path / "relatorio_2026-11-18.md")
    assert (tmp_path / "casos_diarios.png").exists()
    assert (tmp_path / "casos_mensais.png").exists()


def test_report_contains_code_generated_numbers_and_the_llm_explanation(tmp_path):
    run(tmp_path, FakeLLM("Texto da explicação."))
    markdown = read_report(tmp_path)
    assert "Texto da explicação." in markdown
    assert "Taxa de aumento de casos (28 dias)" in markdown
    assert "[SRAG em alta](https://example.com/a)" in markdown
    assert "## Limitações" in markdown
    assert "Texto gerado por IA" in markdown


def test_llm_receives_the_computed_metrics_and_the_news_as_untrusted_data(tmp_path):
    _, llm = run(tmp_path)
    system, human = llm.received[0]
    assert "Nunca calcule" in system.content
    assert "Mortalidade entre casos com desfecho" in human.content
    assert "não confiáveis" in human.content
    assert "SRAG em alta" in human.content


def test_the_prompt_tells_the_model_not_to_repeat_the_section_title(tmp_path):
    _, llm = run(tmp_path)
    assert "não escreva título" in llm.received[0][0].content


def test_report_is_still_produced_when_the_news_feed_is_down(tmp_path):
    def broken_feed() -> NewsResult:
        raise httpx.ConnectError("sem rede")

    _, llm = run(tmp_path, news_fetcher=broken_feed)
    assert "Notícias indisponíveis nesta execução." in read_report(tmp_path)
    assert "Notícias indisponíveis" in llm.received[0][1].content
    assert any(e["event"] == "decision.news_unavailable" for e in audit_events(tmp_path))


def test_report_is_still_produced_when_the_feed_host_is_rejected(tmp_path):
    def disallowed_host() -> NewsResult:
        raise ValueError("host not allowed")

    run(tmp_path, news_fetcher=disallowed_host)
    assert "Notícias indisponíveis nesta execução." in read_report(tmp_path)
    unavailable = [e for e in audit_events(tmp_path) if e["event"] == "decision.news_unavailable"]
    assert [(e["status"], e["error"]) for e in unavailable] == [("degraded", "ValueError")]


class FlakyNews:
    """Fails with the given errors in order, then answers like fake_news."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> NewsResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return fake_news()


def test_news_are_fetched_again_when_every_feed_failed_on_a_network_error(tmp_path):
    flaky = FlakyNews(httpx.ConnectError("sem rede"))
    pauses = []
    run(tmp_path, news_fetcher=flaky, sleep=pauses.append)
    assert flaky.calls == 2
    assert "SRAG em alta" in read_report(tmp_path)
    assert pauses == [graph_module.NEWS_RETRY_DELAY_SECONDS]
    retries = [e for e in audit_events(tmp_path) if e["event"] == "decision.news_retry"]
    assert [(e["status"], e["attempt"]) for e in retries] == [("degraded", 2)]
    assert [e["items"] for e in audit_events(tmp_path) if e["event"] == "step.news" and "items" in e] == [1]


def test_the_retry_gives_up_after_the_maximum_attempts_and_the_report_goes_on_without_news(tmp_path):
    always_down = FlakyNews(*[httpx.ConnectError("sem rede")] * 5)
    run(tmp_path, news_fetcher=always_down)
    assert always_down.calls == graph_module.MAX_NEWS_ATTEMPTS
    assert "Notícias indisponíveis nesta execução." in read_report(tmp_path)
    unavailable = [e for e in audit_events(tmp_path) if e["event"] == "decision.news_unavailable"]
    assert [e["attempt"] for e in unavailable] == list(range(1, graph_module.MAX_NEWS_ATTEMPTS + 1))


def test_a_rejected_feed_host_is_not_retried_because_waiting_would_not_fix_it(tmp_path):
    rejected = FlakyNews(ValueError("host not allowed"), ValueError("host not allowed"))
    run(tmp_path, news_fetcher=rejected)
    assert rejected.calls == 1
    assert not any(e["event"] == "decision.news_retry" for e in audit_events(tmp_path))


def test_news_that_arrive_the_first_time_are_not_fetched_again(tmp_path):
    flaky = FlakyNews()
    run(tmp_path, news_fetcher=flaky)
    assert flaky.calls == 1
    assert not any(e["event"] == "decision.news_retry" for e in audit_events(tmp_path))


def test_a_feed_that_failed_is_recorded_as_a_decision_while_the_others_still_feed_the_report(tmp_path):
    def partly_available() -> NewsResult:
        return NewsResult(fake_news().items, 0, failed_feeds=(FailedFeed("b.example.com", "HTTPStatusError", 500),))

    run(tmp_path, news_fetcher=partly_available)
    assert "SRAG em alta" in read_report(tmp_path)
    failed = [e for e in audit_events(tmp_path) if e["event"] == "decision.news_feed_unavailable"]
    assert [(e["status"], e["host"], e["error"], e["http_status"]) for e in failed] == [
        ("degraded", "b.example.com", "HTTPStatusError", 500)
    ]
    assert not any(e["event"] == "decision.news_unavailable" for e in audit_events(tmp_path))


def test_a_second_run_on_the_same_day_never_overwrites_the_first_report(tmp_path):
    first, _ = run(tmp_path, FakeLLM("Primeira execução."))
    second, _ = run(tmp_path, FakeLLM("Segunda execução."))
    third, _ = run(tmp_path, FakeLLM("Terceira execução."))
    assert [first["report_path"], second["report_path"], third["report_path"]] == [
        str(tmp_path / "relatorio_2026-11-18.md"),
        str(tmp_path / "relatorio_2026-11-18_2.md"),
        str(tmp_path / "relatorio_2026-11-18_3.md"),
    ]
    assert "Primeira execução." in read_report(tmp_path)
    assert "Segunda execução." in (tmp_path / "relatorio_2026-11-18_2.md").read_text(encoding="utf-8")


def test_explanation_is_extracted_from_a_list_of_content_blocks(tmp_path):
    run(tmp_path, FakeLLM([{"type": "text", "text": "Parte 1. "}, {"type": "text", "text": "Parte 2."}]))
    assert "Parte 1. Parte 2." in read_report(tmp_path)


def test_an_explanation_with_an_invented_number_is_retried_with_feedback(tmp_path):
    result, llm = run(tmp_path, FakeLLM("A mortalidade foi de 77%.", "Texto sem números inventados."))
    assert len(llm.received) == 2
    assert "violou estas regras" in llm.received[1][-1].content
    assert "77%" in llm.received[1][-1].content
    assert result["used_fallback"] is False
    assert "Texto sem números inventados." in read_report(tmp_path)


def test_when_every_attempt_is_blocked_a_code_generated_explanation_is_used(tmp_path):
    result, llm = run(tmp_path, FakeLLM("A mortalidade foi de 77%."))
    assert len(llm.received) == 2
    assert result["used_fallback"] is True
    markdown = read_report(tmp_path)
    assert "77%" not in markdown
    assert "sem IA" in markdown
    events = audit_events(tmp_path)
    assert [e["status"] for e in events if e["event"] == "guardrail.explanation"] == ["blocked", "blocked"]
    assert any(e["event"] == "decision.fallback_explanation" for e in events)


class FailingLLM:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise self.error


def test_a_temporary_llm_failure_still_produces_a_report_marked_as_not_ai(tmp_path):
    llm = FailingLLM(httpx.ConnectError("sem rede"))
    result, _ = run(tmp_path, llm)
    assert llm.calls == 1  # no point hammering a service that is down
    assert result["used_fallback"] is True
    assert "sem IA" in read_report(tmp_path)
    events = {e["event"]: e for e in audit_events(tmp_path)}
    assert events["llm.unavailable"]["status"] == "degraded"
    assert "ConnectError" in events["decision.fallback_explanation"]["reason"]


def test_a_configuration_error_stops_the_run_instead_of_hiding_behind_the_fallback(tmp_path):
    class Unauthorized(Exception):
        status_code = 401

    with pytest.raises(Unauthorized):
        run(tmp_path, FailingLLM(Unauthorized("token inválido")))
    assert not (tmp_path / "relatorio_2026-11-18.md").exists()
    assert any(e["event"] == "step.explain" and e["status"] == "error" for e in audit_events(tmp_path))


def test_the_fallback_explanation_itself_passes_the_guardrails(tmp_path):
    result, _ = run(tmp_path)
    metrics = result["metrics"]
    check = guardrails.check_explanation(report.fallback_explanation(metrics), report.metrics_for_prompt(metrics))
    assert check.passed, check.violations


def test_every_step_is_audited_with_the_same_run_id(tmp_path):
    run(tmp_path)
    events = audit_events(tmp_path)
    steps = [e["event"] for e in events if e["event"].startswith("step.")]
    assert steps == ["step.metrics", "step.charts", "step.news", "step.explain", "step.assemble"]
    assert {e["run_id"] for e in events} == {"test-run"}
    assert events[0]["event"] == "guardrail.input"


def test_the_audit_log_holds_summaries_and_hashes_never_case_rows(tmp_path):
    run(tmp_path)
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "report_sha256" in text and "explanation_sha256" in text
    assert "NU_IDADE_N" not in text


def test_columns_outside_the_allowlist_stop_the_run_before_any_processing(tmp_path):
    leaky = make_cases().assign(NU_IDADE_N="30")
    with pytest.raises(guardrails.GuardrailViolation):
        run(tmp_path, cases=leaky)
    assert not (tmp_path / "relatorio_2026-11-18.md").exists()
    blocked = [e for e in audit_events(tmp_path) if e["event"] == "guardrail.input"]
    assert [e["status"] for e in blocked] == ["blocked"]


def test_an_old_download_stops_the_run_because_the_windows_would_be_empty(tmp_path):
    old = make_cases().assign(DT_SIN_PRI=pd.Timestamp("2026-08-01"))
    with pytest.raises(guardrails.GuardrailViolation, match="01/08/2026"):
        run(tmp_path, cases=old)
    assert not (tmp_path / "relatorio_2026-11-18.md").exists()
    freshness = [e for e in audit_events(tmp_path) if e["event"] == "guardrail.freshness"]
    assert [e["status"] for e in freshness] == ["blocked"]


def test_a_stray_recent_date_does_not_keep_an_old_download_alive(tmp_path):
    old = make_cases().assign(DT_SIN_PRI=pd.Timestamp("2026-08-01"))
    old.loc[0, "DT_SIN_PRI"] = pd.Timestamp("2026-11-10")  # one mistyped date is all the "latest case" would see
    with pytest.raises(guardrails.GuardrailViolation, match="Only 1 cases"):
        run(tmp_path, cases=old)
    freshness = [e for e in audit_events(tmp_path) if e["event"] == "guardrail.freshness"]
    assert [(e["status"], e["recent_cases"]) for e in freshness] == [("blocked", 1)]


def test_fresh_data_records_a_passing_freshness_check(tmp_path):
    run(tmp_path)
    freshness = [e for e in audit_events(tmp_path) if e["event"] == "guardrail.freshness"]
    assert [(e["status"], e["recent_cases"]) for e in freshness] == [("ok", 10)]
