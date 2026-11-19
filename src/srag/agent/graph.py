"""Report orchestrator: a LangGraph pipeline with a fixed order and one conditional edge.

The tool steps always run in the same order and the LLM only writes the explanation. Compared
with letting the LLM pick tools, this is more reliable with small local models and easier to audit.
The one branch is code-decided: when every news feed fails on a network error, the news step runs
once more before the report goes on without news.
Every step is recorded in the audit log and the LLM output passes through the guardrails.
"""
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, TypedDict

import httpx
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from srag.agent import report
from srag.agent.prompts import EXPLAIN_SYSTEM_PROMPT, EXPLAIN_USER_TEMPLATE
from srag.governance import guardrails
from srag.governance.audit import AuditLog, sha256_text
from srag.llm.errors import is_transient
from srag.tools import charts, news as news_tool
from srag.tools.news import NewsResult

MAX_EXPLANATION_ATTEMPTS = 2
MAX_DATA_AGE_DAYS = 21  # recent cases are counted over this many days (cutoff is 14 days, plus notification slack)
MIN_RECENT_CASES = 10  # a stray recent date must not keep an old download alive
MAX_NEWS_ATTEMPTS = 2  # one retry: a passing network failure should not leave the report without news
NEWS_RETRY_DELAY_SECONDS = 2


class ReportState(TypedDict, total=False):
    metrics: report.ReportMetrics
    news: NewsResult | None
    news_attempts: int
    news_retryable: bool  # a network error may pass; a rejected feed host will not
    explanation: str
    used_fallback: bool
    report_path: str


def build_graph(
    cases: pd.DataFrame,
    llm: BaseChatModel,
    today: pd.Timestamp,
    out_dir: Path,
    audit: AuditLog,
    news_fetcher: Callable[[], NewsResult] = news_tool.fetch_news,
    sleep: Callable[[float], None] = time.sleep,
):
    try:
        guardrails.check_allowed_columns(cases)
    except guardrails.GuardrailViolation:
        audit.record("guardrail.input", "blocked", columns=sorted(cases.columns), rows=len(cases))
        raise
    audit.record("guardrail.input", columns=sorted(cases.columns), rows=len(cases))
    _check_freshness(cases, today, audit)

    def metrics_node(state: ReportState) -> dict:
        return {"metrics": report.collect_metrics(cases, today)}

    def charts_node(state: ReportState) -> dict:
        charts.daily_cases_chart(cases, today, out_dir / report.DAILY_CHART_FILE)
        charts.monthly_cases_chart(cases, today, out_dir / report.MONTHLY_CHART_FILE)
        return {}

    def news_node(state: ReportState) -> dict:
        attempt = state.get("news_attempts", 0) + 1
        try:
            result = news_fetcher()
        except (httpx.HTTPError, ValueError) as error:  # ValueError: feed host not on the allowlist
            audit.record("decision.news_unavailable", "degraded", error=type(error).__name__, attempt=attempt)
            return {"news": None, "news_attempts": attempt, "news_retryable": isinstance(error, httpx.HTTPError)}
        for feed in result.failed_feeds:  # some feeds answered: the report goes on with what arrived
            audit.record(
                "decision.news_feed_unavailable", "degraded",
                host=feed.host, error=feed.error, http_status=feed.status_code,
            )
        return {"news": result, "news_attempts": attempt, "news_retryable": False}

    def after_news(state: ReportState) -> str:
        """The graph's only branch: fetch the news again, or go on to the explanation."""
        if state["news"] is None and state["news_retryable"] and state["news_attempts"] < MAX_NEWS_ATTEMPTS:
            audit.record("decision.news_retry", "degraded", attempt=state["news_attempts"] + 1,
                         max_attempts=MAX_NEWS_ATTEMPTS)
            sleep(NEWS_RETRY_DELAY_SECONDS)
            return "retry"
        return "continue"

    def explain_node(state: ReportState) -> dict:
        news = state["news"]
        news_text = news_tool.format_for_prompt(news) if news else "Notícias indisponíveis nesta execução."
        prompt = EXPLAIN_USER_TEMPLATE.format(metrics=report.metrics_for_prompt(state["metrics"]), news=news_text)
        explanation, failure = _explain_with_guardrails(llm, prompt, audit)
        if explanation is None:
            audit.record("decision.fallback_explanation", "degraded", reason=failure)
            return {"explanation": report.fallback_explanation(state["metrics"]), "used_fallback": True}
        return {"explanation": explanation, "used_fallback": False}

    def assemble_node(state: ReportState) -> dict:
        markdown = report.render_markdown(
            today, state["metrics"], state["news"], state["explanation"], generated_by_llm=not state["used_fallback"]
        )
        # Charts keep fixed names on purpose: they are regenerated artifacts, only reports are preserved.
        path = _free_report_path(out_dir, today)
        path.write_text(markdown, encoding="utf-8")
        return {"report_path": str(path)}

    def llm_name() -> str:
        return str(getattr(llm, "model", type(llm).__name__))

    steps = [
        ("metrics", metrics_node, lambda update: {"metrics": asdict(update["metrics"])}),
        ("charts", charts_node, lambda update: {"files": [report.DAILY_CHART_FILE, report.MONTHLY_CHART_FILE]}),
        ("news", news_node, _summarize_news),
        (
            "explain",
            explain_node,
            lambda update: {
                "model": llm_name(),
                "used_fallback": update["used_fallback"],
                "explanation_sha256": sha256_text(update["explanation"]),
            },
        ),
        (
            "assemble",
            assemble_node,
            lambda update: {
                # relative, so the log never records a machine path
                "report_path": Path(update["report_path"]).relative_to(out_dir.parent).as_posix(),
                "report_sha256": sha256_text(Path(update["report_path"]).read_text(encoding="utf-8")),
            },
        ),
    ]
    workflow = StateGraph(ReportState)
    for name, node, summarize in steps:
        workflow.add_node(name, _audited(name, node, summarize, audit))
    names = [name for name, _, _ in steps]
    retryable_steps = {"news": after_news}
    workflow.add_edge(START, names[0])
    for current, following in zip(names, names[1:]):
        if current in retryable_steps:
            workflow.add_conditional_edges(current, retryable_steps[current], {"retry": current, "continue": following})
        else:
            workflow.add_edge(current, following)
    workflow.add_edge(names[-1], END)
    return workflow.compile()


def _free_report_path(out_dir: Path, today: pd.Timestamp) -> Path:
    """First run of the day keeps the plain name; later runs get _2, _3, ... so no report is overwritten."""
    base = f"relatorio_{today:%Y-%m-%d}"
    path = out_dir / f"{base}.md"
    suffix = 2
    while path.exists():
        path = out_dir / f"{base}_{suffix}.md"
        suffix += 1
    return path


def _check_freshness(cases: pd.DataFrame, today: pd.Timestamp, audit: AuditLog) -> None:
    """Stop on an old download: the windows would be empty and the report would look confidently calm."""
    dates = cases["DT_SIN_PRI"]
    latest = dates.max()
    recent = int((dates >= today - pd.Timedelta(days=MAX_DATA_AGE_DAYS)).sum())
    summary = {"latest_case": latest, "recent_cases": recent, "max_age_days": MAX_DATA_AGE_DAYS}
    if pd.isna(latest) or recent < MIN_RECENT_CASES:
        audit.record("guardrail.freshness", "blocked", **summary, min_recent_cases=MIN_RECENT_CASES)
        raise guardrails.GuardrailViolation(
            f"Only {recent} cases in the last {MAX_DATA_AGE_DAYS} days (latest: {latest:%d/%m/%Y}): "
            "download a newer Open DATASUS file." if pd.notna(latest) else "No case with a valid symptom date."
        )
    audit.record("guardrail.freshness", **summary)


def _explain_with_guardrails(llm: BaseChatModel, prompt: str, audit: AuditLog) -> tuple[str | None, str | None]:
    """Ask the LLM, validate the answer, and retry once with feedback.

    Returns (text, None) on success or (None, reason) when the code-generated explanation must be used.
    A temporary LLM failure gives up right away (the SDKs already retry internally); any other
    exception propagates so a broken configuration is never hidden behind the fallback.
    """
    messages = [SystemMessage(EXPLAIN_SYSTEM_PROMPT), HumanMessage(prompt)]
    for attempt in range(1, MAX_EXPLANATION_ATTEMPTS + 1):
        try:
            text = _text(llm.invoke(messages).content)
        except Exception as error:
            if not is_transient(error):
                raise
            audit.record("llm.unavailable", "degraded", attempt=attempt, error=type(error).__name__)
            return None, f"LLM temporarily unavailable ({type(error).__name__})"
        result = guardrails.check_explanation(text, prompt)
        audit.record(
            "guardrail.explanation",
            "pass" if result.passed else "blocked",
            attempt=attempt,
            violations=result.violations,
            response_sha256=sha256_text(text),
        )
        if result.passed:
            return text, None
        feedback = "Sua resposta violou estas regras: " + "; ".join(result.violations) + ". Reescreva corrigindo."
        messages += [AIMessage(text), HumanMessage(feedback)]
    return None, "guardrails blocked every LLM attempt"


def _audited(step: str, node: Callable[[ReportState], dict], summarize: Callable[[dict], dict], audit: AuditLog):
    def wrapper(state: ReportState) -> dict:
        started = time.perf_counter()
        try:
            update = node(state)
        except Exception as error:
            audit.record(f"step.{step}", "error", error=type(error).__name__, duration_ms=_elapsed_ms(started))
            raise
        audit.record(f"step.{step}", duration_ms=_elapsed_ms(started), **summarize(update))
        return update

    return wrapper


def _summarize_news(update: dict) -> dict:
    news = update["news"]
    if news is None:
        return {"unavailable": True}
    return {
        "items": len(news.items),
        "dropped_suspicious": news.dropped_suspicious,
        "titles": [item.title for item in news.items],
    }


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _text(content: str | list) -> str:
    """Chat models return either a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
