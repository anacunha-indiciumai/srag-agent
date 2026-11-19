from datetime import datetime, timezone

import pandas as pd

from srag.agent import report
from srag.tools.news import NewsItem, NewsResult

TODAY = pd.Timestamp("2026-11-18")  # cutoff 2026-11-04; current 28-day window starts 2026-10-08


def metrics_for(dates: list[str]) -> report.ReportMetrics:
    cases = pd.DataFrame({
        "DT_SIN_PRI": pd.to_datetime(dates),
        "EVOLUCAO": "1", "HOSPITAL": "1", "UTI": "2", "VACINA": "2", "VACINA_COV": "2",
    })
    return report.collect_metrics(cases, TODAY)


def news_with(title: str, source: str = "Fonte", link: str = "https://example.com/a") -> NewsResult:
    item = NewsItem(title, source, datetime(2026, 11, 16, tzinfo=timezone.utc), link)
    return NewsResult([item], dropped_suspicious=0)


def render(news: NewsResult) -> str:
    return report.render_markdown(TODAY, metrics_for(["2026-10-20"]), news, "Texto.")


def test_a_headline_cannot_open_its_own_markdown_link():
    markdown = render(news_with("x](https://evil.example)"))
    assert "- 16/11/2026 — Fonte: [x\\](https://evil.example)](https://example.com/a)" in markdown


def test_a_source_name_cannot_open_a_link_either():
    assert "Fonte\\[x\\]" in render(news_with("Título", source="Fonte[x]"))


def test_spaces_and_parentheses_in_a_feed_url_cannot_end_the_link_early():
    markdown = render(news_with("Título", link="https://example.com/a b(c)"))
    assert "](https://example.com/a%20b%28c%29)" in markdown


def test_fallback_says_there_is_no_comparison_base_when_the_previous_window_is_empty():
    text = report.fallback_explanation(metrics_for(["2026-10-20"] * 3))
    assert "sem base de comparação" in text
    assert "queda" not in text and "n/d" not in text.split("mortalidade")[0]


def test_fallback_says_stability_when_the_change_is_exactly_zero():
    text = report.fallback_explanation(metrics_for(["2026-10-20"] * 2 + ["2026-09-20"] * 2))
    assert "estabilidade" in text
    assert "queda" not in text and "alta" not in text


def test_fallback_says_queda_only_for_a_real_drop():
    text = report.fallback_explanation(metrics_for(["2026-10-20"] * 2 + ["2026-09-20"] * 4))
    assert "uma queda de 50,0%" in text
