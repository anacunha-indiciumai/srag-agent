"""Report assembly. All numbers, tables and links are produced here from code; the LLM only
contributes the explanation text."""
from dataclasses import dataclass

import pandas as pd

from srag.tools import metrics
from srag.tools.news import NewsResult

DAILY_CHART_FILE = "casos_diarios.png"
MONTHLY_CHART_FILE = "casos_mensais.png"

LIMITATIONS = [
    "Casos recentes ainda estão sendo notificados: as métricas usam um corte de "
    f"{metrics.MATURATION_DAYS} dias e a taxa pode estar levemente subestimada.",
    "A mortalidade considera apenas casos com desfecho registrado; os demais ainda podem evoluir.",
    "A proporção em UTI indica gravidade entre os internados; o conjunto de dados não informa o número de leitos.",
    "A vacinação é a proporção entre os casos de SRAG, não a cobertura vacinal da população.",
    "As notícias são manchetes de fontes públicas e não substituem boletins oficiais.",
]


@dataclass(frozen=True)
class ReportMetrics:
    growth: metrics.CaseGrowth
    growth_short: metrics.CaseGrowth
    mortality: metrics.Mortality
    icu: metrics.IcuShare
    vaccination: metrics.VaccinationShare


def collect_metrics(cases: pd.DataFrame, today: pd.Timestamp) -> ReportMetrics:
    return ReportMetrics(
        growth=metrics.case_growth(cases, today, metrics.LONG_WINDOW_DAYS),
        growth_short=metrics.case_growth(cases, today, metrics.SHORT_WINDOW_DAYS),
        mortality=metrics.mortality(cases, today),
        icu=metrics.icu_share(cases, today),
        vaccination=metrics.vaccination_share(cases, today),
    )


def metrics_for_prompt(report_metrics: ReportMetrics) -> str:
    m = report_metrics
    return "\n".join([
        f"- Dados consolidados até {m.growth.cutoff:%d/%m/%Y}.",
        f"- Casos nos últimos {m.growth.window_days} dias: {_int(m.growth.current)} contra "
        f"{_int(m.growth.previous)} nos {m.growth.window_days} dias anteriores ({_pct(m.growth.change_pct, signed=True)}).",
        f"- Casos nos últimos {m.growth_short.window_days} dias: {_pct(m.growth_short.change_pct, signed=True)} "
        "em relação aos 7 dias anteriores.",
        f"- Mortalidade entre casos com desfecho: {_pct(m.mortality.rate_pct)} "
        f"({_int(m.mortality.deaths)} óbitos em {_int(m.mortality.closed_cases)} casos); "
        f"{_pct(m.mortality.pending_pct)} dos casos do período ainda sem desfecho.",
        f"- Proporção de internados em UTI: {_pct(m.icu.share_pct)}.",
        f"- Casos com vacina de gripe registrada: {_pct(m.vaccination.flu_pct)}; "
        f"com vacina de COVID-19 registrada: {_pct(m.vaccination.covid_pct)}.",
    ])


def fallback_explanation(report_metrics: ReportMetrics) -> str:
    """Deterministic text used when the LLM explanation is rejected by the guardrails."""
    m = report_metrics
    if m.growth.change_pct is None:
        change = f"sem base de comparação com os {m.growth.window_days} dias anteriores ({_int(m.growth.previous)} casos)"
    else:
        direction = "alta" if m.growth.change_pct > 0 else "queda" if m.growth.change_pct < 0 else "estabilidade"
        change = (
            f"uma {direction} de {_pct(abs(m.growth.change_pct))} em relação aos "
            f"{m.growth.window_days} dias anteriores ({_int(m.growth.previous)} casos)"
        )
    return (
        f"Nos últimos {m.growth.window_days} dias com dados consolidados (até {m.growth.cutoff:%d/%m/%Y}), "
        f"foram registrados {_int(m.growth.current)} casos de SRAG, {change}. "
        f"Entre os casos com desfecho registrado, a mortalidade foi de {_pct(m.mortality.rate_pct)}, mas "
        f"{_pct(m.mortality.pending_pct)} dos casos do período ainda não têm desfecho, então esse valor é provisório. "
        f"{_pct(m.icu.share_pct)} dos internados estiveram em UTI. "
        "Os casos mais recentes ainda estão sendo notificados."
    )


def render_markdown(
    today: pd.Timestamp,
    report_metrics: ReportMetrics,
    news: NewsResult | None,
    explanation: str,
    generated_by_llm: bool = True,
) -> str:
    m = report_metrics
    window = f"{m.growth.window_days} dias até {m.growth.cutoff:%d/%m/%Y}"
    lines = [
        f"# Relatório de SRAG — {today:%d/%m/%Y}",
        "",
        f"Dados consolidados até {m.growth.cutoff:%d/%m/%Y} (fonte: Open DATASUS, SIVEP-Gripe).",
        "",
        "## Métricas",
        "",
        "| Métrica | Valor | Detalhe |",
        "|---|---|---|",
        f"| Taxa de aumento de casos ({m.growth.window_days} dias) | {_pct(m.growth.change_pct, signed=True)} | "
        f"{_int(m.growth.current)} vs {_int(m.growth.previous)} casos nos {m.growth.window_days} dias anteriores |",
        f"| Taxa de aumento de casos ({m.growth_short.window_days} dias) | {_pct(m.growth_short.change_pct, signed=True)} | "
        f"{_int(m.growth_short.current)} vs {_int(m.growth_short.previous)} casos |",
        f"| Taxa de mortalidade ({window}) | {_pct(m.mortality.rate_pct)} | "
        f"{_int(m.mortality.deaths)} óbitos em {_int(m.mortality.closed_cases)} casos com desfecho; "
        f"{_pct(m.mortality.pending_pct)} do período ainda sem desfecho |",
        f"| Internados em UTI ({window}) | {_pct(m.icu.share_pct)} | "
        f"{_int(m.icu.icu)} de {_int(m.icu.hospitalized)} internados (indicador de gravidade, não de leitos) |",
        f"| Vacinação registrada nos casos, gripe ({window}) | {_pct(m.vaccination.flu_pct)} | "
        "entre os casos que informaram, não é cobertura da população |",
        f"| Vacinação registrada nos casos, COVID-19 ({window}) | {_pct(m.vaccination.covid_pct)} | "
        "entre os casos que informaram, não é cobertura da população |",
        "",
        "## Gráficos",
        "",
        f"![Casos por dia]({DAILY_CHART_FILE})",
        "",
        f"![Casos por mês]({MONTHLY_CHART_FILE})",
        "",
        "## Cenário",
        "",
        _explanation_notice(generated_by_llm),
        "",
        explanation.strip(),
        "",
        "## Notícias consideradas",
        "",
        *_news_lines(news),
        "",
        "## Limitações",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
    ]
    return "\n".join(lines)


def _explanation_notice(generated_by_llm: bool) -> str:
    if generated_by_llm:
        return "_Texto gerado por IA a partir das métricas e manchetes acima; revise antes de usar._"
    return (
        "_Texto gerado automaticamente por regras, sem IA: a explicação do modelo não passou nas "
        "verificações de segurança desta execução._"
    )


def _news_lines(news: NewsResult | None) -> list[str]:
    if news is None:
        return ["Notícias indisponíveis nesta execução."]
    if not news.items:
        return ["Nenhuma notícia recente encontrada."]
    return [
        f"- {item.published:%d/%m/%Y} — {_escape_markdown(item.source)}: "
        f"[{_escape_markdown(item.title)}]({_safe_link(item.link)})"
        for item in news.items
    ]


def _escape_markdown(text: str) -> str:
    # Headlines are untrusted: brackets would let a title open its own link.
    return text.replace("[", "\\[").replace("]", "\\]")


def _safe_link(link: str) -> str:
    # Whitespace or parentheses in a feed URL would end the markdown link early.
    return link.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/d"
    text = f"{value:+.1f}" if signed else f"{value:.1f}"
    return text.replace(".", ",") + "%"
