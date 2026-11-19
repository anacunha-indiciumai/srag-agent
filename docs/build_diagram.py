"""Builds the conceptual architecture diagram: docs/arquitetura.pdf (+ .png preview).

Run: python docs/build_diagram.py
"""
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"

# kind -> (fill, edge). Color encodes the type of component, repeated in the legend.
STYLES = {
    "data": ("#f0efec", "#898781"),
    "orchestrator": ("#e6f0fc", "#2a78d6"),
    "tool": ("#f6faff", "#2a78d6"),
    "llm": ("#fdeee8", "#eb6834"),
    "governance": ("#e3f5ec", "#1baf7a"),
}
LEGEND = [
    ("data", "Dados e fontes externas"),
    ("orchestrator", "Orquestração e tools"),
    ("llm", "LLM"),
    ("governance", "Governança"),
]

TITLE_OFFSET = 4.0
LINE_HEIGHT = 1.7


def box(ax, x, y, w, h, title, lines=(), kind="tool"):
    fill, edge = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    ax.text(x + 1.6, y + h - 1.6, title, fontsize=9.5, fontweight="bold", color=INK, va="top", ha="left")
    for i, line in enumerate(lines):
        ax.text(x + 1.6, y + h - TITLE_OFFSET - 1.2 - i * LINE_HEIGHT, line,
                fontsize=7.8, color=INK_SECONDARY, va="top", ha="left")


def arrow(ax, start, end, both=False, scale=11):
    style = "<|-|>" if both else "-|>"
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle=style, color=INK_SECONDARY, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=scale))


def build() -> Figure:
    figure = Figure(figsize=(16, 9), facecolor=SURFACE)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(3, 87.5, "Arquitetura da solução — agente de relatório sobre SRAG", fontsize=16,
            fontweight="bold", color=INK, va="center")
    ax.text(3, 83.3, "O código calcula números, gráficos e links; o LLM só redige o parágrafo de explicação, "
            "que passa por guardrails antes de entrar no relatório.", fontsize=9.5, color=INK_SECONDARY, va="center")
    for (kind, label), x in zip(LEGEND, [80, 98, 114, 124]):
        fill, edge = STYLES[kind]
        ax.add_patch(Rectangle((x, 86.6), 1.8, 1.8, facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + 2.6, 87.5, label, fontsize=8, color=INK_SECONDARY, va="center")

    # Orchestrator band with its five fixed steps
    box(ax, 36, 64, 88, 15, "Orquestrador (LangGraph) — ordem fixa, uma execução por relatório", kind="orchestrator")
    steps = ["① Métricas", "② Gráficos", "③ Notícias ↻", "④ Explicação", "⑤ Montagem"]
    ax.text(38, 74.2, "↻ aresta condicional: repete as notícias 1 vez se todos os feeds falharem por erro de rede",
            fontsize=7.8, color=INK_SECONDARY, va="center")
    for i, label in enumerate(steps):
        x = 38 + i * 17.2
        edge = STYLES["llm"][1] if i == 3 else STYLES["orchestrator"][1]
        ax.add_patch(FancyBboxPatch((x, 66), 15.2, 6, boxstyle="round,pad=0,rounding_size=1",
                                    facecolor="white", edgecolor=edge, linewidth=1.3))
        ax.text(x + 7.6, 69, label, fontsize=8.5, color=INK, ha="center", va="center", fontweight="bold")
        if i < len(steps) - 1:
            arrow(ax, (x + 15.2, 69), (x + 17.2, 69), scale=7)

    # Tools
    box(ax, 36, 40, 28, 18, "① Tool de métricas", [
        "tools/metrics.py",
        "• aumento de casos (7 e 28 dias)",
        "• mortalidade (casos com desfecho)",
        "• internados em UTI",
        "• vacinação registrada nos casos"])
    box(ax, 66, 40, 28, 18, "② Tool de gráficos", [
        "tools/charts.py",
        "• casos por dia (últimos 30 dias)",
        "• casos por mês (últimos 12 meses)",
        "• hachura = dados ainda incompletos"])
    box(ax, 96, 40, 28, 18, "③ Tool de notícias", [
        "tools/news.py",
        "• só hosts da allowlist (2 feeds)",
        "• sanitiza e limita o texto",
        "• descarta tentativas de injeção",
        "• links apenas http/https"])
    for x in (50, 80, 110):
        arrow(ax, (x, 64), (x, 58), both=True)

    # Data layer and external sources
    box(ax, 3, 20, 28, 14, "Banco de dados (CSV/DataFrame)", [
        "Open DATASUS · SRAG 2025 + 2026",
        "585 mil casos · 194 colunas",
        "em memória, sem SGBD (PoC)"], kind="data")
    box(ax, 36, 20, 58, 14, "Camada de dados (data/loader.py, schema.py)", [
        "• lê só 6 das 194 colunas (allowlist)",
        "• colunas sensíveis nunca entram na memória",
        "• valida datas e códigos; inválido vira vazio, sem chute",
        "• 2 regras fixas: UTI=sim ⇒ internado=sim; códigos",
        "  como ‘6.0000000000’ são normalizados"])
    box(ax, 96, 20, 28, 14, "Fontes de notícias", [
        "Google News (RSS) · Agência Brasil",
        "manchetes dos últimos 14 dias",
        "dado externo não confiável"], kind="data")
    arrow(ax, (31, 27), (36, 27))
    arrow(ax, (50, 34), (50, 40))
    arrow(ax, (80, 34), (80, 40))
    arrow(ax, (110, 34), (110, 40))

    # Output
    box(ax, 3, 64, 28, 15, "Saída", [
        "relatorio_AAAA-MM-DD.md",
        "+ 2 gráficos (PNG)",
        "tabela e links: código",
        "explicação: LLM, marcada como IA"])
    arrow(ax, (36, 71), (31, 71))

    # LLM and output guardrails
    box(ax, 132, 64, 26, 16, "LLM (por configuração)", [
        "• Claude (API da Anthropic)",
        "• Ollama (modelo local)",
        "• AI Gateway da empresa",
        "Só redige o texto ‘Cenário’.",
        "Não calcula nem chama tools."], kind="llm")
    arrow(ax, (124, 72), (132, 72), both=True)
    ax.text(128, 74, "prompt/texto", fontsize=6.8, color=INK_SECONDARY, ha="center", va="bottom")
    box(ax, 132, 40, 26, 18, "Guardrails de saída", [
        "• números com origem nos dados",
        "• sem links, PII ou previsões",
        "• 2 tentativas; depois, texto",
        "  de reserva gerado por regras",
        "• falha temporária do LLM: texto do",
        "  código, marcado ‘sem IA’",
        "• erro de configuração: execução para"], kind="governance")
    arrow(ax, (145, 64), (145, 58))
    # approved text (or fallback) goes to the report
    ax.plot([138, 138, 17], [58, 61, 61], color=INK_SECONDARY, lw=1.3, linestyle=(0, (4, 2)))
    arrow(ax, (17, 61), (17, 64))
    ax.text(53, 61.4, "texto aprovado (ou reserva) → relatório", fontsize=6.8, color=INK_SECONDARY,
            ha="left", va="bottom")
    ax.plot([132, 128, 128], [49, 49, 67], color=INK_SECONDARY, lw=1.3)
    arrow(ax, (128, 67), (124, 67))

    # Audit bar
    box(ax, 3, 3, 155, 12, "Governança e auditoria (todas as etapas)", [
        "• logs/audit.jsonl (append-only, com cadeia de hashes): cada etapa, bloqueio e decisão (notícias indisponíveis, texto de reserva), "
        "com run_id, duração e hash do resultado; registra o hash dos arquivos de entrada",
        "• guardrail de entrada: recusa coluna fora da allowlist e dados defasados (menos de 10 casos nos últimos 21 dias) · "
        "o log guarda resumos e hashes, nunca linhas de pacientes"], kind="governance")
    return figure


def main() -> None:
    figure = build()
    figure.savefig(OUT_DIR / "arquitetura.pdf", facecolor=SURFACE)
    figure.savefig(OUT_DIR / "arquitetura.png", dpi=110, facecolor=SURFACE)
    print(f"Diagram written to {OUT_DIR / 'arquitetura.pdf'}")


if __name__ == "__main__":
    main()
