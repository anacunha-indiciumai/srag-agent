"""Report charts: daily cases (last 30 days) and monthly cases (last 12 months).

Bars still inside the notification-delay window (see metrics.MATURATION_DAYS) are
drawn lighter and hatched, so a drop caused by late notifications is not read as
a real drop. Hatching (not color alone) keeps the distinction readable without color.
"""
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from srag.tools.metrics import MATURATION_DAYS, cutoff_date

DAILY_DAYS = 30
MONTHLY_MONTHS = 12

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
SERIES_INCOMPLETE = "#86b6ef"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

MONTH_ABBREVIATIONS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def daily_series(cases: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Cases per symptom day for the last DAILY_DAYS days; `complete` is False inside the delay window."""
    days = pd.date_range(today - pd.Timedelta(days=DAILY_DAYS - 1), today)
    counts = cases.groupby("DT_SIN_PRI").size().reindex(days, fill_value=0)
    return pd.DataFrame({"period": days, "cases": counts.to_numpy(), "complete": days <= cutoff_date(today)})


def monthly_series(cases: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Cases per symptom month for the last MONTHLY_MONTHS months; `complete` is False if the month
    ends after the cutoff."""
    months = pd.period_range(end=today.to_period("M"), periods=MONTHLY_MONTHS, freq="M")
    counts = cases.groupby(cases["DT_SIN_PRI"].dt.to_period("M")).size().reindex(months, fill_value=0)
    complete = [month.end_time.normalize() <= cutoff_date(today) for month in months]
    return pd.DataFrame({"period": months, "cases": counts.to_numpy(), "complete": complete})


def daily_cases_chart(cases: pd.DataFrame, today: pd.Timestamp, out_path: Path) -> Path:
    series = daily_series(cases, today)
    labels = [period.strftime("%d/%m") for period in series["period"]]
    title = f"Casos de SRAG por dia — últimos {DAILY_DAYS} dias"
    return _render(series, labels, title, today, out_path, tick_every=7)


def monthly_cases_chart(cases: pd.DataFrame, today: pd.Timestamp, out_path: Path) -> Path:
    series = monthly_series(cases, today)
    labels = [f"{MONTH_ABBREVIATIONS[period.month - 1]}/{str(period.year)[2:]}" for period in series["period"]]
    title = f"Casos de SRAG por mês — últimos {MONTHLY_MONTHS} meses"
    return _render(series, labels, title, today, out_path, tick_every=1)


def _render(series: pd.DataFrame, labels: list[str], title: str, today: pd.Timestamp, out_path: Path, tick_every: int) -> Path:
    positions = list(range(len(series)))
    colors = [SERIES if complete else SERIES_INCOMPLETE for complete in series["complete"]]
    hatches = ["" if complete else "////" for complete in series["complete"]]

    with mpl.rc_context({"hatch.linewidth": 0.8, "font.family": "DejaVu Sans"}):
        figure = Figure(figsize=(9, 4.4), facecolor=SURFACE)
        axes = figure.subplots()
        axes.set_facecolor(SURFACE)
        axes.bar(positions, series["cases"], width=0.78, color=colors, hatch=hatches, edgecolor=SERIES, linewidth=0)

        axes.set_axisbelow(True)
        axes.yaxis.grid(True, color=GRID, linewidth=0.8)
        axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", ".")))
        axes.set_xticks(positions[::tick_every], [labels[i] for i in positions[::tick_every]])
        axes.tick_params(axis="both", colors=INK_SECONDARY, length=0, labelsize=9)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(BASELINE)

        figure.text(0.07, 0.95, title, fontsize=13, fontweight="bold", color=INK_PRIMARY, ha="left", va="top")
        cutoff = cutoff_date(today).strftime("%d/%m/%Y")
        subtitle = f"Data dos primeiros sintomas. Dados consolidados até {cutoff} (corte de {MATURATION_DAYS} dias)."
        figure.text(0.07, 0.895, subtitle, fontsize=9, color=INK_SECONDARY, ha="left", va="top")

        key = [
            Patch(facecolor=SERIES, label="Consolidado"),
            Patch(facecolor=SERIES_INCOMPLETE, edgecolor=SERIES, hatch="////", linewidth=0, label="Incompleto (notificação em andamento)"),
        ]
        axes.legend(
            handles=key, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9,
            labelcolor=INK_SECONDARY, borderaxespad=0,
        )

        figure.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=150, facecolor=SURFACE)
    return out_path
