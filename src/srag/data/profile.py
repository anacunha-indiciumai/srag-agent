"""Column profiling of the raw SRAG CSVs (EDA step 1).

Reads the files in chunks and writes an aggregated markdown report. Row-level
values are never printed: value distributions are shown only for low-cardinality
columns that are not on the sensitive list.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CHUNK_SIZE = 100_000
MAX_TRACKED_VALUES = 30  # beyond this a column is treated as high-cardinality
MAX_SHOWN_VALUES = 12

# Columns that can identify a person, a facility, or hold free text / lot numbers.
SENSITIVE_PREFIXES = ("NU_", "DT_NASC", "CO_", "ID_", "NM_", "PAC_", "LOTE_", "LOT_", "DS_", "OUT_")
SENSITIVE_COLUMNS = {"MORB_DESC", "OUTRO_DES", "RAIOX_OUT", "TOMO_OUT", "FLUASU_OUT", "FLUBLI_OUT",
                     "SG_UF", "SG_UF_NOT", "SG_UF_INTE", "CS_RACA", "CS_ETINIA", "CS_ESCOL_N", "TEM_CPF",
                     "CS_SEXO", "CS_GESTANT", "TP_IDADE", "COD_IDADE"}


def is_sensitive(column: str) -> bool:
    return column in SENSITIVE_COLUMNS or column.startswith(SENSITIVE_PREFIXES)


def profile(paths: list[Path]) -> tuple[pd.DataFrame, int]:
    total_rows = 0
    nulls: Counter = Counter()
    values: dict[str, Counter] = {}
    overflowed: set[str] = set()

    for path in paths:
        for chunk in pd.read_csv(path, sep=";", dtype=str, encoding="latin-1", chunksize=CHUNK_SIZE):
            total_rows += len(chunk)
            for column in chunk.columns:
                nulls[column] += int(chunk[column].isna().sum())
                if column in overflowed:
                    continue
                counts = values.setdefault(column, Counter())
                counts.update(chunk[column].dropna().value_counts().to_dict())
                if len(counts) > MAX_TRACKED_VALUES:
                    overflowed.add(column)
                    values[column] = Counter()

    rows = []
    for column in nulls:
        distribution = ""
        if column not in overflowed and not is_sensitive(column):
            top = values[column].most_common(MAX_SHOWN_VALUES)
            distribution = ", ".join(f"{v}: {n / (total_rows - nulls[column]) * 100:.1f}%" for v, n in top)
        rows.append({
            "coluna": column,
            "% vazio": round(nulls[column] / total_rows * 100, 1),
            "cardinalidade": f">{MAX_TRACKED_VALUES}" if column in overflowed else len(values[column]),
            "sensível": "sim" if is_sensitive(column) else "",
            "distribuição (só baixa cardinalidade, não sensível)": distribution,
        })
    return pd.DataFrame(rows).sort_values("% vazio").reset_index(drop=True), total_rows


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    paths = [Path(p) for p in sys.argv[2:]]
    report, total_rows = profile(paths)
    header = f"# Perfil das colunas — SRAG\n\nArquivos: {', '.join(p.name for p in paths)}  \nLinhas: {total_rows:,}\n\n"
    Path(sys.argv[1]).write_text(header + to_markdown(report), encoding="utf-8")
    print(f"{len(report)} colunas perfiladas, {total_rows:,} linhas -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
