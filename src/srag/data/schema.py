"""Data contract for the SRAG cases table: what may be loaded and what each code means."""

DATE_COLUMNS = ["DT_SIN_PRI"]
CATEGORICAL_COLUMNS = ["EVOLUCAO", "HOSPITAL", "UTI", "VACINA", "VACINA_COV"]

# Allowlist: only these columns are ever read from the raw CSV (data minimization).
ALLOWED_COLUMNS = DATE_COLUMNS + CATEGORICAL_COLUMNS

# Valid codes per categorical column; anything else is treated as missing.
VALID_CODES = {
    "EVOLUCAO": {"1", "2", "3", "9"},  # 1=cura 2=óbito SRAG 3=óbito outras causas 9=ignorado
    "HOSPITAL": {"1", "2", "9"},
    "UTI": {"1", "2", "9"},
    "VACINA": {"1", "2", "9"},  # gripe
    "VACINA_COV": {"1", "2", "9"},  # COVID-19
}

# Symptom dates before this are typos (the files start at epidemiological week 1 of 2025).
EARLIEST_VALID_DATE = "2024-12-01"
