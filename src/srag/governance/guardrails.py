"""Guardrails enforced in code, not in prompts.

Input: only allowlisted columns may reach the pipeline.
Output: the LLM explanation is checked before it can enter the report. Numbers must come from the
data the model was given, and links, personal data, forecasts and medical advice are rejected.
"""
import re
from dataclasses import dataclass, field

import pandas as pd

from srag.data.schema import ALLOWED_COLUMNS
from srag.tools import metrics
from srag.tools.news import looks_like_injection

MAX_WORDS = 250
SMALL_INTEGER_LIMIT = 10  # list markers and counts like "dois" need no source; percentages always do
KNOWN_CONSTANTS = {metrics.MATURATION_DAYS, metrics.LONG_WINDOW_DAYS, metrics.SHORT_WINDOW_DAYS}

_NUMBER = re.compile(r"(\d+(?:[.,]\d+)*)\s*(%)?")
_THOUSANDS = re.compile(r"\d{1,3}(\.\d{3})+")
_SIGNED_PERCENT = re.compile(r"([+-])(\d+(?:,\d+)?)%")
_UP_WORDS = re.compile(r"\b(alta|aument\w*|cresc\w*|sub(?:iu|ir|e|iram)|elev\w*)\b", re.IGNORECASE)
_DOWN_WORDS = re.compile(r"\b(queda|redu\w*|diminu\w*|ca(?:iu|iram|i|ir)|recuo|declín\w*|declin\w*)\b", re.IGNORECASE)
_URL = re.compile(
    r"https?://|www\."
    r"|(?<![\w.-])[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|br|org|gov|net|edu|io)(?![\w-])",  # bare domain
    re.IGNORECASE,
)
# Person names are deliberately not detected: a name regex flags too many ordinary words, and the
# LLM only receives aggregates and sanitized headlines, so this is a backstop for identifiers only.
# Phone numbers need a DDD/+55 prefix or a 9-digit local part: a bare "2024-2025" is a season, not a phone.
_PERSONAL_DATA = re.compile(
    "|".join([
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",  # CPF
        r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",  # CNPJ
        r"(?<!\d)\d{10,14}(?!\d)",  # unformatted CPF, CNPJ or phone; counts always carry separators
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        r"(?:\+55[\s-]?)?\(\d{2}\)\s?9?\d{4}[-\s]?\d{4}(?!\d)",
        r"\+55[\s-]?\d{2}[\s-]?9?\d{4}[-\s]?\d{4}(?!\d)",
        r"(?<![\d.,/-])\d{2}[\s-]9\d{4}[-\s]?\d{4}(?!\d)",
        r"(?<![\d.,/-])9\d{4}-\d{4}(?!\d)",
        r"\b(?:tel\w*|fone|cel\w*|whats\w*|contato)\W{0,3}\d{4}-?\d{4}(?!\d)",
    ]),
    re.IGNORECASE,
)
_ADVICE_OR_FORECAST = re.compile(
    r"\b(você deve|voce deve|procure (um )?m[eé]dico"
    r"|(vai|v[aã]o|ir[aá]|ir[aã]o) (aumentar|cair|piorar|diminuir))\b",
    re.IGNORECASE,
)


class GuardrailViolation(ValueError):
    pass


@dataclass
class GuardrailResult:
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def check_allowed_columns(cases: pd.DataFrame) -> None:
    extra = sorted(set(cases.columns) - set(ALLOWED_COLUMNS))
    if extra:
        raise GuardrailViolation(f"Columns outside the allowlist reached the pipeline: {extra}")


def check_explanation(text: str, source_text: str) -> GuardrailResult:
    """Validate the LLM explanation. `source_text` is everything the model was given."""
    violations = []
    if not text.strip():
        violations.append("texto vazio")
    if len(text.split()) > MAX_WORDS:
        violations.append(f"mais de {MAX_WORDS} palavras")
    if _URL.search(text):
        violations.append("contém link (links só podem vir da lista de notícias gerada pelo código)")
    if _PERSONAL_DATA.search(text):
        violations.append("contém dado pessoal (CPF, CNPJ, telefone ou e-mail)")
    if looks_like_injection(text):
        violations.append("contém padrão de instrução suspeito")
    if _ADVICE_OR_FORECAST.search(text):
        violations.append("contém conselho médico ou previsão")
    violations += [f"número sem origem nos dados: {raw}" for raw in _ungrounded_numbers(text, source_text)]
    violations += [f"direção contrária ao dado: {raw}" for raw in _wrong_direction(text, source_text)]
    return GuardrailResult(violations)


def _wrong_direction(text: str, source_text: str) -> list[str]:
    """Flag a sentence that says "alta" about a percentage the data shows as negative, or the reverse.

    Only magnitudes that the source shows with a single sign are checked, and a sentence that uses
    words of both directions is left alone: the aim is to catch an inverted trend, not to parse prose.
    """
    signs: dict[float, set[str]] = {}
    for sign, magnitude in _SIGNED_PERCENT.findall(source_text):
        signs.setdefault(float(magnitude.replace(",", ".")), set()).add(sign)
    wrong = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        goes_up, goes_down = bool(_UP_WORDS.search(sentence)), bool(_DOWN_WORDS.search(sentence))
        if goes_up == goes_down:
            continue
        for value, is_percent in _numbers(sentence):
            shown = signs.get(value, set()) if is_percent else set()
            if (goes_up and shown == {"-"}) or (goes_down and shown == {"+"}):
                wrong.append(f"{value:g}%")
    return wrong


def _ungrounded_numbers(text: str, source_text: str) -> list[str]:
    allowed = {value for value, _ in _numbers(source_text)} | KNOWN_CONSTANTS
    ungrounded = []
    for value, is_percent in _numbers(text):
        small_integer = value.is_integer() and value <= SMALL_INTEGER_LIMIT and not is_percent
        if value not in allowed and not small_integer:
            ungrounded.append(f"{value:g}{'%' if is_percent else ''}")
    return ungrounded


def _numbers(text: str) -> list[tuple[float, bool]]:
    found = []
    for raw, percent in _NUMBER.findall(text):
        if _THOUSANDS.fullmatch(raw):
            raw = raw.replace(".", "")
        try:
            found.append((float(raw.replace(",", ".")), bool(percent)))
        except ValueError:
            continue  # not a number (e.g. a version string like 1.2.3)
    return found
