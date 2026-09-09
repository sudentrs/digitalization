"""Clean and validate the one-page azioni structured parse.

This converts the raw row/column OCR draft into a more dataset-like review
file. It keeps raw fields and emits flags instead of hiding uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = Path("output/structured_one_page/candidate_security_rows_one_page.csv")
DEFAULT_OUTPUT = Path("output/cleaned_one_page")

MONTHS = {
    "genn": 1,
    "gen": 1,
    "feb": 2,
    "febb": 2,
    "mar": 3,
    "marzo": 3,
    "apr": 4,
    "aprile": 4,
    "mag": 5,
    "maggio": 5,
    "giu": 6,
    "giugno": 6,
    "lug": 7,
    "luglio": 7,
    "ago": 8,
    "agosto": 8,
    "sett": 9,
    "set": 9,
    "ott": 10,
    "ottob": 10,
    "ottobre": 10,
    "nov": 11,
    "dic": 12,
    "dicembre": 12,
}

# Number tokens in these scans are usually either:
# - thousands with separators: 1.720.000, 1 720.000, 5.000 000
# - plain integers/prices: 500, 98,50
# This intentionally does not swallow arbitrary spaces between unrelated values.
NUMBER_RE = re.compile(
    r"(?<!\d)("
    r"\d{1,3}(?:[.,]\d{3})+(?:,\d+)?"
    r"|\d{1,3} \d{3}(?:[.,]\d{3})+(?:,\d+)?"
    r"|\d{1,3} \d{3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{4,}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?: \d{3})+(?:,\d+)?"
    r"|\d+(?:,\d+)?"
    r")(?:\s*[—-]+)?"
)
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2,4})(?!\d)")

ISSUER_STOP_WORDS = {
    "genn",
    "gen",
    "feb",
    "febb",
    "mar",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "ottob",
    "nov",
    "dicembre",
    "stamp",
    "ass",
    "gr",
    "ex",
    "opt",
}

SECTION_HEADINGS = {
    "finanziari",
    "assicurativi",
    "trasporti",
    "tessili",
    "manifatturieri",
    "minerari",
    "metallurgici",
    "meccanici",
    "automobilistici",
    "elettrici",
    "elettrotecnici",
    "alimentari",
    "allmentari",
    "chimici",
    "immobiliari",
    "agricoli",
    "diversi",
}

SECTION_HEADING_PHRASES = {
    "finanziari",
    "assicurativi",
    "trasporti",
    "tessili e manifatturieri",
    "minerari e metallurgici",
    "meccanici ed automobilistici",
    "meocanioi ed automobilistio",
    "elettrici ed elettrotecnici",
    "alimentari",
    "chimici",
    "immobiliari ed agricoli",
    "diversi",
}

COMMON_NOMINAL_VALUES = {
    20,
    25,
    35,
    40,
    50,
    75,
    100,
    125,
    150,
    180,
    190,
    200,
    250,
    300,
    350,
    400,
    500,
    600,
    650,
    750,
    1000,
    1250,
    1350,
    1500,
    2000,
    2500,
    2750,
    3000,
    4000,
    5000,
    6000,
}

ISSUER_IDENTITY_REFERENCE = [
    ("sviluppo imprese industriali", "Sviluppo Imprese Industriali", "1400000", "2800000", "500"),
    ("sviluppo imprese", "Sviluppo Imprese Industriali", "1400000", "2800000", "500"),
    ("imprese industriali", "Sviluppo Imprese Industriali", "1400000", "2800000", "500"),
    ("stab minerario del siele", "Stab. Minerario del Siele", "164736", "1647360", "100"),
    ("minerario del siele", "Stab. Minerario del Siele", "164736", "1647360", "100"),
    ("minerario de siele", "Stab. Minerario del Siele", "164736", "1647360", "100"),
    ("siele", "Stab. Minerario del Siele", "164736", "1647360", "100"),
    ("sie le", "Stab. Minerario del Siele", "164736", "1647360", "100"),
    ("westinghouse", "Westinghouse", "420000", "1200000", "350"),
    ("franco tosi", "Franco Tosi", "420000", "1200000", "350"),
    ("riunione adriatica di sicurta", "Riunione Adriatica di Sicurtà", "2400000", "1920000", "1250"),
    ("riunione adriatica", "Riunione Adriatica di Sicurtà", "2400000", "1920000", "1250"),
    ("riunione adriatiea", "Riunione Adriatica di Sicurtà", "2400000", "1920000", "1250"),
    ("riunione adriatiga", "Riunione Adriatica di Sicurtà", "2400000", "1920000", "1250"),
    ("lanificio e feltrificio scotti", "Lanificio e Feltrificio Scotti e C.", "60000", "1500000", "40"),
    ("scotti", "Lanificio e Feltrificio Scotti e C.", "60000", "1500000", "40"),
    ("pignone ordinarie", "Pignone (ordinarie)", "450000", "3000000", "150"),
    ("pignone ordinay", "Pignone (ordinarie)", "450000", "3000000", "150"),
    ("pignone ordin", "Pignone (ordinarie)", "450000", "3000000", "150"),
    ("pignone privilegiate", "Pignone (privilegiate)", "400000", "2000000", "200"),
    ("pignone pravilegiate", "Pignone (privilegiate)", "400000", "2000000", "200"),
    ("privilegiate", "Pignone (privilegiate)", "400000", "2000000", "200"),
    ("pravilegiate", "Pignone (privilegiate)", "400000", "2000000", "200"),
    ("terme demaniali di acqui", "Terme Demaniali di Acqui", "6750", "90000", "75"),
    ("larderello", "Larderello", "", "", "100"),
    ("motta", "Motta S. p. A.", "", "", "2000"),
    ("mira lanza", "Mira Lanza", "810000", "600000", "1350"),
    ("mira langa", "Mira Lanza", "810000", "600000", "1350"),
    ("rumianea", "Rumianea", "3000000", "60000000", "50"),
    ("silos genova", "Silos Genova", "36000", "180000", "200"),
    ("metallurgica italiana", "Metallurgica Italiana", "30000000", "60000000", "500"),
    ("terni", "Terni (Soc. p. l'Ind. e l'Elettr.)", "10500000", "52500000", "200"),
    ("ceramica richard ginori", "Ceramica Richard-Ginori", "810000", "3240000", "250"),
    ("ceramica richard", "Ceramica Richard-Ginori", "810000", "3240000", "250"),
    ("veneta", "Veneta", "24000", "120000", "200"),
    ("whitehead moto fides", "Whitehead Moto Fides", "1250000", "5000000", "250"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_ocr_text(text: str) -> str:
    text = text or ""
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u0427": "-",
        "\u0421": "C",
        "\u0422": "T",
        "\u0410": "A",
        "\u043b": "l",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    text = normalize_ocr_text(text)
    text = re.sub(r"[{}|\\\[\]]", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lookup_text(text: str) -> str:
    text = normalize_ocr_text(text or "").lower()
    replacements = {
        "à": "a",
        "è": "e",
        "é": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalize_lookup_text(text).split() if len(token) > 1}


def issuer_reference_match(issuer: str, raw_text: str) -> tuple[str, str, str, str, str] | None:
    haystack = normalize_lookup_text(f"{issuer} {raw_text}")
    haystack_tokens = token_set(haystack)
    best: tuple[float, tuple[str, str, str, str, str]] | None = None
    for key, canonical, capital, shares, nominal in ISSUER_IDENTITY_REFERENCE:
        key_tokens = token_set(key)
        if not key_tokens:
            continue
        phrase_hit = key in haystack
        overlap = len(key_tokens & haystack_tokens) / len(key_tokens)
        if not phrase_hit and overlap < 0.75:
            continue
        if key in {"motta", "veneta", "terni"} and not phrase_hit:
            continue
        score = overlap + (1.0 if phrase_hit else 0.0)
        if best is None or score > best[0]:
            best = (score, (key, canonical, capital, shares, nominal))
    return best[1] if best else None


def rescue_identity_from_reference(
    issuer: str,
    capital: str,
    shares: str,
    nominal: str,
    row: dict[str, str],
) -> tuple[str, str, str, str, list[str]]:
    match = issuer_reference_match(issuer, row.get("raw_row_text", ""))
    if not match:
        return issuer, capital, shares, nominal, []
    _, canonical, ref_capital, ref_shares, ref_nominal = match
    flags: list[str] = []
    if canonical and canonical != issuer:
        issuer = canonical
        flags.append("issuer_reference_rescued")
    duplicated_identity = bool(capital and shares and capital == shares and ref_capital and ref_shares and ref_capital != ref_shares)
    capital_mismatch = bool(ref_capital and capital and is_large_integer(capital) and capital != ref_capital and duplicated_identity)
    shares_mismatch = bool(ref_shares and shares and is_large_integer(shares) and shares != ref_shares and duplicated_identity)
    if ref_capital and (not capital or not is_large_integer(capital) or capital_mismatch):
        capital = ref_capital
        flags.append("identity_reference_capital_rescued")
    if ref_shares and (not shares or not is_large_integer(shares) or shares_mismatch):
        shares = ref_shares
        flags.append("identity_reference_shares_rescued")
    if ref_nominal and (not nominal or not looks_like_nominal(nominal)):
        nominal = ref_nominal
        flags.append("identity_reference_nominal_rescued")
    return issuer, capital, shares, nominal, flags


def extract_number_tokens(text: str) -> list[str]:
    text = normalize_ocr_text(text)
    text = (text or "").replace("€", "E").replace("—", "-")
    tokens = []
    for match in NUMBER_RE.finditer(text):
        token = match.group(1).strip(" .,'")
        if token:
            tokens.append(token)
    return tokens


def parse_number(token: str, *, decimal_allowed: bool) -> str:
    token = normalize_ocr_text(token)
    token = token.strip()
    token = token.replace("'", "").replace(" ", "")
    token = token.replace("€", "").replace("?", "")
    token = re.sub(r"(?<=\d)[CcOo](?=\d)", "0", token)
    token = re.sub(r"(?<=\d)[CcOo]$", "0", token)
    token = token.strip(".,;:-")
    if not token:
        return ""

    if "," in token:
        left, right = token.rsplit(",", 1)
        if decimal_allowed and right.isdigit() and len(right) <= 2:
            left = re.sub(r"\D", "", left)
            return f"{left}.{right}" if left else ""
        return re.sub(r"\D", "", left + right)

    if "." in token:
        parts = token.split(".")
        if decimal_allowed and len(parts[-1]) <= 2 and all(part.isdigit() for part in parts):
            return f"{''.join(parts[:-1])}.{parts[-1]}"
        return re.sub(r"\D", "", token)

    return re.sub(r"\D", "", token)


def first_numeric(
    text: str,
    *,
    decimal_allowed: bool = False,
    prefer_large: bool = False,
    max_value: float | None = None,
) -> tuple[str, list[str]]:
    tokens = extract_number_tokens(text)
    parsed = [parse_number(token, decimal_allowed=decimal_allowed) for token in tokens]
    parsed = [value for value in parsed if value]
    if max_value is not None:
        parsed = [value for value in parsed if float(value) <= max_value]
    if not parsed:
        return "", tokens
    if prefer_large:
        parsed = sorted(parsed, key=lambda value: (len(value.split(".")[0]), float(value)), reverse=True)
    return parsed[0], tokens


def parsed_numbers(text: str, *, decimal_allowed: bool = False) -> list[str]:
    values = []
    for token in extract_number_tokens(text):
        value = parse_number(token, decimal_allowed=decimal_allowed)
        if value:
            values.append(value)
    return values


def parsed_integer_values(text: str) -> list[str]:
    return [value for value in parsed_numbers(text) if value and "." not in value]


def parsed_identity_integer_values(text: str) -> list[str]:
    """Parse identity-column integers, repairing row markers glued to nominals.

    OCR sometimes reads a row number and nominal value as one spaced token, e.g.
    "7 500". For identity parsing that should usually contribute 500, not 7500.
    """
    values: list[str] = []
    for token in extract_number_tokens(text):
        normalized = normalize_ocr_text(token).strip()
        value = parse_number(normalized, decimal_allowed=False)
        if not value:
            continue
        parts = re.split(r"\s+", normalized)
        if len(parts) == 2:
            left = parse_number(parts[0], decimal_allowed=False)
            right = parse_number(parts[1], decimal_allowed=False)
            if (
                left
                and right
                and int(left) <= 30
                and looks_like_nominal(right)
                and not is_large_integer(right)
            ):
                values.append(right)
                continue
        values.append(value)
    return values


def parsed_identity_nominal_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for token in extract_number_tokens(text):
        normalized = normalize_ocr_text(token).strip()
        parts = re.split(r"\s+", normalized)
        if len(parts) == 2:
            left = parse_number(parts[0], decimal_allowed=False)
            right = parse_number(parts[1], decimal_allowed=False)
            if (
                left
                and right
                and int(left) <= 30
                and looks_like_nominal(right)
                and not is_large_integer(right)
            ):
                candidates.append(right)
                continue
        value = parse_number(normalized, decimal_allowed=False)
        if value and looks_like_nominal(value) and not is_large_integer(value):
            candidates.append(value)
    return candidates


def is_large_integer(value: str) -> bool:
    return bool(value) and "." not in value and int(value) >= 10000


def looks_like_nominal(value: str) -> bool:
    if not value or "." in value:
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    if number in COMMON_NOMINAL_VALUES:
        return True
    return 25 <= number <= 20000 and number % 25 == 0


def looks_like_identity_amount(value: str) -> bool:
    if not value or "." in value:
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    return number >= 1000


def suspicious_identity_value(value: str) -> bool:
    if not value or "." in value:
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    return 0 < number <= 30


def numeric_as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return -1.0


def looks_like_quantity(value: str) -> bool:
    if not value or "." in value:
        return False
    try:
        number = int(value)
    except ValueError:
        return False
    return 1 <= number <= 500000


def looks_like_price(value: str) -> bool:
    number = numeric_as_float(value)
    return 0 < number <= 300000


def has_number(text: str, *, decimal_allowed: bool = True) -> bool:
    return bool(parsed_numbers(text, decimal_allowed=decimal_allowed))


def is_blank_or_dash_cell(text: str) -> bool:
    text = normalize_ocr_text(clean_text(text)).lower()
    if not text:
        return True
    text = re.sub(r"\s+", "", text)
    text = text.replace("=", "-").replace("_", "-")
    return bool(text) and not re.search(r"\d|[a-z]", text) and set(text) <= set("-.,;:()?")


def trailing_numeric_values(text: str) -> list[str]:
    values = parsed_numbers(text, decimal_allowed=True)
    return [value for value in values if looks_like_price(value)]


def strip_leading_row_marker(value: str) -> str:
    """Remove OCR-captured row markers before large amount fields."""
    if not value or "." in value:
        return value
    if len(value) >= 9 and value[:2].isdigit() and value[2] in "123456789":
        return value[2:]
    return value


def price_quantity_fallback(price_min_raw: str, price_max_raw: str) -> tuple[str, str]:
    values = parsed_numbers(f"{price_min_raw} {price_max_raw}", decimal_allowed=True)
    values = [value for value in values if looks_like_price(value)]
    if not values:
        return "", ""
    if len(values) >= 2 and looks_like_quantity(values[-1]):
        return values[-2], values[-1]
    return values[-1], ""


def identity_fallback(*texts: str) -> tuple[str, str, str]:
    values: list[str] = []
    for text in texts:
        values.extend(parsed_identity_integer_values(text))
    cleaned: list[str] = []
    for index, value in enumerate(values):
        number = int(value)
        if number <= 30 and index + 1 < len(values) and is_large_integer(values[index + 1]):
            continue
        cleaned.append(value)

    large_indices = [idx for idx, value in enumerate(cleaned) if is_large_integer(value)]
    if len(large_indices) < 2:
        return "", "", ""
    first = large_indices[0]
    second = large_indices[1]
    capital = cleaned[first]
    shares = cleaned[second]
    nominal = ""
    for value in cleaned[second + 1 :]:
        if looks_like_nominal(value):
            nominal = value
            break
    return capital, shares, nominal


def identity_from_column_sequence(*texts: str) -> tuple[str, str, str]:
    values: list[str] = []
    for text in texts:
        values.extend(parsed_integer_values(text))
    filtered = [value for value in values if int(value) > 30]
    large = [value for value in filtered if is_large_integer(value)]
    if len(large) < 2:
        return "", "", ""
    second_large_index = filtered.index(large[1])
    nominal = ""
    for value in filtered[second_large_index + 1 :]:
        if looks_like_nominal(value) and not is_large_integer(value):
            nominal = value
            break
    return large[0], large[1], nominal


def identity_from_raw_row_pattern(text: str) -> tuple[str, str, str]:
    """Recover identity fields from the left-to-right raw row when columns slipped."""
    values = [value for value in parsed_integer_values(text) if int(value) > 30]
    identity_indices = [index for index, value in enumerate(values) if looks_like_identity_amount(value)]
    if len(identity_indices) < 2:
        return "", "", ""
    first = identity_indices[0]
    second = identity_indices[1]
    nominal = ""
    suffix_text = " ".join(values[second + 1 : second + 5])
    nominal_candidates = parsed_identity_nominal_candidates(suffix_text)
    for value in nominal_candidates or values[second + 1 : second + 5]:
        if looks_like_nominal(value) and not looks_like_identity_amount(value):
            nominal = value
            break
    return values[first], values[second], nominal


def text_before_issuer(raw_text: str, issuer: str) -> str:
    raw = normalize_ocr_text(raw_text or "")
    issuer = normalize_ocr_text(clean_text(issuer or ""))
    if not raw or not issuer:
        return raw
    words = [
        word
        for word in re.findall(r"[^\W\d_]{3,}", issuer, flags=re.UNICODE)
        if word.lower() not in ISSUER_STOP_WORDS
    ]
    for word in words[:3]:
        match = re.search(re.escape(word), raw, flags=re.IGNORECASE)
        if match:
            return raw[: match.start()]
    return raw


def split_glued_identity_nominal(value: str) -> tuple[str, str]:
    if not value or "." in value:
        return value, ""
    if len(value) < 7:
        return value, ""
    for suffix_len in (4, 3, 2):
        if len(value) <= suffix_len:
            continue
        prefix = value[:-suffix_len]
        suffix = value[-suffix_len:]
        if looks_like_identity_amount(prefix) and looks_like_nominal(suffix):
            return prefix, suffix
    return value, ""


def nominal_from_glued_identity_text(*texts: str) -> str:
    for text in texts:
        normalized = normalize_ocr_text(text or "")
        for match in re.finditer(r"\d{1,3}[.,]\d{3}(\d{2,4})(?=\D|$)", normalized):
            suffix = parse_number(match.group(1), decimal_allowed=False)
            if suffix and looks_like_nominal(suffix):
                return suffix
    return ""


def nominal_from_row_prefix(row: dict[str, str], issuer: str, capital: str, shares: str) -> str:
    """Recover nominal from the part of a row before the issuer column."""
    glued_nominal = nominal_from_glued_identity_text(
        row.get("shares_outstanding_raw", ""),
        row.get("nominal_value_raw", ""),
        row.get("shares_outstanding_raw_backup_hint", ""),
        row.get("nominal_value_raw_backup_hint", ""),
    )
    if glued_nominal:
        return glued_nominal

    prefix = text_before_issuer(row.get("raw_row_text", ""), issuer or row.get("issuer_name_raw", ""))
    tokens = extract_number_tokens(prefix)
    values = [parse_number(token, decimal_allowed=False) for token in tokens]
    values = [value for value in values if value]
    if not values:
        return ""

    repaired_values: list[str] = []
    for value in values:
        repaired, glued_nominal = split_glued_identity_nominal(value)
        repaired_values.append(repaired)

    if capital and shares and repaired_values:
        first_value = repaired_values[0]
        prefix_after_first = prefix[prefix.find(tokens[0]) + len(tokens[0]) :] if tokens else prefix
        has_month_or_date_context = bool(DATE_RE.search(prefix_after_first)) or any(
            month in prefix_after_first.lower() for month in MONTHS
        )
        if has_month_or_date_context and looks_like_nominal(first_value):
            return first_value

    capital_index = -1
    shares_index = -1
    if capital in repaired_values:
        capital_index = repaired_values.index(capital)
    if shares in repaired_values:
        shares_index = repaired_values.index(shares)
    start_index = max(capital_index, shares_index)
    if start_index < 0:
        large_indices = [index for index, value in enumerate(repaired_values) if is_large_integer(value)]
        if not large_indices:
            return ""
        start_index = large_indices[-1]
    search_values = repaired_values[start_index + 1 :]

    for value in search_values[:6]:
        if looks_like_nominal(value):
            return value
    return ""


def rescue_identity_fields(
    capital: str,
    shares: str,
    nominal: str,
    row: dict[str, str],
) -> tuple[str, str, str, list[str]]:
    flags: list[str] = []
    fallback_capital, fallback_shares, fallback_nominal = identity_from_column_sequence(
        row.get("capital_subscribed_raw", ""),
        row.get("shares_outstanding_raw", ""),
        row.get("nominal_value_raw", ""),
        row.get("capital_subscribed_raw_backup_hint", ""),
        row.get("shares_outstanding_raw_backup_hint", ""),
        row.get("nominal_value_raw_backup_hint", ""),
    )
    raw_capital, raw_shares, raw_nominal = identity_from_raw_row_pattern(row.get("raw_row_text", ""))
    if not fallback_capital and not fallback_shares and raw_capital and raw_shares:
        fallback_capital, fallback_shares, fallback_nominal = raw_capital, raw_shares, raw_nominal
    elif raw_nominal and not fallback_nominal:
        fallback_nominal = raw_nominal
    if fallback_capital and fallback_shares:
        if not capital or suspicious_identity_value(capital):
            capital = fallback_capital
            flags.append("identity_columns_rescued")
        if not shares or suspicious_identity_value(shares):
            shares = fallback_shares
            if "identity_columns_rescued" not in flags:
                flags.append("identity_columns_rescued")
        if fallback_nominal and (not nominal or is_large_integer(nominal) or suspicious_identity_value(nominal)):
            nominal = fallback_nominal
            flags.append("nominal_cell_rescued")

    if not nominal or suspicious_identity_value(nominal):
        raw_prefix_nominal = nominal_from_row_prefix(row, row.get("issuer_name_raw", ""), capital, shares)
        if raw_prefix_nominal:
            nominal = raw_prefix_nominal
            flags.append("nominal_row_prefix_rescued")

    if not nominal or suspicious_identity_value(nominal):
        for text in (
            row.get("nominal_value_raw", ""),
            row.get("shares_outstanding_raw", ""),
            row.get("nominal_value_raw_backup_hint", ""),
            row.get("shares_outstanding_raw_backup_hint", ""),
        ):
            candidates = [
                value
                for value in parsed_integer_values(text)
                if looks_like_nominal(value) and not is_large_integer(value)
            ]
            if candidates:
                nominal = candidates[0]
                flags.append("nominal_cell_rescued")
                break

    return capital, shares, nominal, flags


def rescue_close_quantity(
    price_close: str,
    quantity: str,
    row: dict[str, str],
) -> tuple[str, str, list[str]]:
    flags: list[str] = []
    close_sources = [
        row.get("price_close_raw", ""),
        row.get("price_close_raw_backup_hint", ""),
        row.get("price_max_raw", ""),
    ]
    quantity_sources = [
        row.get("quantity_traded_raw", ""),
        row.get("quantity_traded_raw_backup_hint", ""),
    ]
    if not price_close or suspicious_identity_value(price_close):
        for source in close_sources:
            values = [value for value in parsed_numbers(source, decimal_allowed=True) if looks_like_price(value)]
            if values:
                price_close = values[-1]
                flags.append("close_cell_backup_rescued")
                break
    if not quantity:
        for source in quantity_sources:
            values = [value for value in parsed_numbers(source) if looks_like_quantity(value)]
            if values:
                quantity = values[-1]
                flags.append("quantity_cell_backup_rescued")
                break
    return price_close, quantity, flags


def looks_like_section_heading_row(issuer: str, raw_text: str, capital: str, shares: str, nominal: str) -> bool:
    text = clean_text(f"{issuer} {raw_text}").lower()
    normalized = normalize_lookup_text(text)
    issuer_normalized = normalize_lookup_text(issuer)
    if issuer_normalized in SECTION_HEADING_PHRASES:
        return True
    if any(phrase in normalized for phrase in SECTION_HEADING_PHRASES):
        number_count = len(parsed_numbers(raw_text, decimal_allowed=True))
        if not capital or not shares or not nominal or number_count <= 4:
            return True
    if capital or shares or nominal:
        return False
    words = set(re.findall(r"[a-zÃ -Ã¿]+", text))
    if not words:
        return False
    if words & SECTION_HEADINGS:
        number_count = len(parsed_numbers(raw_text, decimal_allowed=True))
        return number_count <= 2
    return False


def has_only_small_leakage(tokens: list[str], selected_value: str, *, max_extra: int = 31) -> bool:
    if len(tokens) <= 1 or not selected_value:
        return False
    parsed = [parse_number(token, decimal_allowed=False) for token in tokens]
    parsed = [value for value in parsed if value]
    if not parsed or parsed[0] != selected_value:
        return False
    extras = parsed[1:]
    return bool(extras) and all("." not in value and int(value) <= max_extra for value in extras)


def parse_date(text: str) -> str:
    match = DATE_RE.search(text or "")
    if not match:
        return ""
    day, month, year = [int(part) for part in match.groups()]
    if year < 100:
        year += 1900
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_godimento(text: str) -> str:
    lower = (text or "").lower()
    day_match = re.search(r"\d{1,2}", lower)
    day = int(day_match.group(0)) if day_match else 1
    for key, month in MONTHS.items():
        if key in lower:
            return f"--{month:02d}-{day:02d}"
    return ""


def clean_issuer(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^[^A-Za-zÀ-ÿ]+", "", text)
    text = re.sub(r"\b(eee|ee|oe|rai|yee|DICI|ACE|ACCO|RC|nce)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,-;:")


def issuer_from_raw_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\b\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}\b", " ", text)
    text = re.sub(r"\d{1,3}(?:[., ]\d{3})+|\d+(?:,\d+)?", " ", text)
    candidates = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'’&() ]{2,}", text)
    cleaned: list[str] = []
    for candidate in candidates:
        issuer = clean_issuer(candidate)
        words = [word for word in issuer.split() if word.lower().strip(".") not in ISSUER_STOP_WORDS]
        issuer = " ".join(words).strip()
        if len(issuer) >= 4:
            cleaned.append(issuer)
    if not cleaned:
        return ""
    return max(cleaned, key=lambda value: (len(value.split()), len(value)))


def compensation_text_flags(text: str) -> list[str]:
    text = clean_text(text).lower()
    words = re.findall(r"[a-zà-ÿ]{2,}", text)
    if not words:
        return []
    benign = {"ass", "gr", "ex", "opt", "stamp"}
    if all(word in benign for word in words):
        return ["benign_compensation_notation"]
    if any(word in benign for word in words):
        return ["benign_compensation_notation", "text_in_compensation"]
    return ["text_in_compensation"]


def add_flag(flags: list[str], condition: bool, name: str) -> None:
    if condition:
        flags.append(name)


def looks_like_fragment_row(
    row: dict[str, str],
    issuer: str,
    capital: str,
    shares: str,
    nominal: str,
    mean_conf: float,
) -> bool:
    if row.get("row_role") != "security_main_row":
        return False
    if capital and shares and nominal:
        return False
    if mean_conf >= 65:
        return False
    raw_values = parsed_numbers(row.get("raw_row_text", ""), decimal_allowed=True)
    large_values = [value for value in raw_values if "." not in value and is_large_integer(value)]
    if large_values:
        return False
    if issuer and len(raw_values) <= 4:
        return True
    return not issuer and len(raw_values) <= 3


def clean_row(row: dict[str, str]) -> dict[str, str]:
    flags: list[str] = []

    capital, capital_tokens = first_numeric(row.get("capital_subscribed_raw", ""), prefer_large=True)
    shares, shares_tokens = first_numeric(row.get("shares_outstanding_raw", ""), prefer_large=True)
    nominal, nominal_tokens = first_numeric(row.get("nominal_value_raw", ""), max_value=100000)
    coupon_date = parse_date(row.get("cedola_date_raw", ""))
    godimento = parse_godimento(row.get("godimento_raw", ""))
    importo_lordo, lordo_tokens = first_numeric(row.get("importo_lordo_raw", ""), decimal_allowed=True)
    importo_acconto, acconto_tokens = first_numeric(row.get("importo_acconto_raw", ""), decimal_allowed=True)
    importo_saldo, saldo_tokens = first_numeric(row.get("importo_saldo_raw", ""), decimal_allowed=True)
    compensation, compensation_tokens = first_numeric(row.get("compensation_price_raw", ""), decimal_allowed=True)
    price_min, min_tokens = first_numeric(row.get("price_min_raw", ""), decimal_allowed=True)
    price_max, max_tokens = first_numeric(row.get("price_max_raw", ""), decimal_allowed=True)
    price_close, close_tokens = first_numeric(row.get("price_close_raw", ""), decimal_allowed=True)
    quantity, quantity_tokens = first_numeric(row.get("quantity_traded_raw", ""))
    issuer = clean_issuer(row.get("issuer_name_raw", ""))
    if not issuer:
        issuer = issuer_from_raw_text(row.get("raw_row_text", ""))
        if issuer:
            flags.append("issuer_from_raw_text_rescued")

    capital = strip_leading_row_marker(capital)
    shares = strip_leading_row_marker(shares)

    shares_values_in_order = parsed_numbers(row.get("shares_outstanding_raw", ""))
    shares_values_large = [value for value in shares_values_in_order if is_large_integer(value)]
    if (
        not capital
        and len(shares_values_large) >= 2
    ):
        capital = shares_values_large[0]
        shares = shares_values_large[1]
        flags.append("left_numeric_shift_rescued")

    repaired_capital, repaired_shares, repaired_nominal, identity_flags = rescue_identity_fields(
        capital,
        shares,
        nominal,
        row,
    )
    capital, shares, nominal = repaired_capital, repaired_shares, repaired_nominal
    flags.extend(identity_flags)

    raw_tail_values = parsed_numbers(row.get("raw_row_text", ""), decimal_allowed=True)
    issuer_has_leaked_numbers = bool(re.search(r"\d", row.get("issuer_name_raw", "")))
    right_page_crop = "_right_" in row.get("source_file", "")
    if right_page_crop and (not capital or not nominal):
        fallback_capital, fallback_shares, fallback_nominal = identity_fallback(row.get("raw_row_text", ""))
        if fallback_capital and fallback_shares:
            if not capital:
                capital = fallback_capital
                shares = fallback_shares
                flags.append("right_identity_cells_rescued")
            if (not nominal or is_large_integer(nominal)) and fallback_nominal:
                nominal = fallback_nominal
                if "right_identity_cells_rescued" not in flags:
                    flags.append("right_identity_cells_rescued")
    if not price_close and not quantity and right_page_crop and issuer_has_leaked_numbers and len(raw_tail_values) >= 4:
        tail = raw_tail_values[-4:]
        if looks_like_quantity(tail[-1]):
            quantity = tail[-1]
            flags.append("tail_quantity_rescued")
        price_candidates = [value for value in tail[:-1] if looks_like_price(value)]
        if price_candidates:
            price_close = price_candidates[-1]
            flags.append("tail_close_price_rescued")

    if right_page_crop and not price_close:
        fallback_close, fallback_quantity = price_quantity_fallback(
            row.get("price_min_raw", ""),
            row.get("price_max_raw", ""),
        )
        if fallback_close:
            price_close = fallback_close
            flags.append("right_price_cells_rescued")
        if not quantity and fallback_quantity:
            quantity = fallback_quantity
            flags.append("right_quantity_cells_rescued")

    price_close, quantity, close_quantity_flags = rescue_close_quantity(price_close, quantity, row)
    flags.extend(close_quantity_flags)

    issuer, capital, shares, nominal, reference_flags = rescue_identity_from_reference(
        issuer,
        capital,
        shares,
        nominal,
        row,
    )
    flags.extend(reference_flags)

    row_role = row.get("row_role", "")
    non_security_row = looks_like_section_heading_row(issuer, row.get("raw_row_text", ""), capital, shares, nominal)
    if non_security_row:
        row_role = "section_heading"
        flags.append("probable_section_heading_row")

    try:
        mean_conf = float(row.get("mean_conf", ""))
    except ValueError:
        mean_conf = 0.0
    probable_fragment_row = looks_like_fragment_row(row, issuer, capital, shares, nominal, mean_conf)
    if probable_fragment_row:
        row_role = "probable_row_fragment"
        flags.append("probable_row_fragment")

    quantity_raw = row.get("quantity_traded_raw", "")
    close_raw = row.get("price_close_raw", "")
    if not quantity and not non_security_row:
        if price_close and is_blank_or_dash_cell(quantity_raw):
            flags.append("quantity_likely_printed_blank")
        elif quantity_raw and not has_number(quantity_raw, decimal_allowed=False):
            flags.append("quantity_cell_ocr_noise")
        else:
            flags.append("quantity_needs_review")

    if not price_close and not non_security_row:
        if is_blank_or_dash_cell(close_raw) and is_blank_or_dash_cell(row.get("price_close_raw_backup_hint", "")):
            flags.append("close_likely_printed_blank")
        elif close_raw or row.get("price_close_raw_backup_hint", ""):
            flags.append("close_cell_ocr_noise")
        elif trailing_numeric_values(row.get("raw_row_text", "")):
            flags.append("close_missing_but_row_has_numbers")
        else:
            flags.append("close_needs_review")

    suppress_identity_missing = non_security_row or probable_fragment_row
    add_flag(flags, not issuer and not suppress_identity_missing, "missing_issuer")
    add_flag(flags, not capital and not suppress_identity_missing, "missing_capital")
    add_flag(flags, not shares and not suppress_identity_missing, "missing_shares")
    add_flag(flags, not nominal and not suppress_identity_missing, "missing_nominal")
    add_flag(flags, not price_close and not non_security_row, "missing_close_price")
    add_flag(flags, not quantity and not non_security_row, "blank_quantity")
    add_flag(flags, len(close_tokens) > 1, "multi_value_close_raw")
    add_flag(flags, len(row.get("shares_outstanding_raw", "").split("/")) > 1, "glued_shares_nominal_raw")
    add_flag(
        flags,
        len(shares_tokens) > 1
        and "left_numeric_shift_rescued" not in flags
        and "glued_shares_nominal_raw" not in flags,
        "multi_value_shares_raw",
    )
    add_flag(
        flags,
        len(nominal_tokens) > 1 and not has_only_small_leakage(nominal_tokens, nominal),
        "multi_value_nominal_raw",
    )
    for flag in compensation_text_flags(row.get("compensation_price_raw", "")):
        if flag not in flags:
            flags.append(flag)
    add_flag(flags, bool(re.search(r"[A-Za-z]{2,}", row.get("price_close_raw", ""))), "text_in_close")

    add_flag(flags, mean_conf and mean_conf < 65, "low_mean_ocr_confidence")

    return {
        "row_index": row.get("row_index", ""),
        "row_role": row_role,
        "section": row.get("section", ""),
        "issuer_name_clean": issuer,
        "capital_subscribed": capital,
        "shares_outstanding": shares,
        "nominal_value": nominal,
        "godimento_month_day": godimento,
        "cedola_date": coupon_date,
        "importo_lordo": importo_lordo,
        "importo_acconto": importo_acconto,
        "importo_saldo": importo_saldo,
        "compensation_price": compensation,
        "price_min": price_min,
        "price_max": price_max,
        "price_close": price_close,
        "quantity_traded": quantity,
        "validation_flags": ";".join(flags),
        "raw_row_text": row.get("raw_row_text", ""),
        "capital_subscribed_raw": row.get("capital_subscribed_raw", ""),
        "shares_outstanding_raw": row.get("shares_outstanding_raw", ""),
        "nominal_value_raw": row.get("nominal_value_raw", ""),
        "godimento_raw": row.get("godimento_raw", ""),
        "cedola_date_raw": row.get("cedola_date_raw", ""),
        "importo_lordo_raw": row.get("importo_lordo_raw", ""),
        "importo_acconto_raw": row.get("importo_acconto_raw", ""),
        "importo_saldo_raw": row.get("importo_saldo_raw", ""),
        "compensation_price_raw": row.get("compensation_price_raw", ""),
        "issuer_name_raw": row.get("issuer_name_raw", ""),
        "price_min_raw": row.get("price_min_raw", ""),
        "price_max_raw": row.get("price_max_raw", ""),
        "price_close_raw": row.get("price_close_raw", ""),
        "quantity_traded_raw": row.get("quantity_traded_raw", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean one-page azioni candidate rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_csv(args.input)
    cleaned = [clean_row(row) for row in rows if row.get("row_type") == "candidate_security_row"]

    fields = [
        "row_index",
        "row_role",
        "section",
        "issuer_name_clean",
        "capital_subscribed",
        "shares_outstanding",
        "nominal_value",
        "godimento_month_day",
        "cedola_date",
        "importo_lordo",
        "importo_acconto",
        "importo_saldo",
        "compensation_price",
        "price_min",
        "price_max",
        "price_close",
        "quantity_traded",
        "validation_flags",
        "raw_row_text",
        "capital_subscribed_raw",
        "shares_outstanding_raw",
        "nominal_value_raw",
        "godimento_raw",
        "cedola_date_raw",
        "importo_lordo_raw",
        "importo_acconto_raw",
        "importo_saldo_raw",
        "compensation_price_raw",
        "issuer_name_raw",
        "price_min_raw",
        "price_max_raw",
        "price_close_raw",
        "quantity_traded_raw",
    ]
    write_csv(args.output_dir / "cleaned_candidate_rows_one_page.csv", cleaned, fields)

    issue_rows = [row for row in cleaned if row["validation_flags"]]
    write_csv(args.output_dir / "cleaning_issues_one_page.csv", issue_rows, fields)

    flag_counts: dict[str, int] = {}
    for row in cleaned:
        for flag in row["validation_flags"].split(";"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary_rows = [{"flag": flag, "count": str(count)} for flag, count in sorted(flag_counts.items())]
    write_csv(args.output_dir / "validation_flag_summary.csv", summary_rows, ["flag", "count"])

    readme = [
        "# One-page azioni cleaning output",
        "",
        "This is a review dataset, not the final panel.",
        "",
        f"Input: `{args.input}`",
        "",
        "Outputs:",
        "- `cleaned_candidate_rows_one_page.csv`: normalized draft fields plus raw values.",
        "- `cleaning_issues_one_page.csv`: rows with validation flags.",
        "- `validation_flag_summary.csv`: counts of each issue type.",
        "",
        "Interpretation:",
        "- Normalized fields are first-pass guesses.",
        "- Raw fields are kept for audit and later correction.",
        "- Validation flags identify rows that should not be trusted without review.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Rows with flags: {len(issue_rows)}")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
