"""Clean and validate prototype regular_obbligazioni records."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = Path("output/parsed_obbligazioni_sample/regular_obbligazioni/structured_records.csv")
DEFAULT_OUTPUT = Path("output/cleaned_obbligazioni_sample")

NUMBER_RE = re.compile(r"\d[\d.,]*")
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2,4})(?!\d)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str) -> str:
    text = text or ""
    replacements = {
        "Ã ": "a",
        "Ã¨": "e",
        "Ã©": "e",
        "Ã¬": "i",
        "Ã²": "o",
        "Ã¹": "u",
        "â€™": "'",
        "â€”": "-",
        "â€“": "-",
        "—": "-",
        "–": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[{}|\\\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def number_tokens(text: str) -> list[str]:
    return [match.group(0).strip(".,") for match in NUMBER_RE.finditer(clean_text(text))]


def parse_number(token: str, *, decimal_allowed: bool = False) -> str:
    token = clean_text(token).replace(" ", "").replace("'", "")
    token = re.sub(r"(?<=\d)[OoC](?=\d)", "0", token)
    token = token.strip(".,;:-")
    if not token:
        return ""
    if "," in token:
        left, right = token.rsplit(",", 1)
        left = re.sub(r"\D", "", left)
        if decimal_allowed and right.isdigit() and len(right) <= 2:
            return f"{left}.{right}" if left else ""
        return re.sub(r"\D", "", left + right)
    if "." in token:
        parts = token.split(".")
        if decimal_allowed and len(parts[-1]) <= 2 and all(part.isdigit() for part in parts):
            return f"{''.join(parts[:-1])}.{parts[-1]}"
        return re.sub(r"\D", "", token)
    return re.sub(r"\D", "", token)


def first_number(text: str, *, decimal_allowed: bool = False) -> str:
    for token in number_tokens(text):
        value = parse_number(token, decimal_allowed=decimal_allowed)
        if value:
            return value
    return ""


def parsed_numbers(text: str, *, decimal_allowed: bool = False) -> list[str]:
    values: list[str] = []
    for token in number_tokens(text):
        value = parse_number(token, decimal_allowed=decimal_allowed)
        if value:
            values.append(value)
    return values


def last_number(text: str, *, decimal_allowed: bool = False) -> str:
    values = parsed_numbers(text, decimal_allowed=decimal_allowed)
    return values[-1] if values else ""


def parse_price_and_quantity(price_text: str, quantity_text: str) -> tuple[str, str, list[str], list[str]]:
    validation_flags: list[str] = []
    processing_flags: list[str] = []
    cleaned_price = clean_text(price_text)
    price_values = parsed_numbers(price_text, decimal_allowed=True)
    quantity = first_number(quantity_text)
    price = ""
    if (
        len(price_values) == 2
        and re.search(r"^\D*\d{1,3}\s+\d{1,2}\D*$", cleaned_price)
        and "," not in cleaned_price
        and "-" not in cleaned_price
    ):
        price = f"{parse_number(price_values[0])}.{parse_number(price_values[1]).zfill(2)}"
        processing_flags.append("price_decimal_split_from_ocr_tokens")
    elif len(price_values) >= 3 and re.search(r"[/]\s*\w?", cleaned_price):
        price = f"{parse_number(price_values[0])}.{parse_number(price_values[1]).zfill(2)}"
        if not quantity:
            quantity = parse_number(price_values[-1])
            processing_flags.append("quantity_split_from_price_cell")
        processing_flags.append("price_decimal_split_from_ocr_tokens")
    elif price_values:
        price = price_values[0]
    if len(price_values) > 1:
        validation_flags.append("possible_price_column_merge")
        last_value = parse_number(price_values[-1])
        quantity_hint = "/" in cleaned_price or re.search(r"\bm\b", cleaned_price, flags=re.IGNORECASE)
        if not quantity and quantity_hint and last_value:
            quantity = parse_number(price_values[-1])
            processing_flags.append("quantity_split_from_price_cell")
    return price, quantity, validation_flags, processing_flags


def parse_year(text: str) -> str:
    cleaned = clean_text(text)
    range_match = re.search(r"(?<!\d)(\d{2})\s*[-/]\s*(\d{2})(?!\d)", cleaned)
    if range_match:
        start, end = [int(part) for part in range_match.groups()]
        start += 1900 if start >= 30 else 2000
        end += 1900 if end >= 30 else 2000
        return f"{start}-{end}"
    for token in number_tokens(text):
        value = parse_number(token)
        if not value:
            continue
        if len(value) == 4 and 1850 <= int(value) <= 2100:
            return value
        if re.match(r"^\d{2}-\d{2}$", token):
            return token
    return ""


def infer_left_fields_from_row(raw_text: str) -> dict[str, str]:
    text = clean_text(raw_text)
    tokens = number_tokens(text)
    inferred = {
        "capital_in_circulation": "",
        "nominal_value": "",
        "maturity_year": "",
    }
    if len(tokens) < 3:
        return inferred
    parsed = [parse_number(token) for token in tokens[:8]]

    def looks_like_nominal(value: str) -> bool:
        return value in {"50", "75", "100", "150", "190", "200", "250", "300", "350", "400", "500", "750", "1000", "1250", "1500", "2000", "2500", "3000", "4000", "5000"}

    def looks_like_year_token(token: str, value: str) -> bool:
        if parse_year(token):
            return True
        return len(value) == 4 and 1850 <= int(value) <= 2100

    def looks_like_capital(token: str, value: str) -> bool:
        if not value or looks_like_nominal(value):
            return False
        if len(value) == 4 and 1850 <= int(value) <= 2100:
            return False
        return "." in token or len(value) >= 5

    def combined_nominal_at(index: int) -> tuple[str, int]:
        if index + 1 >= len(parsed):
            return "", index
        combined = f"{parsed[index] or ''}{parsed[index + 1] or ''}"
        if looks_like_nominal(combined):
            return combined, index + 1
        return "", index

    for index, value in enumerate(parsed[:5]):
        if not value:
            continue
        if not inferred["capital_in_circulation"] and looks_like_capital(tokens[index], value):
            inferred["capital_in_circulation"] = value
            continue

        if not inferred["nominal_value"] and looks_like_nominal(value):
            inferred["nominal_value"] = value
            continue

        if not inferred["nominal_value"]:
            combined_nominal, used_index = combined_nominal_at(index)
            if combined_nominal:
                inferred["nominal_value"] = combined_nominal
                if used_index != index:
                    continue

        if inferred["nominal_value"] and not inferred["maturity_year"]:
            token = tokens[index]
            year = parse_year(token)
            if year or looks_like_year_token(token, value):
                inferred["maturity_year"] = year or value
                break
    return inferred


def parse_date(text: str) -> str:
    match = DATE_RE.search(clean_text(text))
    if not match:
        return ""
    day, month, year = [int(part) for part in match.groups()]
    if year < 100:
        year += 1900
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def clean_security_name(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"\b(eee|ee|oe|rrr|ree|papi|tizia)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;-")


def looks_like_section(row: dict[str, str]) -> bool:
    if row.get("record_type") == "section_heading":
        return True
    text = clean_text(row.get("raw_text", "")).lower()
    terms = [
        "garanzie statali",
        "reddito fisso",
        "partecipazione",
        "titoli di stati esteri",
        "fondiari",
        "equiparati",
        "comunali",
        "provinciali",
        "trasporti",
        "elettrici",
        "diversi",
    ]
    return any(term in text for term in terms) and len(number_tokens(text)) <= 2


def looks_like_footnote(row: dict[str, str]) -> bool:
    text = clean_text(row.get("raw_text", "")).lower()
    terms = [
        "diritto a",
        "azioni di categoria",
        "sono due tipi",
        "riserva legale",
        "cartelle di godimento",
        "privilegiate nella ripartizione",
        "comitato direttivo",
    ]
    return any(term in text for term in terms)


def looks_like_continuation(row: dict[str, str], security: str) -> bool:
    raw_security = clean_text(row.get("security_name_raw", ""))
    raw_left = " ".join(
        clean_text(row.get(field, ""))
        for field in [
            "capital_in_circulation_raw",
            "nominal_value_raw",
            "maturity_year_raw",
            "extraction_period_raw",
            "godimento_raw",
        ]
    )
    if re.match(r"^[»\"'`.,;:\-\s]+", raw_security):
        return True
    if security and len(security) <= 8 and re.search(r"[%»\"]", raw_security):
        return True
    if not raw_left.strip() and security:
        return True
    return False


def non_dash_text(text: str) -> str:
    text = clean_text(text)
    return re.sub(r"[-—–_.,;:\s]+", "", text)


def clean_row(row: dict[str, str]) -> dict[str, str]:
    flags: list[str] = []
    processing_flags: list[str] = []
    row_type = row.get("record_type", "")
    row_role = row.get("record_role", "")
    if looks_like_footnote(row):
        row_type = "other_text"
        row_role = "footnote_text"
    if looks_like_section(row):
        row_type = "section_heading"
        row_role = "section_heading"

    security = clean_security_name(row.get("security_name_raw", ""))
    capital = first_number(row.get("capital_in_circulation_raw", ""))
    nominal = first_number(row.get("nominal_value_raw", ""))
    inferred_left = infer_left_fields_from_row(row.get("raw_text", ""))
    capital_extra = ""
    nominal_extra = ""
    capital_values = parsed_numbers(row.get("capital_in_circulation_raw", ""))
    nominal_values = parsed_numbers(row.get("nominal_value_raw", ""))
    if len(capital_values) > 1:
        capital_extra = capital_values[1]
    if len(nominal_values) > 1:
        nominal_extra = nominal_values[1]
    maturity = parse_year(row.get("maturity_year_raw", ""))
    extraction = clean_text(row.get("extraction_period_raw", ""))
    godimento = clean_text(row.get("godimento_raw", ""))
    coupon_date = parse_date(row.get("coupon_date_raw", ""))
    coupon_net = first_number(row.get("coupon_net_amount_raw", ""), decimal_allowed=True)
    coupon_number = first_number(row.get("coupon_number_raw", ""))
    compensation = first_number(row.get("compensation_price_raw", ""), decimal_allowed=True)
    price_min = first_number(row.get("price_min_raw", ""), decimal_allowed=True)
    price_max = first_number(row.get("price_max_raw", ""), decimal_allowed=True)
    price_close, quantity, split_flags, price_processing_flags = parse_price_and_quantity(
        row.get("price_close_raw", ""),
        row.get("quantity_traded_raw", ""),
    )
    price_close_filled = price_close
    price_close_fill_source = "ocr" if price_close else ""
    if not price_close:
        if price_max:
            price_close_filled = last_number(row.get("price_max_raw", ""), decimal_allowed=True) or price_max
            price_close_fill_source = "inferred_from_price_max"
            processing_flags.append("price_close_inferred_from_price_max")
        elif price_min:
            price_close_filled = last_number(row.get("price_min_raw", ""), decimal_allowed=True) or price_min
            price_close_fill_source = "inferred_from_price_min"
            processing_flags.append("price_close_inferred_from_price_min")
    processing_flags.extend(price_processing_flags)
    is_continuation = looks_like_continuation(row, security)
    if row_type == "candidate_record" and is_continuation:
        row_role = "candidate_continuation"

    is_candidate = row_type == "candidate_record"
    if is_candidate:
        flags.extend(split_flags)
        if not security:
            flags.append("missing_security_name")
        if not capital and not re.search(r"[»\"]|^\s*[-.]", row.get("security_name_raw", "")):
            flags.append("missing_capital")
        if not nominal:
            flags.append("missing_nominal")
        if not maturity:
            flags.append("missing_maturity_year")
        if not price_close:
            flags.append("missing_close_price")
        if row.get("mean_conf") and float(row.get("mean_conf", "0") or 0) < 55:
            flags.append("low_mean_ocr_confidence")
        if len(number_tokens(row.get("capital_in_circulation_raw", ""))) > 1:
            flags.append("multi_value_capital_raw")
        if len(number_tokens(row.get("nominal_value_raw", ""))) > 1:
            flags.append("multi_value_nominal_raw")
        if len(number_tokens(row.get("price_close_raw", ""))) > 1:
            flags.append("multi_value_close_raw")
        raw_price_text = " ".join(
            row.get(field, "")
            for field in ["price_min_raw", "price_max_raw", "price_close_raw", "quantity_traded_raw"]
        )
        normalized_flags: list[str] = []
        for flag in flags:
            if is_continuation and flag in {"missing_capital", "missing_nominal", "missing_maturity_year"}:
                continue
            if flag in {"missing_capital", "missing_nominal", "missing_maturity_year"}:
                continue
            if flag == "missing_close_price" and price_close_filled:
                continue
            if flag == "missing_close_price" and not non_dash_text(raw_price_text):
                continue
            if flag in {"multi_value_capital_raw", "multi_value_nominal_raw"}:
                flag = "possible_left_column_merge"
            if flag == "multi_value_close_raw":
                flag = "possible_price_column_merge"
            normalized_flags.append(flag)
        if not is_continuation:
            missing_left = [not capital, not nominal, not maturity].count(True)
            if missing_left >= 2:
                normalized_flags.append("missing_key_left_fields")
        flags = sorted(set(normalized_flags))

    return {
        "sample_id": row.get("sample_id", ""),
        "record_index": row.get("record_index", ""),
        "record_type": row_type,
        "record_role": row_role,
        "section": row.get("section", ""),
        "security_name_clean": security,
        "capital_in_circulation": capital,
        "nominal_value": nominal,
        "capital_in_circulation_inferred": inferred_left.get("capital_in_circulation", ""),
        "nominal_value_inferred": inferred_left.get("nominal_value", ""),
        "maturity_year_inferred": inferred_left.get("maturity_year", ""),
        "capital_in_circulation_extra": capital_extra,
        "nominal_value_extra": nominal_extra,
        "maturity_year": maturity,
        "extraction_period": extraction,
        "godimento": godimento,
        "coupon_date": coupon_date,
        "coupon_net_amount": coupon_net,
        "coupon_number": coupon_number,
        "compensation_price": compensation,
        "price_min": price_min,
        "price_max": price_max,
        "price_close": price_close,
        "price_close_filled": price_close_filled,
        "price_close_fill_source": price_close_fill_source,
        "quantity_traded": quantity,
        "processing_flags": ";".join(sorted(set(processing_flags))),
        "validation_flags": ";".join(flags),
        "raw_text": row.get("raw_text", ""),
        "capital_in_circulation_raw": row.get("capital_in_circulation_raw", ""),
        "nominal_value_raw": row.get("nominal_value_raw", ""),
        "maturity_year_raw": row.get("maturity_year_raw", ""),
        "extraction_period_raw": row.get("extraction_period_raw", ""),
        "godimento_raw": row.get("godimento_raw", ""),
        "coupon_date_raw": row.get("coupon_date_raw", ""),
        "coupon_net_amount_raw": row.get("coupon_net_amount_raw", ""),
        "coupon_number_raw": row.get("coupon_number_raw", ""),
        "compensation_price_raw": row.get("compensation_price_raw", ""),
        "security_name_raw": row.get("security_name_raw", ""),
        "price_min_raw": row.get("price_min_raw", ""),
        "price_max_raw": row.get("price_max_raw", ""),
        "price_close_raw": row.get("price_close_raw", ""),
        "quantity_traded_raw": row.get("quantity_traded_raw", ""),
    }


FILLABLE_FIELDS = [
    "security_name_clean",
    "capital_in_circulation",
    "nominal_value",
    "maturity_year",
    "extraction_period",
    "godimento",
]


def add_flag(row: dict[str, str], flag: str) -> None:
    flags = [item for item in row.get("validation_flags", "").split(";") if item]
    flags.append(flag)
    row["validation_flags"] = ";".join(sorted(set(flags)))


def add_processing_flag(row: dict[str, str], flag: str) -> None:
    flags = [item for item in row.get("processing_flags", "").split(";") if item]
    flags.append(flag)
    row["processing_flags"] = ";".join(sorted(set(flags)))


def remove_flags(row: dict[str, str], flags_to_remove: set[str]) -> None:
    flags = [item for item in row.get("validation_flags", "").split(";") if item and item not in flags_to_remove]
    row["validation_flags"] = ";".join(sorted(set(flags)))


def apply_carry_forward(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    last_by_sample: dict[str, dict[str, str]] = {}
    last_sample = ""
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id != last_sample:
            last_by_sample = {}
            last_sample = sample_id

        for field in FILLABLE_FIELDS:
            row[f"{field}_filled"] = row.get(field, "")
            row[f"{field}_fill_source"] = "ocr" if row.get(field, "") else ""

        if row.get("record_type") != "candidate_record":
            continue

        row_role = row.get("record_role", "")
        inherited_any = False
        has_enough_left_values = [
            bool(row.get("capital_in_circulation")),
            bool(row.get("nominal_value")),
            bool(row.get("maturity_year")),
        ].count(True) >= 2
        for field in FILLABLE_FIELDS:
            if row.get(field):
                continue
            carried = last_by_sample.get(field, "")
            can_carry_security = field == "security_name_clean" and has_enough_left_values
            can_carry_repeated_field = row_role == "candidate_continuation" or field in {"extraction_period", "godimento"}
            if carried and (can_carry_security or can_carry_repeated_field):
                row[f"{field}_filled"] = carried
                row[f"{field}_fill_source"] = "carry_forward"
                inherited_any = True
                if field == "security_name_clean":
                    add_processing_flag(row, "security_name_carried_forward")

        present_left = [
            bool(row.get("capital_in_circulation_filled")),
            bool(row.get("nominal_value_filled")),
            bool(row.get("maturity_year_filled")),
        ]
        if inherited_any:
            add_processing_flag(row, "used_carry_forward")
        if present_left.count(True) >= 2:
            remove_flags(row, {"missing_key_left_fields"})
        if row.get("security_name_clean_filled") and present_left.count(True) >= 2:
            remove_flags(row, {"missing_security_name"})
        inferred_any = False
        for field in ["capital_in_circulation", "nominal_value", "maturity_year"]:
            filled_key = f"{field}_filled"
            source_key = f"{field}_fill_source"
            if row.get(filled_key):
                continue
            inferred_value = row.get(f"{field}_inferred", "")
            if inferred_value:
                row[filled_key] = inferred_value
                row[source_key] = "inferred_from_raw_row"
                inferred_any = True
        if inferred_any:
            add_processing_flag(row, "left_fields_inferred_from_raw_row")
            present_left = [
                bool(row.get("capital_in_circulation_filled")),
                bool(row.get("nominal_value_filled")),
                bool(row.get("maturity_year_filled")),
            ]
            if present_left.count(True) >= 2:
                remove_flags(row, {"missing_key_left_fields"})

        if row_role != "candidate_continuation":
            for field in FILLABLE_FIELDS:
                if row.get(field):
                    last_by_sample[field] = row[field]
        else:
            for field in FILLABLE_FIELDS:
                if row.get(field):
                    last_by_sample[field] = row[field]

    return rows


FIELDS = [
    "sample_id",
    "record_index",
    "record_type",
    "record_role",
    "section",
    "security_name_clean",
    "security_name_clean_filled",
    "security_name_clean_fill_source",
    "capital_in_circulation",
    "capital_in_circulation_inferred",
    "capital_in_circulation_extra",
    "nominal_value",
    "nominal_value_inferred",
    "nominal_value_extra",
    "maturity_year",
    "maturity_year_inferred",
    "extraction_period",
    "godimento",
    "coupon_date",
    "coupon_net_amount",
    "coupon_number",
    "compensation_price",
    "price_min",
    "price_max",
    "price_close",
    "price_close_filled",
    "price_close_fill_source",
    "quantity_traded",
    "capital_in_circulation_filled",
    "capital_in_circulation_fill_source",
    "nominal_value_filled",
    "nominal_value_fill_source",
    "maturity_year_filled",
    "maturity_year_fill_source",
    "extraction_period_filled",
    "extraction_period_fill_source",
    "godimento_filled",
    "godimento_fill_source",
    "processing_flags",
    "validation_flags",
    "raw_text",
    "capital_in_circulation_raw",
    "nominal_value_raw",
    "maturity_year_raw",
    "extraction_period_raw",
    "godimento_raw",
    "coupon_date_raw",
    "coupon_net_amount_raw",
    "coupon_number_raw",
    "compensation_price_raw",
    "security_name_raw",
    "price_min_raw",
    "price_max_raw",
    "price_close_raw",
    "quantity_traded_raw",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean regular_obbligazioni structured records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_csv(args.input)
    cleaned = apply_carry_forward([clean_row(row) for row in rows])
    candidates = [row for row in cleaned if row["record_type"] == "candidate_record"]
    issues = [row for row in candidates if row["validation_flags"]]
    flag_counts: dict[str, int] = {}
    processing_flag_counts: dict[str, int] = {}
    for row in candidates:
        for flag in row["validation_flags"].split(";"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        for flag in row["processing_flags"].split(";"):
            if flag:
                processing_flag_counts[flag] = processing_flag_counts.get(flag, 0) + 1

    write_csv(args.output_dir / "cleaned_obbligazioni_records.csv", cleaned, FIELDS)
    write_csv(args.output_dir / "candidate_obbligazioni_records.csv", candidates, FIELDS)
    write_csv(args.output_dir / "obbligazioni_review_issues.csv", issues, FIELDS)
    write_csv(
        args.output_dir / "validation_flag_summary.csv",
        [{"flag": flag, "count": str(count)} for flag, count in sorted(flag_counts.items())],
        ["flag", "count"],
    )
    write_csv(
        args.output_dir / "processing_flag_summary.csv",
        [{"flag": flag, "count": str(count)} for flag, count in sorted(processing_flag_counts.items())],
        ["flag", "count"],
    )
    readme = [
        "# Cleaned regular_obbligazioni output",
        "",
        "This is the first validation layer for the prototype obbligazioni parser.",
        "",
        "Outputs:",
        "- `cleaned_obbligazioni_records.csv`: all parsed rows, including section headings.",
        "- `candidate_obbligazioni_records.csv`: candidate bond/security rows only.",
        "- `obbligazioni_review_issues.csv`: candidate rows with validation flags.",
        "- `validation_flag_summary.csv`: counts of repeated issues.",
        "- `processing_flag_summary.csv`: counts of automatic repairs and transformations.",
        "",
        "The `_filled` columns apply conservative carry-forward within each page side for repeated bond fields.",
        "The `_extra` columns preserve a second numeric value when OCR appears to merge adjacent cells.",
        "",
        "Important: this is not as mature as the azioni cleaner yet. It is designed to reveal recurring error patterns for the next feedback loop.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Rows with review flags: {len(issues)}")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
