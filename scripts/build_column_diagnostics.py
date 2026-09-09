"""Build column-level diagnostics for the azioni batch parse.

The cleaner flags rows, but many remaining problems are column-specific:
identity columns, close price, quantity, or issuer text. This script summarizes
those issues by field and writes example rows for manual review.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_BATCH = Path("output/batch_30_azioni")

FIELDS = [
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
]

RAW_FIELD = {
    "issuer_name_clean": "issuer_name_raw",
    "capital_subscribed": "capital_subscribed_raw",
    "shares_outstanding": "shares_outstanding_raw",
    "nominal_value": "nominal_value_raw",
    "godimento_month_day": "godimento_raw",
    "cedola_date": "cedola_date_raw",
    "importo_lordo": "importo_lordo_raw",
    "importo_acconto": "importo_acconto_raw",
    "importo_saldo": "importo_saldo_raw",
    "compensation_price": "compensation_price_raw",
    "price_min": "price_min_raw",
    "price_max": "price_max_raw",
    "price_close": "price_close_raw",
    "quantity_traded": "quantity_traded_raw",
}

FIELD_FLAGS = {
    "issuer_name_clean": {
        "missing_issuer",
        "issuer_from_raw_text_rescued",
        "issuer_inherited",
        "text_in_close",
        "text_in_compensation",
    },
    "capital_subscribed": {
        "missing_capital",
        "identity_columns_rescued",
        "right_identity_cells_rescued",
        "left_numeric_shift_rescued",
        "identity_inherited_from_previous_row",
        "probable_split_identity_prefix",
        "probable_row_fragment",
    },
    "shares_outstanding": {
        "missing_shares",
        "multi_value_shares_raw",
        "glued_shares_nominal_raw",
        "identity_columns_rescued",
        "right_identity_cells_rescued",
        "left_numeric_shift_rescued",
        "identity_inherited_from_previous_row",
        "probable_split_identity_prefix",
        "probable_row_fragment",
    },
    "nominal_value": {
        "missing_nominal",
        "multi_value_nominal_raw",
        "glued_shares_nominal_raw",
        "nominal_cell_rescued",
        "right_identity_cells_rescued",
        "identity_inherited_from_previous_row",
        "identity_from_previous_split_prefix",
        "probable_split_identity_prefix",
        "probable_row_fragment",
    },
    "price_close": {
        "missing_close_price",
        "multi_value_close_raw",
        "close_cell_backup_rescued",
        "close_cell_ocr_noise",
        "close_likely_printed_blank",
        "right_price_cells_rescued",
        "tail_close_price_rescued",
        "text_in_close",
    },
    "quantity_traded": {
        "blank_quantity",
        "quantity_cell_ocr_noise",
        "quantity_needs_review",
        "quantity_likely_printed_blank",
        "right_quantity_cells_rescued",
        "tail_quantity_rescued",
    },
    "compensation_price": {
        "benign_compensation_notation",
        "text_in_compensation",
    },
}

GENERIC_FLAGS = {
    "low_mean_ocr_confidence",
    "probable_row_fragment",
}

EXAMPLE_FIELDS = [
    "field",
    "issue_kind",
    "sample_id",
    "source_file",
    "row_index",
    "section",
    "issuer_name_clean",
    "clean_value",
    "raw_value",
    "validation_flags",
    "raw_row_text",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_flags(row: dict[str, str]) -> set[str]:
    return {flag for flag in row.get("validation_flags", "").split(";") if flag}


def is_empty(value: str) -> bool:
    return not (value or "").strip()


def add_example(
    examples: list[dict[str, str]],
    *,
    field: str,
    issue_kind: str,
    row: dict[str, str],
    max_per_field_issue: int,
    counts: Counter[tuple[str, str]],
) -> None:
    key = (field, issue_kind)
    if counts[key] >= max_per_field_issue:
        return
    raw_field = RAW_FIELD[field]
    examples.append(
        {
            "field": field,
            "issue_kind": issue_kind,
            "sample_id": row.get("sample_id", ""),
            "source_file": row.get("source_file", ""),
            "row_index": row.get("row_index", ""),
            "section": row.get("section", ""),
            "issuer_name_clean": row.get("issuer_name_clean", ""),
            "clean_value": row.get(field, ""),
            "raw_value": row.get(raw_field, ""),
            "validation_flags": row.get("validation_flags", ""),
            "raw_row_text": row.get("raw_row_text", ""),
        }
    )
    counts[key] += 1


def build_diagnostics(rows: list[dict[str, str]], max_examples: int) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    total_rows = len(rows)
    summaries: list[dict[str, object]] = []
    examples: list[dict[str, str]] = []
    example_counts: Counter[tuple[str, str]] = Counter()

    for field in FIELDS:
        raw_field = RAW_FIELD[field]
        field_specific_flags = FIELD_FLAGS.get(field, set())
        empty_clean = 0
        empty_raw = 0
        raw_present_clean_empty = 0
        clean_present_raw_empty = 0
        flagged_rows = 0
        rescued_rows = 0
        likely_blank_rows = 0
        review_rows = 0
        flag_counts: Counter[str] = Counter()

        for row in rows:
            clean_value = row.get(field, "")
            raw_value = row.get(raw_field, "")
            flags = split_flags(row)
            matched_flags = flags & (field_specific_flags | GENERIC_FLAGS)

            if is_empty(clean_value):
                empty_clean += 1
                add_example(
                    examples,
                    field=field,
                    issue_kind="empty_clean_value",
                    row=row,
                    max_per_field_issue=max_examples,
                    counts=example_counts,
                )
            if is_empty(raw_value):
                empty_raw += 1
            if raw_value.strip() and not clean_value.strip():
                raw_present_clean_empty += 1
                add_example(
                    examples,
                    field=field,
                    issue_kind="raw_present_but_clean_empty",
                    row=row,
                    max_per_field_issue=max_examples,
                    counts=example_counts,
                )
            if clean_value.strip() and not raw_value.strip():
                clean_present_raw_empty += 1

            if matched_flags:
                flagged_rows += 1
                flag_counts.update(sorted(matched_flags))
                add_example(
                    examples,
                    field=field,
                    issue_kind="field_flag",
                    row=row,
                    max_per_field_issue=max_examples,
                    counts=example_counts,
                )
            if any("rescued" in flag or flag.endswith("_inherited") for flag in matched_flags):
                rescued_rows += 1
            if any(flag.endswith("_likely_printed_blank") for flag in matched_flags):
                likely_blank_rows += 1
            hard_flags = {
                flag
                for flag in matched_flags
                if flag.startswith("missing_")
                or flag.startswith("multi_value_")
                or flag.endswith("_ocr_noise")
                or flag.endswith("_needs_review")
                or flag in {"glued_shares_nominal_raw", "probable_row_fragment", "probable_split_identity_prefix"}
            }
            if hard_flags:
                review_rows += 1

        summaries.append(
            {
                "field": field,
                "total_rows": total_rows,
                "empty_clean": empty_clean,
                "empty_raw": empty_raw,
                "raw_present_clean_empty": raw_present_clean_empty,
                "clean_present_raw_empty": clean_present_raw_empty,
                "field_flagged_rows": flagged_rows,
                "field_review_rows": review_rows,
                "rescued_or_inherited_rows": rescued_rows,
                "likely_printed_blank_rows": likely_blank_rows,
                "top_field_flags": "; ".join(f"{flag}:{count}" for flag, count in flag_counts.most_common(8)),
            }
        )

    summaries.sort(
        key=lambda row: (
            int(row["field_review_rows"]),
            int(row["raw_present_clean_empty"]),
            int(row["empty_clean"]),
        ),
        reverse=True,
    )
    return summaries, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    cleaned_path = args.batch_dir / "combined_cleaned_candidate_rows.csv"
    rows = read_csv(cleaned_path)
    summaries, examples = build_diagnostics(rows, args.max_examples)

    write_csv(
        args.batch_dir / "combined_column_diagnostics.csv",
        summaries,
        [
            "field",
            "total_rows",
            "empty_clean",
            "empty_raw",
            "raw_present_clean_empty",
            "clean_present_raw_empty",
            "field_flagged_rows",
            "field_review_rows",
            "rescued_or_inherited_rows",
            "likely_printed_blank_rows",
            "top_field_flags",
        ],
    )
    write_csv(args.batch_dir / "combined_column_issue_examples.csv", examples, EXAMPLE_FIELDS)
    print(args.batch_dir / "combined_column_diagnostics.csv")
    print(args.batch_dir / "combined_column_issue_examples.csv")


if __name__ == "__main__":
    main()
