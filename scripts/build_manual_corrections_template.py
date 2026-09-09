from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_BATCH_DIR = Path("output/batch_30_azioni")


FIELDS = [
    "sample_id",
    "row_index",
    "section",
    "row_role",
    "problem_category",
    "priority_reason",
    "issuer_name_clean",
    "capital_subscribed",
    "shares_outstanding",
    "nominal_value",
    "price_close",
    "quantity_traded",
    "validation_flags",
    "raw_row_text",
    "snippet_path",
    "overlay_path",
    "original_full_path",
    "review_action",
    "corrected_capital_subscribed",
    "corrected_shares_outstanding",
    "corrected_nominal_value",
    "corrected_issuer_name",
    "merge_target_row_index",
    "review_notes",
]


def load_review_rows(batch_dir: Path) -> list[dict[str, str]]:
    source_path = batch_dir / "high_priority_review_packet" / "high_priority_review_packet_index.csv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def packet_path(batch_dir: Path, relative_path: str) -> str:
    if not relative_path:
        return ""
    return str((batch_dir / "high_priority_review_packet" / relative_path).resolve())


def build_rows(batch_dir: Path, review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in review_rows:
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "row_index": row.get("row_index", ""),
                "section": row.get("section", ""),
                "row_role": row.get("row_role", ""),
                "problem_category": row.get("problem_category", ""),
                "priority_reason": row.get("priority_reason", ""),
                "issuer_name_clean": row.get("issuer_name_clean", ""),
                "capital_subscribed": row.get("capital_subscribed", ""),
                "shares_outstanding": row.get("shares_outstanding", ""),
                "nominal_value": row.get("nominal_value", ""),
                "price_close": row.get("price_close", ""),
                "quantity_traded": row.get("quantity_traded", ""),
                "validation_flags": row.get("validation_flags", ""),
                "raw_row_text": row.get("raw_row_text", ""),
                "snippet_path": packet_path(batch_dir, row.get("snippet", "")),
                "overlay_path": packet_path(batch_dir, row.get("overlay", "")),
                "original_full_path": packet_path(batch_dir, row.get("original_full", "")),
                "review_action": "manual_review",
                "corrected_capital_subscribed": "",
                "corrected_shares_outstanding": "",
                "corrected_nominal_value": "",
                "corrected_issuer_name": "",
                "merge_target_row_index": "",
                "review_notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the manual-correction CSV files for high-priority azioni rows."
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument(
        "--overwrite-working",
        action="store_true",
        help="Overwrite manual_corrections.csv if it already exists.",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir
    rows = build_rows(batch_dir, load_review_rows(batch_dir))
    template_path = batch_dir / "manual_corrections_template.csv"
    working_path = batch_dir / "manual_corrections.csv"

    write_csv(template_path, rows)
    if args.overwrite_working or not working_path.exists():
        write_csv(working_path, rows)

    print(f"Wrote {len(rows)} rows to {template_path}")
    print(f"Working correction file: {working_path}")


if __name__ == "__main__":
    main()
