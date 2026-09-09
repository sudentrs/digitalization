from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_BATCH_DIR = Path("output/batch_30_azioni")
CORRECTABLE_FIELDS = {
    "corrected_capital_subscribed": ("capital_subscribed", "missing_capital"),
    "corrected_shares_outstanding": ("shares_outstanding", "missing_shares"),
    "corrected_nominal_value": ("nominal_value", "missing_nominal"),
    "corrected_issuer_name": ("issuer_name_clean", "missing_issuer"),
}
VALID_ACTIONS = {"", "manual_review", "keep", "exclude", "merge"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("sample_id", ""), row.get("row_index", ""))


def flag_set(flags: str) -> list[str]:
    return [flag for flag in (flags or "").split(";") if flag]


def update_flags(row: dict[str, str], remove: set[str], add: set[str]) -> None:
    flags = [flag for flag in flag_set(row.get("validation_flags", "")) if flag not in remove]
    for flag in sorted(add):
        if flag not in flags:
            flags.append(flag)
    row["validation_flags"] = ";".join(flags)


def correction_value(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    return value.strip() if value is not None else ""


def apply_correction(
    cleaned_row: dict[str, str], correction: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    updated = dict(cleaned_row)
    action = correction.get("review_action", "").strip().lower()
    log = {
        "sample_id": cleaned_row.get("sample_id", ""),
        "row_index": cleaned_row.get("row_index", ""),
        "review_action": action,
        "status": "skipped",
        "changed_fields": "",
        "notes": correction.get("review_notes", ""),
    }

    if action not in VALID_ACTIONS:
        log["status"] = "invalid_action"
        log["notes"] = f"Unknown review_action: {action}"
        return updated, log

    if action in {"", "manual_review"}:
        return updated, log

    changed_fields: list[str] = []
    flags_to_remove: set[str] = set()
    flags_to_add: set[str] = set()

    if action == "exclude":
        updated["row_role"] = "manual_exclude"
        flags_to_add.add("manual_exclude")
        changed_fields.append("row_role")

    if action == "merge":
        target = correction_value(correction, "merge_target_row_index")
        updated["row_role"] = "manual_merge_requested"
        flags_to_add.add("manual_merge_requested")
        if target:
            flags_to_add.add(f"manual_merge_target_{target}")
        changed_fields.append("row_role")

    if action in {"keep", "merge"}:
        for correction_field, (target_field, missing_flag) in CORRECTABLE_FIELDS.items():
            value = correction_value(correction, correction_field)
            if not value:
                continue
            if updated.get(target_field, "") != value:
                updated[target_field] = value
                changed_fields.append(target_field)
                flags_to_add.add(f"manual_corrected_{target_field}")
            flags_to_remove.add(missing_flag)

    if changed_fields:
        flags_to_add.add("manual_corrected")
        update_flags(updated, flags_to_remove, flags_to_add)
        log["status"] = "applied"
        log["changed_fields"] = ";".join(changed_fields)

    return updated, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual high-priority row corrections.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="CSV file to apply. Defaults to <batch-dir>/manual_corrections.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <batch-dir>/corrected.",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir
    corrections_path = args.corrections or (batch_dir / "manual_corrections.csv")
    output_dir = args.output_dir or (batch_dir / "corrected")
    cleaned_path = batch_dir / "combined_cleaned_candidate_rows.csv"

    cleaned_fields, cleaned_rows = read_csv(cleaned_path)
    _, correction_rows = read_csv(corrections_path)
    corrections = {row_key(row): row for row in correction_rows}

    output_rows: list[dict[str, str]] = []
    log_rows: list[dict[str, str]] = []
    for row in cleaned_rows:
        correction = corrections.get(row_key(row))
        if not correction:
            output_rows.append(row)
            continue
        updated, log = apply_correction(row, correction)
        output_rows.append(updated)
        log_rows.append(log)

    corrected_path = output_dir / "combined_cleaned_candidate_rows_corrected.csv"
    log_path = output_dir / "manual_corrections_applied_log.csv"
    write_csv(corrected_path, cleaned_fields, output_rows)
    write_csv(
        log_path,
        ["sample_id", "row_index", "review_action", "status", "changed_fields", "notes"],
        log_rows,
    )

    applied = sum(1 for row in log_rows if row["status"] == "applied")
    invalid = sum(1 for row in log_rows if row["status"] == "invalid_action")
    print(f"Rows in cleaned file: {len(cleaned_rows)}")
    print(f"Correction rows matched: {len(log_rows)}")
    print(f"Corrections applied: {applied}")
    print(f"Invalid actions: {invalid}")
    print(f"Wrote {corrected_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
