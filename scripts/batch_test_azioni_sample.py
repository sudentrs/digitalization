"""Run the azioni OCR/parse/clean workflow on a small sample of table crops.

This is a scale-up test, not the final production pipeline. It reuses the
one-page parser and cleaner so we can see which errors repeat across pages.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2

from clean_one_azioni_page import clean_row, is_large_integer, looks_like_nominal, parsed_numbers
from ocr_azioni_tables import DEFAULT_TESSDATA, find_tesseract, parse_tsv, validate_languages
from ocr_one_page_experiment import make_variants, run_tesseract
from parse_one_azioni_page_structured import (
    build_columns,
    draw_overlay,
    group_rows,
    parse_groups,
    row_grouping_diagnostics,
    shift_backup_words,
    write_csv,
)


DEFAULT_TABLE_DIR = Path("output/preprocess_1950_azioni/table_crops")
DEFAULT_OUTPUT = Path("output/batch_sample_azioni")

OCR_RUNS = [
    ("original_full", 12, "main"),
    ("original_full", 11, "full_backup"),
    ("numeric_columns_body", 12, "numeric_backup"),
    ("price_quantity_columns_body", 12, "price_backup"),
]

WORD_FIELDS = [
    "source_file",
    "date",
    "picture",
    "side",
    "zone_crop_file",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "conf",
    "left",
    "top",
    "right",
    "bottom",
    "width",
    "height",
    "text",
]

STRUCTURED_BASE_FIELDS = [
    "sample_id",
    "source_file",
    "row_index",
    "row_type",
    "row_role",
    "section",
    "top",
    "bottom",
    "word_top",
    "word_bottom",
    "left",
    "right",
    "word_count",
    "mean_conf",
    "raw_row_text",
]

RAW_COLS = [
    "capital_subscribed_raw",
    "shares_outstanding_raw",
    "nominal_value_raw",
    "godimento_raw",
    "cedola_date_raw",
    "importo_lordo_raw",
    "importo_acconto_raw",
    "importo_saldo_raw",
    "numero_raw",
    "compensation_price_raw",
    "issuer_name_raw",
    "price_min_raw",
    "price_max_raw",
    "price_close_raw",
    "quantity_traded_raw",
]

IDENTITY_COLS = [
    "capital_subscribed_raw",
    "shares_outstanding_raw",
    "nominal_value_raw",
]
IDENTITY_REVIEW_FLAGS = {
    "missing_capital",
    "missing_shares",
    "missing_nominal",
    "multi_value_shares_raw",
    "multi_value_nominal_raw",
    "glued_shares_nominal_raw",
}

CLEAN_FIELDS = [
    "sample_id",
    "source_file",
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
    *RAW_COLS,
]


def crop_id(path: Path) -> str:
    return path.stem.replace("_table", "")


def parse_crop_metadata(path: Path) -> dict[str, str]:
    match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<picture>\d+)_(?P<side>left|right)_table", path.stem)
    if not match:
        return {"date": "", "picture": "", "side": ""}
    return match.groupdict()


def choose_crops(table_dir: Path, sample_size: int) -> list[Path]:
    crops = sorted(table_dir.glob("*_table.jpg"))
    if sample_size <= 0:
        return crops
    return crops[:sample_size]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def sample_complete(sample_dir: Path) -> bool:
    return (sample_dir / "structured_rows.csv").exists() and (sample_dir / "cleaned_candidate_rows.csv").exists()


def flags_without(flags_text: str, remove: set[str]) -> list[str]:
    return [flag for flag in flags_text.split(";") if flag and flag not in remove]


def set_flags(row: dict[str, str], flags: list[str]) -> None:
    seen: set[str] = set()
    clean_flags = []
    for flag in flags:
        if flag and flag not in seen:
            clean_flags.append(flag)
            seen.add(flag)
    row["validation_flags"] = ";".join(clean_flags)


def words_text(words: list[dict[str, str]]) -> str:
    ordered = sorted(words, key=lambda row: (int(row["top"]), int(row["left"])))
    return " ".join(row["text"] for row in ordered).strip()


def needs_identity_cell_ocr(row: dict[str, str]) -> bool:
    clean = clean_row(row)
    flags = set(clean.get("validation_flags", "").split(";"))
    return bool(flags & IDENTITY_REVIEW_FLAGS)


def field_needs_identity_cell_ocr(row: dict[str, str], field: str) -> bool:
    clean = clean_row(row)
    flags = set(clean.get("validation_flags", "").split(";"))
    if field == "capital_subscribed_raw":
        return "missing_capital" in flags or not row.get(field, "")
    if field == "shares_outstanding_raw":
        return (
            "missing_shares" in flags
            or "multi_value_shares_raw" in flags
            or "glued_shares_nominal_raw" in flags
            or not row.get(field, "")
        )
    if field == "nominal_value_raw":
        return "missing_nominal" in flags or "multi_value_nominal_raw" in flags or not row.get(field, "")
    return False


def prepare_numeric_strip(crop) -> object:
    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    normalized = cv2.equalizeHist(upscaled)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return binary


def good_identity_ocr_text(field: str, text: str) -> bool:
    return bool(best_identity_ocr_value(field, text))


def best_identity_ocr_value(field: str, text: str) -> str:
    values = [value for value in parsed_numbers(text) if value and "." not in value]
    if not values:
        return ""
    alpha_count = len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE))
    if field in {"capital_subscribed_raw", "shares_outstanding_raw"}:
        large_values = [value for value in values if is_large_integer(value)]
        if not large_values:
            return ""
        if alpha_count > 10 and len(large_values) > 1:
            return ""
        return large_values[0]
    if field == "nominal_value_raw":
        nominal_values = [value for value in values if looks_like_nominal(value)]
        if not nominal_values:
            return ""
        if alpha_count > 8 and len(values) > 3:
            return ""
        return nominal_values[0]
    return ""


def run_identity_column_ocr(
    image_path: Path,
    sample_dir: Path,
    rows: list[dict[str, str]],
    columns: list[tuple[str, int, int]],
    tesseract: str,
    lang: str,
    tessdata_dir: Path | None,
) -> list[dict[str, str]]:
    flagged_rows = [row for row in rows if row.get("row_type") == "candidate_security_row" and needs_identity_cell_ocr(row)]
    if not flagged_rows:
        return rows

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read OCR image: {image_path}")
    h, w = image.shape
    col_lookup = {name: (x0, x1) for name, x0, x1 in columns}
    cell_dir = sample_dir / "identity_cell_ocr"
    cell_dir.mkdir(parents=True, exist_ok=True)

    backup_by_row_field: dict[tuple[str, str], str] = {}
    ocr_rows: list[dict[str, str]] = []
    cached_rows_path = cell_dir / "identity_cell_ocr_by_row.csv"
    if cached_rows_path.exists():
        for cached in read_csv(cached_rows_path):
            text = cached.get("text", "")
            if text:
                backup_by_row_field[(cached.get("row_index", ""), cached.get("field", ""))] = text
    else:
        for field in IDENTITY_COLS:
            if field not in col_lookup:
                continue
            x0, x1 = col_lookup[field]
            x0 = max(0, x0 - 8)
            x1 = min(w, x1 + 8)
            strip = image[:, x0:x1]
            prepared = prepare_numeric_strip(strip)
            strip_path = cell_dir / f"{field}.png"
            cv2.imwrite(str(strip_path), prepared)
            tsv = run_tesseract(tesseract, strip_path, psm=6, lang=lang, tessdata_dir=tessdata_dir)
            (cell_dir / f"{field}.tsv").write_text(tsv, encoding="utf-8")
            zone_row = {
                "source_file": image_path.name,
                "date": "",
                "picture": "",
                "side": "",
                "zone_crop_file": str(strip_path),
            }
            words = parse_tsv(tsv, scale=3.0, zone_row=zone_row)
            for word in words:
                word["left"] = str(int(word["left"]) + x0)
                word["right"] = str(int(word["right"]) + x0)
            for row in flagged_rows:
                top = int(float(row.get("top", "0")))
                bottom = int(float(row.get("bottom", "0")))
                pad = max(3, int((bottom - top) * 0.25))
                cell_words = [
                    word
                    for word in words
                    if int(word["bottom"]) >= top - pad and int(word["top"]) <= bottom + pad
                ]
                text = words_text(cell_words)
                if text:
                    key = (row.get("row_index", ""), field)
                    backup_by_row_field[key] = text
                    ocr_rows.append(
                        {
                            "row_index": row.get("row_index", ""),
                            "field": field,
                            "text": text,
                            "top": row.get("top", ""),
                            "bottom": row.get("bottom", ""),
                        }
                    )

        write_csv(cached_rows_path, ocr_rows, ["row_index", "field", "text", "top", "bottom"])

    for row in rows:
        if row.get("row_type") != "candidate_security_row":
            continue
        for field in IDENTITY_COLS:
            text = backup_by_row_field.get((row.get("row_index", ""), field), "")
            if not text:
                continue
            hint_field = f"{field}_backup_hint"
            source_field = f"{field}_backup_source"
            if field_needs_identity_cell_ocr(row, field):
                value = best_identity_ocr_value(field, text)
                if not value:
                    continue
                row[field] = value
                row[hint_field] = text
                row[source_field] = "identity_column_ocr"
    return rows


def weak_continuation_issuer(issuer: str) -> bool:
    issuer = (issuer or "").strip()
    if not issuer:
        return True
    if len(issuer) <= 4:
        return True
    lowered = issuer.lower()
    return lowered.startswith(("b)", "»", ")", "(", "privileg", "ordinar"))


def apply_row_context(cleaned: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fill continuation rows from the previous complete row in the same section."""
    previous_by_section: dict[str, dict[str, str]] = {}
    identity_fields = ["capital_subscribed", "shares_outstanding", "nominal_value"]

    for row in cleaned:
        section = row.get("section", "")
        flags = flags_without(row.get("validation_flags", ""), set())
        previous = previous_by_section.get(section)
        is_continuation = row.get("row_role") == "security_continuation_row"

        if is_continuation and previous:
            inherited_any = False
            if weak_continuation_issuer(row.get("issuer_name_clean", "")) and previous.get("issuer_name_clean"):
                current_issuer = row.get("issuer_name_clean", "").strip()
                if current_issuer and current_issuer not in previous["issuer_name_clean"]:
                    row["issuer_name_clean"] = f"{previous['issuer_name_clean']} {current_issuer}".strip()
                else:
                    row["issuer_name_clean"] = previous["issuer_name_clean"]
                flags = [flag for flag in flags if flag != "missing_issuer"]
                flags.append("issuer_inherited")
                inherited_any = True

            for field in identity_fields:
                if not row.get(field) and previous.get(field):
                    row[field] = previous[field]
                    flags = [flag for flag in flags if flag != f"missing_{field.split('_')[0]}"]
                    inherited_any = True
            if inherited_any:
                flags.append("identity_inherited_from_previous_row")

        if row.get("row_role") == "security_main_row" and row.get("issuer_name_clean"):
            previous_by_section[section] = row.copy()

        set_flags(row, flags)

    for index, row in enumerate(cleaned[:-1]):
        flags = flags_without(row.get("validation_flags", ""), set())
        if "missing_issuer" not in flags:
            continue
        next_row = cleaned[index + 1]
        if row.get("section", "") != next_row.get("section", ""):
            continue
        if not next_row.get("issuer_name_clean", ""):
            continue
        identity_count = sum(1 for field in identity_fields if row.get(field))
        if identity_count < 2:
            continue

        copied_any = False
        next_flags = flags_without(next_row.get("validation_flags", ""), set())
        for field in identity_fields:
            if row.get(field) and not next_row.get(field):
                next_row[field] = row[field]
                next_flags = [flag for flag in next_flags if flag != f"missing_{field.split('_')[0]}"]
                copied_any = True
        if copied_any:
            next_flags.append("identity_from_previous_split_prefix")
            set_flags(next_row, next_flags)

        row["row_role"] = "split_identity_prefix"
        row["issuer_name_clean"] = next_row["issuer_name_clean"]
        flags = [flag for flag in flags if flag != "missing_issuer"]
        flags.append("probable_split_identity_prefix")
        set_flags(row, flags)
    return cleaned


def summarize_sample(sample_id: str, source_file: str, sample_dir: Path) -> dict[str, str]:
    structured = read_csv(sample_dir / "structured_rows.csv")
    cleaned = read_csv(sample_dir / "cleaned_candidate_rows.csv")
    removed = read_csv(sample_dir / "row_grouping_removed_words.csv")
    candidates = [row for row in structured if row.get("row_type") == "candidate_security_row"]
    return {
        "sample_id": sample_id,
        "source_file": source_file,
        "structured_rows": str(len(structured)),
        "candidate_rows": str(len(candidates)),
        "cleaned_rows": str(len(cleaned)),
        "flagged_rows": str(sum(bool(row.get("validation_flags")) for row in cleaned)),
        "removed_row_artifacts": str(len(removed)),
        "median_word_height": "",
        "max_removed_word_height_threshold": "",
    }


def run_ocr_for_crop(
    crop_path: Path,
    sample_dir: Path,
    tesseract: str,
    lang: str,
    tessdata_dir: Path | None,
) -> dict[str, list[dict[str, str]]]:
    variants = make_variants(crop_path, sample_dir)
    metadata = parse_crop_metadata(crop_path)
    zone_row = {
        "source_file": crop_path.name,
        "date": metadata["date"],
        "picture": metadata["picture"],
        "side": metadata["side"],
        "zone_crop_file": str(crop_path),
    }

    words_by_role: dict[str, list[dict[str, str]]] = {}
    for variant, psm, role in OCR_RUNS:
        image_path = variants[variant]
        stem = f"{variant}_psm{psm}"
        tsv = run_tesseract(tesseract, image_path, psm=psm, lang=lang, tessdata_dir=tessdata_dir)
        raw_tsv_path = sample_dir / "raw_tsv" / f"{stem}.tsv"
        raw_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tsv_path.write_text(tsv, encoding="utf-8")
        words = parse_tsv(tsv, scale=1.0, zone_row=zone_row)
        write_csv(sample_dir / "words" / f"{stem}_words.csv", words, WORD_FIELDS)
        words_by_role[role] = words
    return words_by_role


def parse_and_clean_crop(
    crop_path: Path,
    sample_dir: Path,
    words_by_role: dict[str, list[dict[str, str]]],
    row_method: str,
    tesseract: str,
    lang: str,
    tessdata_dir: Path | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    image_path = sample_dir / "images" / "original_full.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read OCR image: {image_path}")
    h, w = image.shape
    body_top = int(h * 0.075)
    body_bottom = int(h * 0.975)
    body_y0 = int(h * 0.075)

    full_backup = words_by_role.get("full_backup", [])
    numeric_backup = shift_backup_words(words_by_role.get("numeric_backup", []), x_offset=0, y_offset=body_y0)
    price_backup = shift_backup_words(
        words_by_role.get("price_backup", []),
        x_offset=int(w * 0.78),
        y_offset=body_y0,
    )

    columns, detected_rules = build_columns(image_path, body_top, body_bottom, words_by_role["main"])
    removed_words, median_height, max_height = row_grouping_diagnostics(
        words_by_role["main"],
        body_top,
        body_bottom,
    )
    groups = group_rows(
        words_by_role["main"],
        body_top=body_top,
        body_bottom=body_bottom,
        columns=columns,
        method=row_method,
    )
    structured = parse_groups(
        groups,
        columns,
        full_backup,
        numeric_backup,
        price_backup,
        body_top,
        body_bottom,
    )
    structured = run_identity_column_ocr(
        image_path,
        sample_dir,
        structured,
        columns,
        tesseract,
        lang,
        tessdata_dir,
    )

    sid = crop_id(crop_path)
    for row in structured:
        row["sample_id"] = sid
        row["source_file"] = crop_path.name

    structured_fields = (
        STRUCTURED_BASE_FIELDS
        + RAW_COLS
        + [f"{name}_source" for name in RAW_COLS]
        + [f"{name}_backup_hint" for name in RAW_COLS]
        + [f"{name}_backup_source" for name in RAW_COLS]
    )
    write_csv(sample_dir / "structured_rows.csv", structured, structured_fields)
    candidates = [row for row in structured if row["row_type"] == "candidate_security_row"]
    write_csv(sample_dir / "candidate_security_rows.csv", candidates, structured_fields)
    write_csv(
        sample_dir / "detected_columns.csv",
        [{"column": name, "x0": str(start), "x1": str(end)} for name, start, end in columns],
        ["column", "x0", "x1"],
    )
    write_csv(
        sample_dir / "row_grouping_removed_words.csv",
        removed_words,
        ["text", "conf", "left", "top", "right", "bottom", "width", "height", "reason"],
    )
    draw_overlay(image_path, structured, columns, detected_rules, sample_dir / "structured_overlay.jpg")

    cleaned = []
    for row in candidates:
        clean = clean_row(row)
        clean["sample_id"] = sid
        clean["source_file"] = crop_path.name
        cleaned.append(clean)
    cleaned = apply_row_context(cleaned)
    write_csv(sample_dir / "cleaned_candidate_rows.csv", cleaned, CLEAN_FIELDS)

    summary = {
        "sample_id": sid,
        "source_file": crop_path.name,
        "structured_rows": str(len(structured)),
        "candidate_rows": str(len(candidates)),
        "cleaned_rows": str(len(cleaned)),
        "flagged_rows": str(sum(bool(row["validation_flags"]) for row in cleaned)),
        "removed_row_artifacts": str(len(removed_words)),
        "median_word_height": f"{median_height:.1f}",
        "max_removed_word_height_threshold": str(max_height),
    }
    return structured, cleaned, [summary]


def write_flag_summary(output_dir: Path, cleaned_rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in cleaned_rows:
        for flag in row.get("validation_flags", "").split(";"):
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    rows = [{"flag": flag, "count": str(count)} for flag, count in sorted(counts.items())]
    write_csv(output_dir / "combined_validation_flag_summary.csv", rows, ["flag", "count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-test azioni parsing/cleaning on a small sample.")
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--tesseract", default=None)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="ita+eng")
    parser.add_argument("--row-method", choices=["center", "issuer_anchor"], default="center")
    parser.add_argument("--resume", action="store_true", help="Reuse completed sample folders instead of rerunning them.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before running.")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Only combine already completed sample folders; do not run OCR.",
    )
    parser.add_argument(
        "--reclean-existing",
        action="store_true",
        help="Rebuild cleaned CSVs from existing structured rows before combining.",
    )
    parser.add_argument(
        "--reparse-existing",
        action="store_true",
        help="Rebuild structured and cleaned CSVs from existing OCR word CSVs.",
    )
    args = parser.parse_args()

    tesseract = find_tesseract(args.tesseract)
    validate_languages(tesseract, args.lang, args.tessdata_dir)

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crops = choose_crops(args.table_dir, args.sample_size)
    all_structured: list[dict[str, str]] = []
    all_cleaned: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []

    for index, crop_path in enumerate(crops, start=1):
        sid = crop_id(crop_path)
        print(f"[{index}/{len(crops)}] {sid}")
        sample_dir = args.output_dir / "samples" / sid
        sample_dir.mkdir(parents=True, exist_ok=True)

        word_files_exist = all(
            (sample_dir / "words" / name).exists()
            for name in [
                "original_full_psm12_words.csv",
                "original_full_psm11_words.csv",
                "numeric_columns_body_psm12_words.csv",
                "price_quantity_columns_body_psm12_words.csv",
            ]
        )
        if args.reparse_existing and word_files_exist:
            words_by_role = {
                "main": read_csv(sample_dir / "words" / "original_full_psm12_words.csv"),
                "full_backup": read_csv(sample_dir / "words" / "original_full_psm11_words.csv"),
                "numeric_backup": read_csv(sample_dir / "words" / "numeric_columns_body_psm12_words.csv"),
                "price_backup": read_csv(sample_dir / "words" / "price_quantity_columns_body_psm12_words.csv"),
            }
            structured, cleaned, summary = parse_and_clean_crop(
                crop_path,
                sample_dir,
                words_by_role,
                args.row_method,
                tesseract,
                args.lang,
                args.tessdata_dir,
            )
        elif sample_complete(sample_dir) and (args.resume or args.summarize_existing or args.reclean_existing):
            structured = read_csv(sample_dir / "structured_rows.csv")
            if args.reclean_existing:
                candidates = [row for row in structured if row.get("row_type") == "candidate_security_row"]
                cleaned = []
                for row in candidates:
                    clean = clean_row(row)
                    clean["sample_id"] = sid
                    clean["source_file"] = crop_path.name
                    cleaned.append(clean)
                cleaned = apply_row_context(cleaned)
                write_csv(sample_dir / "cleaned_candidate_rows.csv", cleaned, CLEAN_FIELDS)
            else:
                cleaned = read_csv(sample_dir / "cleaned_candidate_rows.csv")
            summary = [summarize_sample(sid, crop_path.name, sample_dir)]
        elif args.summarize_existing:
            print(f"  skipped incomplete sample: {sid}")
            continue
        else:
            words_by_role = run_ocr_for_crop(crop_path, sample_dir, tesseract, args.lang, args.tessdata_dir)
            structured, cleaned, summary = parse_and_clean_crop(
                crop_path,
                sample_dir,
                words_by_role,
                args.row_method,
                tesseract,
                args.lang,
                args.tessdata_dir,
            )

        all_structured.extend(structured)
        all_cleaned.extend(cleaned)
        summaries.extend(summary)

    structured_fields = (
        STRUCTURED_BASE_FIELDS
        + RAW_COLS
        + [f"{name}_source" for name in RAW_COLS]
        + [f"{name}_backup_hint" for name in RAW_COLS]
        + [f"{name}_backup_source" for name in RAW_COLS]
    )
    write_csv(args.output_dir / "combined_structured_rows.csv", all_structured, structured_fields)
    write_csv(args.output_dir / "combined_cleaned_candidate_rows.csv", all_cleaned, CLEAN_FIELDS)
    if summaries:
        write_csv(args.output_dir / "sample_summary.csv", summaries, list(summaries[0].keys()))
    write_flag_summary(args.output_dir, all_cleaned)

    readme = [
        "# Batch azioni sample test",
        "",
        "This is a small-scale validation run of the OCR, row/column parser, and cleaner.",
        "",
        f"Requested sample size: {len(crops)}",
        f"Completed samples summarized: {len(summaries)}",
        f"Row method: `{args.row_method}`",
        "",
        "Top-level outputs:",
        "- `sample_summary.csv`: per-page row and flag counts.",
        "- `combined_structured_rows.csv`: raw structured parse across sample pages.",
        "- `combined_cleaned_candidate_rows.csv`: normalized draft fields across sample pages.",
        "- `combined_validation_flag_summary.csv`: repeated issue types across the sample.",
        "- `samples/<sample_id>/`: per-page words, overlays, structured rows, and cleaned rows.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Wrote: {args.output_dir}")
    print(f"Structured rows: {len(all_structured)}")
    print(f"Cleaned candidate rows: {len(all_cleaned)}")


if __name__ == "__main__":
    main()
