"""Run a first-pass multi-page-type OCR pipeline.

This script generalizes the project workflow beyond regular TITOLI AZIONARI
pages. It routes every classified page side through a common OCR/layout layer
and records which page types are ready for specialized structured parsing.

The goal is not to force all document types into the azioni schema. Instead it
creates consistent OCR text, word coordinates, line coordinates, heading
signals, diagnostics, and review flags for each page type.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ocr_azioni_tables import DEFAULT_TESSDATA, find_tesseract, parse_tsv, preprocess_for_ocr, validate_languages
from ocr_one_page_experiment import remove_table_rules, run_tesseract
from preprocess_scans import crop_half_pages, parse_scan_name, write_image


DEFAULT_LABELS = Path("output/preprocess_1950/page_type_labels.csv")
DEFAULT_PREPROCESS_DIR = Path("output/preprocess_1950")
DEFAULT_INPUT_DIR = Path("1950")
DEFAULT_OUTPUT = Path("output/batch_all_page_types")

WORD_FIELDS = [
    "sample_id",
    "page_type",
    "source_file",
    "date",
    "picture",
    "side",
    "crop_file",
    "ocr_variant",
    "psm",
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

LINE_FIELDS = [
    "sample_id",
    "page_type",
    "source_file",
    "date",
    "picture",
    "side",
    "crop_file",
    "ocr_variant",
    "psm",
    "block_num",
    "par_num",
    "line_num",
    "left",
    "top",
    "right",
    "bottom",
    "mean_conf",
    "text",
]

PAGE_TYPES = {
    "regular_azioni": {
        "status": "specialized_parser_available",
        "strategy": "regular ruled equity table; use existing azioni parser for final structured data",
        "psm": [12, 11],
    },
    "regular_obbligazioni": {
        "status": "generic_ocr_ready",
        "strategy": "regular ruled bond table; next needs an obbligazioni schema/parser",
        "psm": [12, 11],
    },
    "cover_valori_stato": {
        "status": "generic_ocr_ready",
        "strategy": "cover/title page with VALORI DI STATO table; next needs a state-securities schema/parser",
        "psm": [12, 11],
    },
    "dividendi_in_pagamento": {
        "status": "generic_ocr_ready",
        "strategy": "dividend-payment list; likely parsed as a simpler line/list table",
        "psm": [6, 11],
    },
    "comunicati_notizie": {
        "status": "generic_ocr_ready",
        "strategy": "notices page; preserve paragraph text and local tables rather than forcing row schema",
        "psm": [6, 11],
    },
    "special_market_lists": {
        "status": "generic_ocr_ready",
        "strategy": "mixed lists and small tables; needs sub-type detection before structured extraction",
        "psm": [6, 11],
    },
    "advertisement_notice": {
        "status": "generic_ocr_ready",
        "strategy": "long company notice; preserve paragraph text",
        "psm": [6, 11],
    },
    "blank_or_divider": {
        "status": "skip_or_minimal_ocr",
        "strategy": "blank/divider page; keep diagnostics, normally skip structured extraction",
        "psm": [11],
    },
    "unknown_review": {
        "status": "needs_page_type_review",
        "strategy": "unknown page type; OCR only and review",
        "psm": [11],
    },
}

HEADING_PATTERNS = {
    "regular_azioni": [r"titoli.*azionari|azionari.*titoli"],
    "regular_obbligazioni": [r"obbligazioni", r"valori.*stato|stato.*valori"],
    "cover_valori_stato": [r"listino.*ufficiale", r"valori.*stato|stato.*valori", r"borsa.*milano|milano.*borsa"],
    "dividendi_in_pagamento": [r"dividendi|dividendo", r"pagamento|pagamenti"],
    "comunicati_notizie": [r"comunicati|comunicat", r"notizie|notiz", r"cronache.*societarie|societarie.*cronache"],
    "special_market_lists": [
        r"mercato.*ristretto|ristretto.*mercato",
        r"altre.*piazze|piazze.*altre",
        r"prossime.*assemblee|assemblee.*prossime",
        r"listino.*secondo|secondo.*listino",
        r"ordine.*chiamata|chiamata.*ordine",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return -1.0


def sample_id(row: dict[str, str]) -> str:
    return f"{row['date']}_{row['picture']}_{row['side']}"


def normalize_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "Ã ": "a",
        "Ã¨": "e",
        "Ã©": "e",
        "Ã¬": "i",
        "Ã²": "o",
        "Ã¹": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def choose_rows(
    rows: list[dict[str, str]],
    page_types: set[str],
    sample_per_type: int,
    limit: int | None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        page_type = row["page_type"]
        if page_types and page_type not in page_types:
            continue
        if sample_per_type > 0 and counts[page_type] >= sample_per_type:
            continue
        selected.append(row)
        counts[page_type] += 1
        if limit is not None and len(selected) >= limit:
            break
    return selected


def resolve_existing_crop(row: dict[str, str], preprocess_dir: Path) -> Path | None:
    crop_file = row.get("crop_file", "")
    if crop_file:
        candidate = preprocess_dir / crop_file
        if candidate.exists():
            return candidate

    # During azioni development, only azioni page crops were preserved in a
    # type-specific folder. Use them when available.
    azioni_candidate = Path("output/preprocess_1950_azioni/page_crops") / f"{sample_id(row)}.jpg"
    if azioni_candidate.exists():
        return azioni_candidate
    return None


def regenerate_crop(row: dict[str, str], input_dir: Path, output_dir: Path, gutter_overlap: int, deskew: bool) -> Path:
    source_path = input_dir / row["source_file"]
    image = cv2.imread(str(source_path))
    if image is None:
        raise OSError(f"Could not read source scan: {source_path}")

    pages = crop_half_pages(image, gutter_overlap=gutter_overlap, deskew=deskew)
    wanted_side = row["side"]
    page = next((item for item in pages if item.side == wanted_side), None)
    if page is None:
        raise OSError(f"Could not regenerate {wanted_side} page for {source_path}")

    crop_path = output_dir / "page_crops" / f"{sample_id(row)}.jpg"
    write_image(crop_path, page.crop)
    return crop_path


def prepare_page_variants(crop_path: Path, sample_dir: Path) -> dict[str, tuple[Path, float]]:
    image = cv2.imread(str(crop_path))
    if image is None:
        raise OSError(f"Could not read crop: {crop_path}")
    normalized, scale = preprocess_for_ocr(image)
    line_removed, rule_mask = remove_table_rules(normalized)

    image_dir = sample_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = image_dir / "normalized_full.png"
    line_removed_path = image_dir / "line_removed_full.png"
    rule_mask_path = image_dir / "detected_rule_mask.png"
    cv2.imwrite(str(normalized_path), normalized)
    cv2.imwrite(str(line_removed_path), line_removed)
    cv2.imwrite(str(rule_mask_path), rule_mask)
    return {
        "normalized_full": (normalized_path, scale),
        "line_removed_full": (line_removed_path, scale),
    }


def enrich_words(
    words: list[dict[str, str]],
    row: dict[str, str],
    crop_path: Path,
    variant: str,
    psm: int,
) -> list[dict[str, str]]:
    sid = sample_id(row)
    enriched = []
    for word in words:
        copy = dict(word)
        copy.update(
            {
                "sample_id": sid,
                "page_type": row["page_type"],
                "source_file": row["source_file"],
                "date": row["date"],
                "picture": row["picture"],
                "side": row["side"],
                "crop_file": str(crop_path),
                "ocr_variant": variant,
                "psm": str(psm),
            }
        )
        enriched.append(copy)
    return enriched


def words_to_lines(words: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for word in words:
        key = (word["ocr_variant"], word["block_num"], word["par_num"], word["line_num"])
        grouped[key].append(word)

    lines: list[dict[str, str]] = []
    for (variant, block, par, line), group in grouped.items():
        group = sorted(group, key=lambda item: (int(item["left"]), int(item["top"])))
        confs = [parse_float(item["conf"]) for item in group if parse_float(item["conf"]) >= 0]
        first = group[0]
        lines.append(
            {
                "sample_id": first["sample_id"],
                "page_type": first["page_type"],
                "source_file": first["source_file"],
                "date": first["date"],
                "picture": first["picture"],
                "side": first["side"],
                "crop_file": first["crop_file"],
                "ocr_variant": variant,
                "psm": first["psm"],
                "block_num": block,
                "par_num": par,
                "line_num": line,
                "left": str(min(int(item["left"]) for item in group)),
                "top": str(min(int(item["top"]) for item in group)),
                "right": str(max(int(item["right"]) for item in group)),
                "bottom": str(max(int(item["bottom"]) for item in group)),
                "mean_conf": f"{(sum(confs) / len(confs)):.2f}" if confs else "",
                "text": " ".join(item["text"] for item in group),
            }
        )
    return sorted(lines, key=lambda item: (item["ocr_variant"], int(item["top"]), int(item["left"])))


def words_to_plain_text(words: list[dict[str, str]]) -> str:
    ordered = sorted(words, key=lambda item: (int(item["top"]), int(item["left"])))
    return " ".join(item["text"] for item in ordered)


def detect_headings(row: dict[str, str], text: str) -> list[dict[str, str]]:
    normalized = normalize_text(text)
    patterns = HEADING_PATTERNS.get(row["page_type"], [])
    hits = []
    for pattern in patterns:
        if re.search(pattern, normalized):
            hits.append(
                {
                    "sample_id": sample_id(row),
                    "page_type": row["page_type"],
                    "heading_pattern": pattern,
                    "matched": "yes",
                }
            )
    return hits


def draw_line_overlay(crop_path: Path, lines: list[dict[str, str]], output_path: Path) -> None:
    image = cv2.imread(str(crop_path))
    if image is None:
        raise OSError(f"Could not read crop for overlay: {crop_path}")
    for line in lines:
        if line["ocr_variant"] != "normalized_full":
            continue
        left = int(line["left"])
        top = int(line["top"])
        right = int(line["right"])
        bottom = int(line["bottom"])
        color = (0, 170, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def page_diagnostics(row: dict[str, str], words: list[dict[str, str]], headings: list[dict[str, str]]) -> dict[str, str]:
    confs = [parse_float(word["conf"]) for word in words if parse_float(word["conf"]) >= 0]
    texts = [word["text"] for word in words]
    numeric_count = sum(bool(re.search(r"\d", text)) for text in texts)
    alpha_count = sum(bool(re.search(r"[^\W\d_]", text, flags=re.UNICODE)) for text in texts)
    noise_count = sum(bool(re.match(r"^[\W_]+$", text.strip())) for text in texts)
    flags = []
    if row.get("needs_review") == "yes":
        flags.append("page_type_needs_review")
    if row["page_type"] == "blank_or_divider":
        flags.append("skip_structured_extraction")
    if len(words) < 30 and row["page_type"] != "blank_or_divider":
        flags.append("low_word_count")
    if confs and (sum(confs) / len(confs)) < 45:
        flags.append("low_mean_ocr_confidence")
    if not headings and row["page_type"] in HEADING_PATTERNS:
        flags.append("expected_heading_not_detected")

    return {
        "sample_id": sample_id(row),
        "source_file": row["source_file"],
        "date": row["date"],
        "picture": row["picture"],
        "side": row["side"],
        "page_type": row["page_type"],
        "classification_confidence": row.get("confidence", ""),
        "strategy_status": PAGE_TYPES.get(row["page_type"], PAGE_TYPES["unknown_review"])["status"],
        "word_count": str(len(words)),
        "line_count": "",
        "mean_conf": f"{(sum(confs) / len(confs)):.2f}" if confs else "",
        "numeric_token_count": str(numeric_count),
        "alpha_token_count": str(alpha_count),
        "noise_token_count": str(noise_count),
        "heading_hits": str(len(headings)),
        "review_flags": ";".join(flags),
    }


def process_page(
    row: dict[str, str],
    args: argparse.Namespace,
    tesseract: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    sid = sample_id(row)
    page_type = row["page_type"]
    sample_dir = args.output_dir / page_type / "samples" / sid
    sample_dir.mkdir(parents=True, exist_ok=True)

    crop_path = resolve_existing_crop(row, args.preprocess_dir)
    if crop_path is None:
        crop_path = regenerate_crop(row, args.input_dir, args.output_dir, args.gutter_overlap, args.deskew)
    shutil.copy2(crop_path, sample_dir / "page_crop.jpg")

    variants = prepare_page_variants(crop_path, sample_dir)
    if args.extra_ocr:
        variant_items = list(variants.items())
        psm_values = PAGE_TYPES.get(page_type, PAGE_TYPES["unknown_review"])["psm"]
    else:
        variant_items = [("normalized_full", variants["normalized_full"])]
        psm_values = [PAGE_TYPES.get(page_type, PAGE_TYPES["unknown_review"])["psm"][0]]
    all_words: list[dict[str, str]] = []
    all_lines: list[dict[str, str]] = []
    all_headings: list[dict[str, str]] = []
    primary_text = ""

    for variant_name, (variant_path, scale) in variant_items:
        if variant_name == "line_removed_full" and page_type in {
            "comunicati_notizie",
            "advertisement_notice",
            "blank_or_divider",
        }:
            continue
        for psm in psm_values:
            tsv_text = run_tesseract(tesseract, variant_path, psm=psm, lang=args.lang, tessdata_dir=args.tessdata_dir)
            raw_tsv_path = sample_dir / "raw_tsv" / f"{variant_name}_psm{psm}.tsv"
            raw_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            raw_tsv_path.write_text(tsv_text, encoding="utf-8")
            zone_row = {
                "source_file": row["source_file"],
                "date": row["date"],
                "picture": row["picture"],
                "side": row["side"],
                "zone_crop_file": str(crop_path),
            }
            words = parse_tsv(tsv_text, scale=scale, zone_row=zone_row)
            words = enrich_words(words, row, crop_path, variant_name, psm)
            write_csv(sample_dir / "words" / f"{variant_name}_psm{psm}_words.csv", words, WORD_FIELDS)
            lines = words_to_lines(words)
            write_csv(sample_dir / "lines" / f"{variant_name}_psm{psm}_lines.csv", lines, LINE_FIELDS)

            text = words_to_plain_text(words)
            text_path = sample_dir / "text" / f"{variant_name}_psm{psm}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")

            if variant_name == "normalized_full" and psm == psm_values[0]:
                primary_text = text
                all_headings = detect_headings(row, text)
                draw_line_overlay(crop_path, lines, sample_dir / "line_overlay.jpg")

            all_words.extend(words)
            all_lines.extend(lines)

    write_csv(sample_dir / "ocr_words.csv", all_words, WORD_FIELDS)
    write_csv(sample_dir / "ocr_lines.csv", all_lines, LINE_FIELDS)
    write_csv(sample_dir / "detected_headings.csv", all_headings, ["sample_id", "page_type", "heading_pattern", "matched"])
    (sample_dir / "primary_text.txt").write_text(primary_text, encoding="utf-8")

    diagnostics = page_diagnostics(row, all_words, all_headings)
    diagnostics["line_count"] = str(len(all_lines))
    write_csv(sample_dir / "page_diagnostics.csv", [diagnostics], list(diagnostics.keys()))
    return all_words, all_lines, all_headings, diagnostics


def write_readme(output_dir: Path, selected_rows: list[dict[str, str]], diagnostics: list[dict[str, str]]) -> None:
    type_counts = Counter(row["page_type"] for row in selected_rows)
    flag_counts: Counter[str] = Counter()
    for row in diagnostics:
        for flag in row.get("review_flags", "").split(";"):
            if flag:
                flag_counts[flag] += 1

    lines = [
        "# Multi-Type OCR Pipeline Output",
        "",
        "This folder is the first shared pipeline layer for all classified page types.",
        "It does not force every page into the azioni table schema.",
        "",
        "## What It Produces",
        "",
        "- `page_type_processing_manifest.csv`: one row per processed page side.",
        "- `combined_ocr_words.csv`: Tesseract word-level OCR with coordinates.",
        "- `combined_ocr_lines.csv`: line-level OCR with coordinates.",
        "- `combined_page_diagnostics.csv`: OCR and review diagnostics.",
        "- `<page_type>/samples/<sample_id>/`: per-page images, OCR text, TSV files, overlays, and diagnostics.",
        "",
        "## Page-Type Strategy",
        "",
        "| Page type | Status | Strategy |",
        "|---|---|---|",
    ]
    for page_type, config in PAGE_TYPES.items():
        lines.append(f"| `{page_type}` | `{config['status']}` | {config['strategy']} |")

    lines.extend(["", "## Processed Counts", ""])
    for page_type, count in sorted(type_counts.items()):
        lines.append(f"- `{page_type}`: {count}")

    lines.extend(["", "## Review Flag Counts", ""])
    if flag_counts:
        for flag, count in sorted(flag_counts.items()):
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- No review flags in this run.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`regular_azioni` already has a specialized downstream parser from the earlier work.",
            "The other page types now have consistent OCR and layout outputs, which is the necessary first step before writing separate parsers for `regular_obbligazioni`, `dividendi_in_pagamento`, `cover_valori_stato`, and the mixed notice/list pages.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all classified page types through shared OCR/layout outputs.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--preprocess-dir", type=Path, default=DEFAULT_PREPROCESS_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tesseract", default=None)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="ita+eng")
    parser.add_argument("--page-types", nargs="*", default=[], help="Optional page types to process.")
    parser.add_argument("--sample-per-type", type=int, default=2, help="0 means all pages of each type.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gutter-overlap", type=int, default=60)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--extra-ocr",
        action="store_true",
        help="Run slower backup OCR variants/PSMs. Default is one fast normalized OCR pass per page.",
    )
    args = parser.parse_args()

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tesseract = find_tesseract(args.tesseract)
    validate_languages(tesseract, args.lang, args.tessdata_dir)

    labels = read_csv(args.labels)
    selected = choose_rows(labels, set(args.page_types), args.sample_per_type, args.limit)
    all_words: list[dict[str, str]] = []
    all_lines: list[dict[str, str]] = []
    all_headings: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for index, row in enumerate(selected, start=1):
        sid = sample_id(row)
        print(f"[{index}/{len(selected)}] {sid} {row['page_type']}", flush=True)
        try:
            words, lines, headings, diag = process_page(row, args, tesseract)
            all_words.extend(words)
            all_lines.extend(lines)
            all_headings.extend(headings)
            diagnostics.append(diag)
            status = "processed"
            error = ""
        except Exception as exc:  # Keep batch runs auditable.
            status = "failed"
            error = str(exc)
            diag = {
                "sample_id": sid,
                "source_file": row["source_file"],
                "date": row["date"],
                "picture": row["picture"],
                "side": row["side"],
                "page_type": row["page_type"],
                "classification_confidence": row.get("confidence", ""),
                "strategy_status": PAGE_TYPES.get(row["page_type"], PAGE_TYPES["unknown_review"])["status"],
                "word_count": "0",
                "line_count": "0",
                "mean_conf": "",
                "numeric_token_count": "0",
                "alpha_token_count": "0",
                "noise_token_count": "0",
                "heading_hits": "0",
                "review_flags": "processing_failed",
            }
            diagnostics.append(diag)

        manifest_rows.append(
            {
                "sample_id": sid,
                "source_file": row["source_file"],
                "date": row["date"],
                "picture": row["picture"],
                "side": row["side"],
                "page_type": row["page_type"],
                "classification_confidence": row.get("confidence", ""),
                "needs_review": row.get("needs_review", ""),
                "strategy_status": PAGE_TYPES.get(row["page_type"], PAGE_TYPES["unknown_review"])["status"],
                "strategy": PAGE_TYPES.get(row["page_type"], PAGE_TYPES["unknown_review"])["strategy"],
                "status": status,
                "error": error,
            }
        )

    manifest_fields = [
        "sample_id",
        "source_file",
        "date",
        "picture",
        "side",
        "page_type",
        "classification_confidence",
        "needs_review",
        "strategy_status",
        "strategy",
        "status",
        "error",
    ]
    write_csv(args.output_dir / "page_type_processing_manifest.csv", manifest_rows, manifest_fields)
    write_csv(args.output_dir / "combined_ocr_words.csv", all_words, WORD_FIELDS)
    write_csv(args.output_dir / "combined_ocr_lines.csv", all_lines, LINE_FIELDS)
    write_csv(args.output_dir / "combined_detected_headings.csv", all_headings, ["sample_id", "page_type", "heading_pattern", "matched"])
    if diagnostics:
        write_csv(args.output_dir / "combined_page_diagnostics.csv", diagnostics, list(diagnostics[0].keys()))
    write_readme(args.output_dir, selected, diagnostics)

    print(f"Wrote: {args.output_dir}")
    print(f"Processed pages: {sum(row['status'] == 'processed' for row in manifest_rows)}")
    print(f"Failed pages: {sum(row['status'] == 'failed' for row in manifest_rows)}")


if __name__ == "__main__":
    main()
