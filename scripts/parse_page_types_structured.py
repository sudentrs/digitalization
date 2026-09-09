"""Prototype structured parsers for all classified page types.

This consumes the shared multi-type OCR output from batch_process_page_types.py
and writes page-type-specific structured CSVs. These parsers are intentionally
conservative: they keep raw row text and validation flags instead of hiding
uncertainty.
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


DEFAULT_INPUT = Path("output/batch_all_page_types_smoke")
DEFAULT_OUTPUT = Path("output/parsed_all_page_types_smoke")

RULELIKE_RE = re.compile(r"^[\s_\-|{}\[\]().,;:!]+$")
NUMBER_RE = re.compile(r"\d[\d.,]*")


TABLE_SCHEMAS = {
    "regular_azioni": [
        ("capital_subscribed_raw", 0.000, 0.070),
        ("shares_outstanding_raw", 0.070, 0.145),
        ("nominal_value_raw", 0.145, 0.190),
        ("godimento_raw", 0.190, 0.245),
        ("cedola_date_raw", 0.245, 0.290),
        ("importo_lordo_raw", 0.290, 0.327),
        ("importo_acconto_raw", 0.327, 0.372),
        ("importo_saldo_raw", 0.372, 0.422),
        ("numero_raw", 0.422, 0.455),
        ("compensation_price_raw", 0.455, 0.520),
        ("security_name_raw", 0.520, 0.760),
        ("price_min_raw", 0.760, 0.822),
        ("price_max_raw", 0.822, 0.880),
        ("price_close_raw", 0.880, 0.945),
        ("quantity_traded_raw", 0.945, 1.000),
    ],
    "regular_obbligazioni": [
        ("capital_in_circulation_raw", 0.000, 0.056),
        ("nominal_value_raw", 0.056, 0.097),
        ("maturity_year_raw", 0.097, 0.139),
        ("extraction_period_raw", 0.139, 0.205),
        ("godimento_raw", 0.205, 0.288),
        ("coupon_date_raw", 0.288, 0.347),
        ("coupon_net_amount_raw", 0.347, 0.393),
        ("coupon_number_raw", 0.393, 0.426),
        ("compensation_price_raw", 0.426, 0.486),
        ("security_name_raw", 0.486, 0.762),
        ("price_min_raw", 0.762, 0.854),
        ("price_max_raw", 0.854, 0.895),
        ("price_close_raw", 0.895, 0.965),
        ("quantity_traded_raw", 0.965, 1.000),
    ],
    "cover_valori_stato": [
        ("label_raw", 0.000, 0.430),
        ("current_year_1_raw", 0.430, 0.555),
        ("current_year_2_raw", 0.555, 0.675),
        ("current_year_total_raw", 0.675, 0.790),
        ("previous_year_1_raw", 0.790, 0.885),
        ("previous_year_2_raw", 0.885, 1.000),
    ],
    "dividendi_in_pagamento": [
        ("left_date_raw", 0.000, 0.095),
        ("left_title_raw", 0.095, 0.285),
        ("left_coupon_raw", 0.285, 0.405),
        ("left_amount_raw", 0.405, 0.500),
        ("right_date_raw", 0.500, 0.595),
        ("right_title_raw", 0.595, 0.785),
        ("right_coupon_raw", 0.785, 0.905),
        ("right_amount_raw", 0.905, 1.000),
    ],
    "special_market_lists": [
        ("list_or_market_raw", 0.000, 0.190),
        ("security_name_raw", 0.190, 0.640),
        ("price_1_raw", 0.640, 0.765),
        ("price_2_raw", 0.765, 0.890),
        ("price_3_or_note_raw", 0.890, 1.000),
    ],
}

TEXT_PAGE_TYPES = {"comunicati_notizie", "advertisement_notice"}
SKIP_PAGE_TYPES = {"blank_or_divider"}


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


def useful_word(row: dict[str, str]) -> bool:
    text = row.get("text", "").strip()
    if not text or RULELIKE_RE.match(text):
        return False
    if parse_float(row.get("conf", "-1")) < 8:
        return False
    return int(row["right"]) > int(row["left"]) and int(row["bottom"]) > int(row["top"])


def mid_x(row: dict[str, str]) -> int:
    return (int(row["left"]) + int(row["right"])) // 2


def mid_y(row: dict[str, str]) -> int:
    return (int(row["top"]) + int(row["bottom"])) // 2


def clean_cell(text: str) -> str:
    text = re.sub(r"[{}|\\\[\]]", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def numbers(text: str) -> list[str]:
    return [match.group(0).strip(".,") for match in NUMBER_RE.finditer(text or "")]


def page_image_width(input_dir: Path, sample_id: str, words: list[dict[str, str]]) -> int:
    if not words:
        return 1
    crop_path = input_dir / words[0]["page_type"] / "samples" / sample_id / "page_crop.jpg"
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return int(image.shape[1])
    return max(int(row["right"]) for row in words) + 1


def page_image_shape(input_dir: Path, sample_id: str, page_type: str) -> tuple[int, int]:
    crop_path = input_dir / page_type / "samples" / sample_id / "page_crop.jpg"
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        h, w = image.shape
        return int(w), int(h)
    return 1, 1


def page_crop_path(input_dir: Path, sample_id: str, page_type: str) -> Path:
    return input_dir / page_type / "samples" / sample_id / "page_crop.jpg"


def primary_words(words: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in words
        if row.get("ocr_variant") == "normalized_full"
        and row.get("psm") in {"6", "11", "12"}
        and useful_word(row)
    ]
    if not candidates:
        return [row for row in words if useful_word(row)]
    first_psm = sorted({row["psm"] for row in candidates})[0]
    return [row for row in candidates if row["psm"] == first_psm]


def detect_vertical_rules(
    input_dir: Path,
    sample_id: str,
    page_type: str,
    y0: int,
    y1: int,
) -> list[int]:
    image = cv2.imread(str(page_crop_path(input_dir, sample_id, page_type)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    h, _ = image.shape
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    roi = image[y0:y1, :]
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    kernel_height = max(45, int((y1 - y0) * 0.08))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    projection = vertical.sum(axis=0) / 255
    threshold = max(30, int((y1 - y0) * 0.08))
    xs = np.where(projection > threshold)[0]
    if len(xs) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for x_val in xs[1:]:
        x = int(x_val)
        if x - prev <= 4:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    centers = [int(round((start + end) / 2)) for start, end in runs]
    merged: list[int] = []
    for center in centers:
        if not merged or center - merged[-1] > 12:
            merged.append(center)
        else:
            merged[-1] = int(round((merged[-1] + center) / 2))
    return merged


def table_body_words(
    input_dir: Path,
    sample_id: str,
    page_type: str,
    words: list[dict[str, str]],
) -> list[dict[str, str]]:
    page_words = primary_words(words)
    _, image_height = page_image_shape(input_dir, sample_id, page_type)
    if image_height <= 1:
        return page_words
    if page_type == "regular_obbligazioni":
        top = int(image_height * 0.135)
        bottom = int(image_height * 0.895)
    elif page_type == "regular_azioni":
        top = int(image_height * 0.070)
        bottom = int(image_height * 0.900)
    elif page_type == "dividendi_in_pagamento":
        top = int(image_height * 0.080)
        bottom = int(image_height * 0.930)
    else:
        top = int(image_height * 0.060)
        bottom = int(image_height * 0.930)
    filtered = [row for row in page_words if top <= mid_y(row) <= bottom]
    return filtered or page_words


def cluster_rows(words: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    if not words:
        return []
    heights = [int(row["bottom"]) - int(row["top"]) for row in words]
    median_height = float(np.median(heights)) if heights else 24.0
    gap = max(18, int(round(median_height * 0.85)))
    groups: list[list[dict[str, str]]] = []
    centers: list[int] = []
    for word in sorted(words, key=lambda row: (mid_y(row), int(row["left"]))):
        center = mid_y(word)
        if not groups or abs(center - centers[-1]) > gap:
            groups.append([word])
            centers.append(center)
        else:
            groups[-1].append(word)
            centers[-1] = int(round(sum(mid_y(item) for item in groups[-1]) / len(groups[-1])))
    return [sorted(group, key=lambda row: int(row["left"])) for group in groups]


def table_bounds(words: list[dict[str, str]], image_width: int) -> tuple[int, int]:
    if len(words) < 10:
        return 0, image_width
    lefts = np.array([int(row["left"]) for row in words])
    rights = np.array([int(row["right"]) for row in words])
    x0 = max(0, int(np.percentile(lefts, 1)) - 12)
    x1 = min(image_width, int(np.percentile(rights, 99)) + 12)
    if x1 - x0 < image_width * 0.45:
        return 0, image_width
    return x0, x1


def scaled_schema(schema: list[tuple[str, float, float]], x0: int, x1: int) -> list[tuple[str, int, int]]:
    width = max(1, x1 - x0)
    return [(name, int(round(x0 + start * width)), int(round(x0 + end * width))) for name, start, end in schema]


def snap_columns_to_rules(
    columns: list[tuple[str, int, int]],
    rules: list[int],
    *,
    tolerance: int = 38,
) -> list[tuple[str, int, int]]:
    if not columns or not rules:
        return columns
    boundaries = [columns[0][1]] + [col[2] for col in columns]
    snapped = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        nearest = min(rules, key=lambda rule: abs(rule - boundary))
        snapped.append(nearest if abs(nearest - boundary) <= tolerance else boundary)
    snapped.append(boundaries[-1])
    fixed: list[int] = [snapped[0]]
    for value in snapped[1:]:
        fixed.append(max(value, fixed[-1] + 8))
    return [(columns[index][0], fixed[index], fixed[index + 1]) for index in range(len(columns))]


def assign_cells(group: list[dict[str, str]], columns: list[tuple[str, int, int]]) -> dict[str, str]:
    cells: dict[str, list[tuple[int, str]]] = {name: [] for name, _, _ in columns}
    for word in group:
        center = mid_x(word)
        selected = None
        for name, x0, x1 in columns:
            if x0 <= center < x1:
                selected = name
                break
        if selected is None:
            selected = min(columns, key=lambda col: min(abs(center - col[1]), abs(center - col[2])))[0]
        cells[selected].append((int(word["left"]), word["text"]))
    return {name: clean_cell(" ".join(text for _, text in sorted(items))) for name, items in cells.items()}


def split_overmerged_group(group: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    if len(group) < 8:
        return [group]
    heights = [int(row["bottom"]) - int(row["top"]) for row in group]
    median_height = float(np.median(heights)) if heights else 24.0
    group_top = min(int(row["top"]) for row in group)
    group_bottom = max(int(row["bottom"]) for row in group)
    if group_bottom - group_top < max(10_000, int(median_height * 400)):
        return [group]

    normal_words = [
        row
        for row in group
        if int(row["bottom"]) - int(row["top"]) <= median_height * 1.8
    ]
    if len(normal_words) < 6:
        return [group]

    threshold = max(14, int(round(median_height * 0.75)))
    clusters: list[list[dict[str, str]]] = []
    centers: list[int] = []
    for word in sorted(normal_words, key=lambda row: (int(row["top"]), int(row["left"]))):
        top = int(word["top"])
        if not clusters or abs(top - centers[-1]) > threshold:
            clusters.append([word])
            centers.append(top)
        else:
            clusters[-1].append(word)
            centers[-1] = int(round(sum(int(item["top"]) for item in clusters[-1]) / len(clusters[-1])))

    kept = [(cluster, center) for cluster, center in zip(clusters, centers) if len(cluster) >= 3]
    clusters = [cluster for cluster, _ in kept]
    centers = [center for _, center in kept]
    if len(clusters) <= 1:
        return [group]

    split_rows: list[list[dict[str, str]]] = [[] for _ in clusters]
    for word in group:
        top = int(word["top"])
        chosen = min(range(len(centers)), key=lambda index: abs(top - centers[index]))
        split_rows[chosen].append(word)
    return [sorted(row, key=lambda item: int(item["left"])) for row in split_rows if row]


def split_obbligazioni_anchor_group(
    group: list[dict[str, str]],
    columns: list[tuple[str, int, int]],
) -> list[list[dict[str, str]]]:
    if len(group) < 14:
        return [group]
    by_name = {name: (x0, x1) for name, x0, x1 in columns}
    anchor_columns = [
        by_name[name]
        for name in ["capital_in_circulation_raw", "nominal_value_raw", "maturity_year_raw"]
        if name in by_name
    ]
    if not anchor_columns or "security_name_raw" not in by_name:
        return [group]
    heights = [int(row["bottom"]) - int(row["top"]) for row in group]
    median_height = float(np.median(heights)) if heights else 24.0
    anchors = []
    for word in group:
        text = word.get("text", "")
        if not NUMBER_RE.search(text):
            continue
        if int(word["bottom"]) - int(word["top"]) > median_height * 2.5:
            continue
        center = mid_x(word)
        if any(x0 <= center < x1 for x0, x1 in anchor_columns):
            anchors.append(word)
    if len(anchors) < 4:
        return [group]

    threshold = max(18, int(round(median_height * 0.85)))
    anchor_tops: list[int] = []
    for word in sorted(anchors, key=lambda row: (int(row["top"]), int(row["left"]))):
        top = int(word["top"])
        if not anchor_tops or top - anchor_tops[-1] > threshold:
            anchor_tops.append(top)
    if len(anchor_tops) < 2:
        return [group]
    if len(anchor_tops) > 4:
        return [group]

    sec_x0, sec_x1 = by_name["security_name_raw"]
    rows: list[list[dict[str, str]]] = [[] for _ in anchor_tops]
    security_margin = max(5, int(median_height * 0.25))
    for word in group:
        top = int(word["top"])
        height = int(word["bottom"]) - int(word["top"])
        if height > median_height * 5:
            continue
        center = mid_x(word)
        if sec_x0 <= center < sec_x1:
            chosen = 0
            for index, anchor_top in enumerate(anchor_tops):
                if anchor_top <= top - security_margin:
                    chosen = index
        else:
            chosen = min(range(len(anchor_tops)), key=lambda index: abs(top - anchor_tops[index]))
        rows[chosen].append(word)

    split_rows = []
    for row in rows:
        if len(row) < 5:
            return [group]
        row_text = clean_cell(" ".join(word["text"] for word in sorted(row, key=lambda item: int(item["left"]))))
        if len(numbers(row_text)) < 2:
            return [group]
        split_rows.append(sorted(row, key=lambda item: int(item["left"])))
    return split_rows


def classify_table_row(raw_text: str, cells: dict[str, str]) -> tuple[str, str]:
    lowered = raw_text.lower()
    section_terms = [
        "finanzi",
        "assic",
        "trasport",
        "tessil",
        "miner",
        "chimic",
        "diversi",
        "garanzie statali",
        "reddito fisso",
        "partecipazione",
        "titoli di stati esteri",
        "obbligazioni",
        "fondiari",
        "equiparati",
        "comunali",
        "provinciali",
        "elettrici",
    ]
    if not numbers(raw_text):
        if any(token in lowered for token in section_terms):
            return "section_heading", "section_heading"
        if len(raw_text.split()) <= 8:
            return "heading_or_label", "header"
    if any(token in lowered for token in section_terms) and len(numbers(raw_text)) <= 2:
        return "section_heading", "section_heading"
    name = cells.get("security_name_raw", "") or cells.get("left_title_raw", "") or cells.get("label_raw", "")
    if numbers(raw_text) and (name or re.search(r"[^\W\d_]", raw_text, flags=re.UNICODE)):
        return "candidate_record", ""
    return "other_text", ""


def parse_table_page(
    input_dir: Path,
    sample_id: str,
    page_type: str,
    words: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    schema = TABLE_SCHEMAS[page_type]
    page_words = table_body_words(input_dir, sample_id, page_type, words)
    image_width = page_image_width(input_dir, sample_id, page_words)
    x0, x1 = table_bounds(page_words, image_width)
    columns = scaled_schema(schema, x0, x1)
    if page_type == "regular_obbligazioni" and page_words:
        y0 = min(int(row["top"]) for row in page_words)
        y1 = max(int(row["bottom"]) for row in page_words)
        columns = snap_columns_to_rules(columns, detect_vertical_rules(input_dir, sample_id, page_type, y0, y1))
    groups = []
    for group in cluster_rows(page_words):
        if page_type == "regular_obbligazioni":
            groups.extend(split_obbligazioni_anchor_group(group, columns))
        else:
            groups.append(group)
    rows: list[dict[str, str]] = []
    active_section = ""
    for index, group in enumerate(groups, start=1):
        raw_text = clean_cell(" ".join(word["text"] for word in group))
        cells = assign_cells(group, columns)
        row_type, role = classify_table_row(raw_text, cells)
        if row_type == "section_heading":
            active_section = raw_text
        flags = []
        if row_type == "candidate_record":
            if page_type in {"regular_azioni", "regular_obbligazioni"} and not cells.get("security_name_raw", ""):
                flags.append("missing_security_name")
            if len(numbers(raw_text)) < 2:
                flags.append("few_numeric_tokens")
        if parse_float(str(sum(parse_float(word["conf"]) for word in group) / max(1, len(group)))) < 45:
            flags.append("low_mean_ocr_confidence")
        out = {
            "sample_id": sample_id,
            "page_type": page_type,
            "record_index": str(index),
            "record_type": row_type,
            "record_role": role,
            "section": active_section,
            "top": str(min(int(word["top"]) for word in group)),
            "bottom": str(max(int(word["bottom"]) for word in group)),
            "left": str(min(int(word["left"]) for word in group)),
            "right": str(max(int(word["right"]) for word in group)),
            "word_count": str(len(group)),
            "mean_conf": f"{sum(parse_float(word['conf']) for word in group) / max(1, len(group)):.2f}",
            "raw_text": raw_text,
            "numeric_token_count": str(len(numbers(raw_text))),
            "validation_flags": ";".join(flags),
        }
        out.update(cells)
        rows.append(out)
    column_rows = [
        {"sample_id": sample_id, "page_type": page_type, "column": name, "x0": str(start), "x1": str(end)}
        for name, start, end in columns
    ]
    return rows, column_rows


def merge_lines_to_blocks(lines: list[dict[str, str]], vertical_gap: int = 45) -> list[list[dict[str, str]]]:
    primary = [line for line in lines if line.get("ocr_variant") == "normalized_full"]
    if not primary:
        primary = lines
    primary = sorted(primary, key=lambda row: (int(row["top"]), int(row["left"])))
    blocks: list[list[dict[str, str]]] = []
    last_bottom = -1
    for line in primary:
        top = int(line["top"])
        if not blocks or top - last_bottom > vertical_gap:
            blocks.append([line])
        else:
            blocks[-1].append(line)
        last_bottom = max(last_bottom, int(line["bottom"]))
    return blocks


def parse_text_page(sample_id: str, page_type: str, lines: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    primary_lines = [line for line in lines if line.get("ocr_variant") == "normalized_full"]
    if not primary_lines:
        primary_lines = lines
    primary_lines = sorted(primary_lines, key=lambda row: (int(row["top"]), int(row["left"])))
    for index, line in enumerate(primary_lines, start=1):
        text = clean_cell(line["text"])
        if not text:
            continue
        upperish = sum(1 for ch in text if ch.isupper())
        alpha = sum(1 for ch in text if ch.isalpha())
        role = "heading" if alpha and upperish / max(1, alpha) > 0.55 and len(text) < 120 else "paragraph"
        rows.append(
            {
                "sample_id": sample_id,
                "page_type": page_type,
                "record_index": str(index),
                "record_type": "text_line",
                "record_role": role,
                "section": "",
                "top": line["top"],
                "bottom": line["bottom"],
                "left": line["left"],
                "right": line["right"],
                "line_count": "1",
                "mean_conf": line.get("mean_conf", ""),
                "raw_text": text,
                "numeric_token_count": str(len(numbers(text))),
                "validation_flags": "",
            }
        )
    return rows


def parse_skip_page(sample_id: str, page_type: str, words: list[dict[str, str]]) -> list[dict[str, str]]:
    text = clean_cell(" ".join(word["text"] for word in primary_words(words)))
    return [
        {
            "sample_id": sample_id,
            "page_type": page_type,
            "record_index": "1",
            "record_type": "skip_page",
            "record_role": "blank_or_divider",
            "section": "",
            "raw_text": text,
            "numeric_token_count": str(len(numbers(text))),
            "validation_flags": "skip_structured_extraction",
        }
    ]


def fieldnames_for(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "sample_id",
        "page_type",
        "record_index",
        "record_type",
        "record_role",
        "section",
        "top",
        "bottom",
        "left",
        "right",
        "word_count",
        "line_count",
        "mean_conf",
        "raw_text",
        "numeric_token_count",
        "validation_flags",
    ]
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prototype structured outputs for all page types.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    words = read_csv(args.input_dir / "combined_ocr_words.csv")
    lines = read_csv(args.input_dir / "combined_ocr_lines.csv")
    words_by_sample: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    lines_by_sample: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    page_type_by_sample: dict[str, str] = {}
    for row in words:
        words_by_sample[row["sample_id"]].append(row)
        page_type_by_sample[row["sample_id"]] = row["page_type"]
    for row in lines:
        lines_by_sample[row["sample_id"]].append(row)
        page_type_by_sample[row["sample_id"]] = row["page_type"]

    all_rows: list[dict[str, str]] = []
    all_columns: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    by_type: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for sample_id in sorted(page_type_by_sample):
        page_type = page_type_by_sample[sample_id]
        if page_type in TABLE_SCHEMAS:
            parsed, columns = parse_table_page(args.input_dir, sample_id, page_type, words_by_sample[sample_id])
            all_columns.extend(columns)
            parser_level = "prototype_table_parser"
        elif page_type in TEXT_PAGE_TYPES:
            parsed = parse_text_page(sample_id, page_type, lines_by_sample[sample_id])
            parser_level = "prototype_text_block_parser"
        elif page_type in SKIP_PAGE_TYPES:
            parsed = parse_skip_page(sample_id, page_type, words_by_sample[sample_id])
            parser_level = "skip_parser"
        else:
            parsed = parse_text_page(sample_id, page_type, lines_by_sample[sample_id])
            parser_level = "generic_text_block_parser"

        by_type[page_type].extend(parsed)
        all_rows.extend(parsed)
        record_counts = Counter(row["record_type"] for row in parsed)
        flagged = sum(bool(row.get("validation_flags")) for row in parsed)
        summary_rows.append(
            {
                "sample_id": sample_id,
                "page_type": page_type,
                "parser_level": parser_level,
                "records": str(len(parsed)),
                "candidate_records": str(record_counts.get("candidate_record", 0)),
                "text_records": str(record_counts.get("text_block", 0) + record_counts.get("text_line", 0)),
                "flagged_records": str(flagged),
            }
        )

    for page_type, rows_for_type in by_type.items():
        write_csv(args.output_dir / page_type / "structured_records.csv", rows_for_type, fieldnames_for(rows_for_type))
    write_csv(args.output_dir / "combined_structured_records.csv", all_rows, fieldnames_for(all_rows))
    write_csv(args.output_dir / "detected_table_columns.csv", all_columns, ["sample_id", "page_type", "column", "x0", "x1"])
    write_csv(args.output_dir / "parse_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])

    type_counts = Counter(row["page_type"] for row in summary_rows)
    readme = [
        "# Prototype Structured Parsers For All Page Types",
        "",
        "This folder converts the shared OCR outputs into page-type-specific structured records.",
        "The parsers are first-pass prototypes for non-azioni pages and should be validated on more samples before being treated as final data.",
        "",
        "## Outputs",
        "",
        "- `combined_structured_records.csv`: all prototype records across page types.",
        "- `parse_summary.csv`: record counts and parser level per page side.",
        "- `detected_table_columns.csv`: column bands used by table-like parsers.",
        "- `<page_type>/structured_records.csv`: page-type-specific output.",
        "",
        "## Parser Levels",
        "",
        "- `prototype_table_parser`: row/column parser using a page-type schema.",
        "- `prototype_text_block_parser`: paragraph/list block parser for notice pages.",
        "- `skip_parser`: blank or divider pages preserved only as diagnostics.",
        "",
        "## Processed Page Types",
        "",
    ]
    for page_type, count in sorted(type_counts.items()):
        readme.append(f"- `{page_type}`: {count} page side(s)")
    readme.extend(
        [
            "",
            "## Important Limitation",
            "",
            "The mature validated parser is still the azioni parser. These new parsers give every other page type its own structured starting point, but their schemas need iterative review and rule refinement just like the azioni parser did.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Wrote: {args.output_dir}")
    print(f"Structured records: {len(all_rows)}")
    for page_type, count in sorted(type_counts.items()):
        print(f"{page_type}: {count} page side(s)")


if __name__ == "__main__":
    main()
