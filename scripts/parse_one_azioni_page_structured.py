"""Create a structured parse for one azioni OCR page.

This is still a one-page prototype, but it is less brittle than the first
version. It detects vertical table rules from the image, snaps the expected
azioni columns to those rules, fuzzy-matches OCR-damaged section headings, and
uses existing column-specific OCR as a fallback for weak numeric cells.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np


DEFAULT_WORDS = Path("output/ocr_one_page_experiment/words/original_full_psm12_words.csv")
DEFAULT_IMAGE = Path("output/ocr_one_page_experiment/images/original_full.png")
DEFAULT_FULL_BACKUP_WORDS = Path("output/ocr_one_page_experiment/words/original_full_psm11_words.csv")
DEFAULT_NUMERIC_BACKUP_WORDS = Path("output/ocr_one_page_experiment/words/numeric_columns_body_psm12_words.csv")
DEFAULT_PRICE_BACKUP_WORDS = Path("output/ocr_one_page_experiment/words/price_quantity_columns_body_psm12_words.csv")
DEFAULT_OUTPUT = Path("output/structured_one_page")


BASE_WIDTH = 2600
BASE_COLS = [
    ("capital_subscribed_raw", 0, 180),
    ("shares_outstanding_raw", 180, 370),
    ("nominal_value_raw", 370, 485),
    ("godimento_raw", 485, 630),
    ("cedola_date_raw", 630, 740),
    ("importo_lordo_raw", 740, 835),
    ("importo_acconto_raw", 835, 950),
    ("importo_saldo_raw", 950, 1080),
    ("numero_raw", 1080, 1165),
    ("compensation_price_raw", 1165, 1325),
    ("issuer_name_raw", 1325, 1945),
    ("price_min_raw", 1945, 2105),
    ("price_max_raw", 2105, 2248),
    ("price_close_raw", 2248, 2418),
    ("quantity_traded_raw", 2418, 2600),
]

NUMERIC_BACKUP_FIELDS = {
    "capital_subscribed_raw",
    "shares_outstanding_raw",
    "nominal_value_raw",
    "cedola_date_raw",
    "importo_lordo_raw",
    "importo_acconto_raw",
    "importo_saldo_raw",
    "numero_raw",
    "compensation_price_raw",
}
PRICE_BACKUP_FIELDS = {
    "price_min_raw",
    "price_max_raw",
    "price_close_raw",
    "quantity_traded_raw",
}

SECTION_LABELS = [
    "Finanziari",
    "Assicurativi",
    "Trasporti",
    "Tessili e Manifatturieri",
    "Minerari e Metallurgici",
    "Meccanici ed Automobilistici",
    "Elettrici ed Elettrotecnici",
    "Alimentari",
    "Chimici",
    "Immobiliari ed Agricoli",
    "Diversi",
]
SECTION_ALIASES = {
    "finanziari": "Finanziari",
    "assicurativi": "Assicurativi",
    "assicurazioni": "Assicurativi",
    "asslourativi": "Assicurativi",
    "trasporti": "Trasporti",
    "tessili": "Tessili e Manifatturieri",
    "manifatturieri": "Tessili e Manifatturieri",
    "manlfatturieri": "Tessili e Manifatturieri",
    "minerari": "Minerari e Metallurgici",
    "metallurgici": "Minerari e Metallurgici",
    "metallurgiol": "Minerari e Metallurgici",
    "meccanici": "Meccanici ed Automobilistici",
    "automobilistici": "Meccanici ed Automobilistici",
    "automobillstici": "Meccanici ed Automobilistici",
    "elettrici": "Elettrici ed Elettrotecnici",
    "elettriol": "Elettrici ed Elettrotecnici",
    "elettrotecnici": "Elettrici ed Elettrotecnici",
    "elettroteonol": "Elettrici ed Elettrotecnici",
    "alimentari": "Alimentari",
    "alimentar": "Alimentari",
    "chimici": "Chimici",
    "ohimlol": "Chimici",
    "immobiliari": "Immobiliari ed Agricoli",
    "agricoli": "Immobiliari ed Agricoli",
    "agrlcol": "Immobiliari ed Agricoli",
    "diversi": "Diversi",
    "dlversi": "Diversi",
}

RULELIKE_RE = re.compile(r"^[\s_\-|{}\[\]().,;:!]+$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def conf(row: dict[str, str]) -> float:
    try:
        return float(row["conf"])
    except ValueError:
        return -1.0


def mid_x(row: dict[str, str]) -> int:
    return (int(row["left"]) + int(row["right"])) // 2


def mid_y(row: dict[str, str]) -> int:
    return (int(row["top"]) + int(row["bottom"])) // 2


def is_noise(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    return bool(RULELIKE_RE.match(text))


def useful_word(row: dict[str, str]) -> bool:
    text = row["text"].strip()
    if is_noise(text):
        return False
    if conf(row) < 12:
        return False
    width = int(row["right"]) - int(row["left"])
    height = int(row["bottom"]) - int(row["top"])
    return width > 1 and height > 1


def clean_cell(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"[{}\[\]|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def digit_count(text: str) -> int:
    return len(re.findall(r"\d", text))


def looks_like_large_amount(text: str) -> bool:
    return bool(re.search(r"\d[\d. ]{3,}\d", text))


def normalize_letters(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]+", " ", ascii_text.lower()).strip()


def fuzzy_section(raw_text: str) -> str:
    words = normalize_letters(raw_text).split()
    candidates = set(words)
    candidates.update(" ".join(words[i : i + 2]) for i in range(max(0, len(words) - 1)))

    for candidate in candidates:
        if candidate in SECTION_ALIASES:
            return SECTION_ALIASES[candidate]

    best_label = ""
    best_score = 0.0
    alias_items = list(SECTION_ALIASES.items()) + [(normalize_letters(label), label) for label in SECTION_LABELS]
    for candidate in candidates:
        for alias, label in alias_items:
            score = SequenceMatcher(None, candidate, alias).ratio()
            if score > best_score:
                best_score = score
                best_label = label
    return best_label if best_score >= 0.76 else ""


def detect_vertical_rules(image_path: Path, body_top: int, body_bottom: int) -> list[int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read image: {image_path}")

    h, _ = image.shape
    y0 = max(0, min(body_top, h - 1))
    y1 = max(y0 + 1, min(body_bottom, h))
    body = image[y0:y1, :]
    binary = cv2.adaptiveThreshold(
        body,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    # Shorter kernels catch broken table rules in these scans. False positives
    # are filtered later by snapping only near expected azioni boundaries.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    scores = np.sum(vertical > 0, axis=0)
    threshold = max(10, int((y1 - y0) * 0.005))
    xs = np.where(scores >= threshold)[0]
    if len(xs) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = int(xs[0])
    previous = int(xs[0])
    for raw_x in xs[1:]:
        x = int(raw_x)
        if x - previous > 8:
            segments.append((start, previous))
            start = x
        previous = x
    segments.append((start, previous))

    centers = [int(round((start + end) / 2)) for start, end in segments if end - start <= 24]
    deduped: list[int] = []
    for x in sorted(centers):
        if not deduped or x - deduped[-1] > 20:
            deduped.append(x)
        else:
            deduped[-1] = int(round((deduped[-1] + x) / 2))
    return deduped


def estimate_table_bounds(
    image_width: int,
    body_words: list[dict[str, str]],
    rules: list[int],
) -> tuple[int, int, str]:
    """Estimate the printed table span, excluding page margin/background.

    The page crops often include torn edge/background outside the table. Scaling
    column coordinates to the full crop width shifts right-page columns, so use
    OCR word extents plus nearby detected rules as a more local table frame.
    """
    useful = [
        row
        for row in body_words
        if useful_word(row)
        and (re.search(r"\d", row["text"]) or len(row["text"].strip()) >= 3)
    ]
    if len(useful) < 20:
        return 0, image_width, "full_image_fallback"

    lefts = np.array([int(row["left"]) for row in useful])
    rights = np.array([int(row["right"]) for row in useful])
    word_left = int(np.percentile(lefts, 1))
    word_right = int(np.percentile(rights, 99))
    padding = max(18, int(image_width * 0.012))
    x0 = max(0, word_left - padding)
    x1 = min(image_width, word_right + padding)

    nearby_left_rules = [x for x in rules if x <= word_left + 45]
    nearby_right_rules = [x for x in rules if x >= word_right - 45]
    if nearby_left_rules:
        x0 = max(0, min(nearby_left_rules))
    if nearby_right_rules:
        x1 = min(image_width, max(nearby_right_rules))

    min_width = int(image_width * 0.72)
    max_width = int(image_width * 0.98)
    if x1 - x0 < min_width or x1 - x0 > max_width:
        return 0, image_width, "full_image_width_guard"
    return x0, x1, "word_rule_bounds"


def build_columns(
    image_path: Path,
    body_top: int,
    body_bottom: int,
    words: list[dict[str, str]] | None = None,
) -> tuple[list[tuple[str, int, int]], list[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read image: {image_path}")

    _, w = image.shape
    rules = detect_vertical_rules(image_path, body_top, body_bottom)
    body_words = [
        row
        for row in (words or [])
        if int(row.get("bottom", "0")) >= body_top and int(row.get("top", "0")) <= body_bottom
    ]
    table_x0, table_x1, _ = estimate_table_bounds(w, body_words, rules) if body_words else (0, w, "no_words")
    table_width = table_x1 - table_x0
    scale = table_width / BASE_WIDTH
    snapped: dict[int, int] = {}

    for _, start, end in BASE_COLS:
        for boundary in (start, end):
            expected = int(round(table_x0 + boundary * scale))
            nearest = min(rules, key=lambda x: abs(x - expected), default=expected)
            snapped[boundary] = nearest if table_x0 - 15 <= nearest <= table_x1 + 15 and abs(nearest - expected) <= 35 else expected

    columns: list[tuple[str, int, int]] = []
    for name, start, end in BASE_COLS:
        x0 = snapped[start]
        x1 = snapped[end]
        if x1 <= x0 + 10:
            x0 = int(round(table_x0 + start * scale))
            x1 = int(round(table_x0 + end * scale))
        columns.append((name, max(0, x0), min(w, x1)))
    return columns, rules


def filtered_row_words(words: list[dict[str, str]], body_top: int, body_bottom: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in words
        if useful_word(row) and int(row["bottom"]) >= body_top and int(row["top"]) <= body_bottom
    ]
    heights = [int(row["bottom"]) - int(row["top"]) for row in candidates]
    median_height = float(np.median(heights)) if heights else 24.0
    max_word_height = 66
    kept = [
        row
        for row in candidates
        if (int(row["bottom"]) - int(row["top"])) <= max_word_height
        or (int(row["right"]) - int(row["left"])) >= 260
    ]
    return kept


def cluster_by_y(words: list[dict[str, str]], gap: int) -> list[list[dict[str, str]]]:
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
    return groups


def group_rows(
    words: list[dict[str, str]],
    body_top: int,
    body_bottom: int,
    columns: list[tuple[str, int, int]],
    method: str = "center",
) -> list[list[dict[str, str]]]:
    kept = filtered_row_words(words, body_top, body_bottom)
    heights = [int(row["bottom"]) - int(row["top"]) for row in kept]
    median_height = float(np.median(heights)) if heights else 24.0
    gap = max(17, int(round(median_height * 0.75)))
    if method == "center":
        return cluster_by_y(kept, gap=gap)

    issuer_start, issuer_end = next(
        (start, end) for name, start, end in columns if name == "issuer_name_raw"
    )
    issuer_words = [
        word
        for word in kept
        if issuer_start <= mid_x(word) < issuer_end
        and re.search(r"[A-Za-z]", word["text"])
        and len(word["text"].strip()) > 1
    ]
    issuer_groups = [
        group
        for group in cluster_by_y(issuer_words, gap=max(18, int(round(median_height * 0.95))))
        if len(" ".join(word["text"] for word in group).strip()) >= 3
    ]
    if len(issuer_groups) < 10:
        return cluster_by_y(kept, gap=gap)

    anchors = [
        {
            "y": int(round(np.median([mid_y(word) for word in group]))),
            "words": list(group),
        }
        for group in issuer_groups
    ]

    unassigned: list[dict[str, str]] = []
    anchor_ys = [anchor["y"] for anchor in anchors]
    for word in kept:
        if any(word in anchor["words"] for anchor in anchors):
            continue
        center = mid_y(word)
        nearest_index = min(range(len(anchor_ys)), key=lambda idx: abs(anchor_ys[idx] - center))
        nearest_distance = abs(anchor_ys[nearest_index] - center)
        tolerance = max(26, int(round(median_height * 1.15)))
        if nearest_distance <= tolerance:
            anchors[nearest_index]["words"].append(word)
        else:
            unassigned.append(word)

    groups = [sorted(anchor["words"], key=lambda row: int(row["left"])) for anchor in anchors]
    groups.extend(cluster_by_y(unassigned, gap=gap))
    return sorted(groups, key=lambda group: min(mid_y(word) for word in group))


def row_grouping_diagnostics(
    words: list[dict[str, str]],
    body_top: int,
    body_bottom: int,
) -> tuple[list[dict[str, str]], float, int]:
    candidates = [
        row
        for row in words
        if useful_word(row) and int(row["bottom"]) >= body_top and int(row["top"]) <= body_bottom
    ]
    heights = [int(row["bottom"]) - int(row["top"]) for row in candidates]
    median_height = float(np.median(heights)) if heights else 24.0
    max_word_height = 66
    removed = [
        {
            "text": row["text"],
            "conf": row["conf"],
            "left": row["left"],
            "top": row["top"],
            "right": row["right"],
            "bottom": row["bottom"],
            "width": str(int(row["right"]) - int(row["left"])),
            "height": str(int(row["bottom"]) - int(row["top"])),
            "reason": "tall_row_spanning_artifact",
        }
        for row in candidates
        if (int(row["bottom"]) - int(row["top"])) > max_word_height
        and (int(row["right"]) - int(row["left"])) < 260
    ]
    return removed, median_height, max_word_height


def numeric_token_parts(text: str) -> list[str]:
    parts = re.findall(r"\d[\d.,]*\s*(?:[—,-]+)?|[A-Za-z]+|[^\s]+", text)
    cleaned = []
    for part in parts:
        part = part.strip().lstrip("/").strip()
        if part:
            cleaned.append(part)
    return cleaned


def word_column_overlaps(word: dict[str, str], columns: list[tuple[str, int, int]]) -> list[tuple[str, int, int, int]]:
    left = int(word["left"])
    right = int(word["right"])
    overlaps = []
    for name, start, end in columns:
        overlap = max(0, min(right, end) - max(left, start))
        if overlap > 0:
            overlaps.append((name, start, end, overlap))
    return overlaps


def assign_word_to_cells(
    word: dict[str, str],
    columns: list[tuple[str, int, int]],
) -> list[tuple[str, int, str]]:
    text = word["text"].strip()
    overlaps = word_column_overlaps(word, columns)
    if not overlaps:
        return []

    meaningful = [item for item in overlaps if item[3] >= 8]
    if not meaningful:
        meaningful = overlaps

    center = mid_x(word)
    center_col = next((item for item in overlaps if item[1] <= center < item[2]), None)
    # Tesseract often glues adjacent numeric cells into one wide token, e.g.
    # "31.250.000/3000.—". Split only number-like tokens; keep names intact.
    mostly_numeric = bool(re.search(r"\d", text)) and not re.search(r"[A-Za-z]{3,}", text)
    parts = numeric_token_parts(text)
    has_glue_marker = "/" in text or bool(re.search(r"\d\s+\d", text))
    if mostly_numeric and has_glue_marker and len(meaningful) > 1 and len(parts) >= 2:
        ordered_cols = sorted(meaningful, key=lambda item: item[1])
        assignments: list[tuple[str, int, str]] = []
        for idx, part in enumerate(parts):
            col_idx = min(idx, len(ordered_cols) - 1)
            assignments.append((ordered_cols[col_idx][0], int(word["left"]) + idx, part))
        return assignments

    best = center_col if center_col else max(meaningful, key=lambda item: item[3])
    return [(best[0], int(word["left"]), text)]


def classify_row(raw_text: str, issuer_text: str) -> tuple[str, str]:
    section = fuzzy_section(raw_text)
    if section and not re.search(r"\d", raw_text):
        return "section_heading", section
    if section and len(re.findall(r"\d", raw_text)) <= 2:
        return "section_heading", section
    if re.search(r"\d", raw_text) and (issuer_text.strip() or re.search(r"[A-Za-z]", raw_text)):
        return "candidate_security_row", ""
    return "other_text", ""


def infer_row_role(row_type: str, raw_text: str, cell_text: dict[str, str]) -> str:
    if row_type == "section_heading":
        return "section_heading"
    if row_type != "candidate_security_row":
        return "other_text"

    issuer = cell_text.get("issuer_name_raw", "").strip()
    has_left_identity = any(
        digit_count(cell_text.get(field, "")) > 0
        for field in ("capital_subscribed_raw", "shares_outstanding_raw", "nominal_value_raw")
    )
    starts_like_continuation = bool(re.match(r"^[»\"'().,\-\s]+", issuer))
    raw_has_ditto = bool(re.search(r"[»\"]|^\s*[.,;:-]", raw_text))
    if not issuer or starts_like_continuation or (raw_has_ditto and not has_left_identity):
        return "security_continuation_row"
    return "security_main_row"


def shift_backup_words(words: list[dict[str, str]], x_offset: int, y_offset: int) -> list[dict[str, str]]:
    shifted: list[dict[str, str]] = []
    for row in words:
        copy = dict(row)
        for key in ("left", "right"):
            copy[key] = str(int(copy[key]) + x_offset)
        for key in ("top", "bottom"):
            copy[key] = str(int(copy[key]) + y_offset)
        shifted.append(copy)
    return shifted


def backup_text_for_row(
    backup_words: list[dict[str, str]],
    top: int,
    bottom: int,
    col_start: int,
    col_end: int,
) -> str:
    candidates = [
        word
        for word in backup_words
        if useful_word(word)
        and top <= mid_y(word) < bottom
        and col_start <= mid_x(word) < col_end
    ]
    return clean_cell(" ".join(word["text"] for word in sorted(candidates, key=lambda row: int(row["left"]))))


def merge_backup_cells(
    row: dict[str, str],
    columns: list[tuple[str, int, int]],
    full_backup_words: list[dict[str, str]],
    numeric_backup_words: list[dict[str, str]],
    price_backup_words: list[dict[str, str]],
) -> None:
    top = int(row["top"])
    bottom = int(row["bottom"])
    for name, start, end in columns:
        backup_sources: list[tuple[str, list[dict[str, str]]]] = []
        if name in NUMERIC_BACKUP_FIELDS or name in PRICE_BACKUP_FIELDS:
            backup_sources.append(("full_psm11", full_backup_words))
        if name in NUMERIC_BACKUP_FIELDS:
            backup_sources.append(("numeric_psm12", numeric_backup_words))
        elif name in PRICE_BACKUP_FIELDS:
            backup_sources.append(("price_psm12", price_backup_words))
        if not backup_sources:
            continue

        current = row.get(name, "")
        current_digit_count = digit_count(current)
        backup_options: list[tuple[str, str, int]] = []
        for source, backup_words in backup_sources:
            if not backup_words:
                continue
            backup_text = backup_text_for_row(backup_words, top, bottom, start, end)
            backup_digit_count = digit_count(backup_text)
            if backup_text and backup_digit_count > 0:
                backup_options.append((source, backup_text, backup_digit_count))
        if not backup_options:
            continue

        source, backup_text, backup_digit_count = max(backup_options, key=lambda item: item[2])
        row[f"{name}_backup_hint"] = backup_text
        row[f"{name}_backup_source"] = source
        if not current or (current_digit_count == 0 and backup_digit_count > 0):
            row[name] = backup_text
            row[f"{name}_source"] = source
        elif (
            name in {"capital_subscribed_raw", "shares_outstanding_raw"}
            and looks_like_large_amount(backup_text)
            and backup_digit_count >= current_digit_count + 3
        ):
            row[name] = backup_text
            row[f"{name}_source"] = source
        else:
            row[f"{name}_source"] = "main"


def row_bands(groups: list[list[dict[str, str]]], body_top: int, body_bottom: int) -> list[tuple[int, int]]:
    centers = [int(round(np.median([mid_y(word) for word in group]))) for group in groups]
    bands: list[tuple[int, int]] = []
    for index, center in enumerate(centers):
        if index == 0:
            top = max(body_top, center - 18)
        else:
            top = int(round((centers[index - 1] + center) / 2))
        if index == len(centers) - 1:
            bottom = min(body_bottom, center + 18)
        else:
            bottom = int(round((center + centers[index + 1]) / 2))
        if bottom <= top:
            bottom = top + 1
        bands.append((top, bottom))
    return bands


def parse_groups(
    groups: list[list[dict[str, str]]],
    columns: list[tuple[str, int, int]],
    full_backup_words: list[dict[str, str]],
    numeric_backup_words: list[dict[str, str]],
    price_backup_words: list[dict[str, str]],
    body_top: int,
    body_bottom: int,
) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    active_section = ""
    bands = row_bands(groups, body_top, body_bottom)
    for index, (group, band) in enumerate(zip(groups, bands, strict=True), start=1):
        band_top, band_bottom = band
        group = sorted(group, key=lambda row: int(row["left"]))
        raw_text = clean_cell(" ".join(row["text"] for row in group))
        cells: dict[str, list[tuple[int, str]]] = {name: [] for name, _, _ in columns}
        for word in group:
            for col, left, text in assign_word_to_cells(word, columns):
                cells[col].append((left, text))

        cell_text = {
            name: clean_cell(" ".join(text for _, text in sorted(items, key=lambda item: item[0])))
            for name, items in cells.items()
        }
        row_type, section_label = classify_row(raw_text, cell_text["issuer_name_raw"])
        row_role = infer_row_role(row_type, raw_text, cell_text)
        if row_type == "candidate_security_row":
            evidence_fields = [
                "capital_subscribed_raw",
                "shares_outstanding_raw",
                "nominal_value_raw",
                "price_min_raw",
                "price_max_raw",
                "price_close_raw",
                "quantity_traded_raw",
            ]
            evidence_count = sum(1 for field in evidence_fields if digit_count(cell_text.get(field, "")) > 0)
            if evidence_count < 2:
                row_type = "other_text"
                row_role = "other_text"
        if row_type == "section_heading":
            active_section = section_label or raw_text
            row_role = "section_heading"

        confs = [conf(word) for word in group if conf(word) >= 0]
        out = {
            "row_index": str(index),
            "row_type": row_type,
            "row_role": row_role,
            "section": active_section if row_type != "section_heading" else active_section,
            "top": str(band_top),
            "bottom": str(band_bottom),
            "word_top": str(min(int(word["top"]) for word in group)),
            "word_bottom": str(max(int(word["bottom"]) for word in group)),
            "left": str(min(int(word["left"]) for word in group)),
            "right": str(max(int(word["right"]) for word in group)),
            "word_count": str(len(group)),
            "mean_conf": f"{(sum(confs) / len(confs)):.2f}" if confs else "",
            "raw_row_text": raw_text,
        }
        out.update(cell_text)
        for name, _, _ in columns:
            out[f"{name}_source"] = "main" if out.get(name) else ""
            out[f"{name}_backup_hint"] = ""
            out[f"{name}_backup_source"] = ""
        merge_backup_cells(out, columns, full_backup_words, numeric_backup_words, price_backup_words)
        parsed.append(out)
    return parsed


def draw_overlay(
    image_path: Path,
    rows: list[dict[str, str]],
    columns: list[tuple[str, int, int]],
    detected_rules: list[int],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")
    h, _ = image.shape[:2]

    for x in detected_rules:
        cv2.line(image, (x, 0), (x, h), (220, 220, 220), 1)
    for _, x0, _ in columns:
        cv2.line(image, (x0, 0), (x0, h), (180, 180, 180), 1)
    cv2.line(image, (columns[-1][2], 0), (columns[-1][2], h), (180, 180, 180), 1)

    for row in rows:
        top = int(row["top"])
        bottom = int(row["bottom"])
        left = int(row["left"])
        right = int(row["right"])
        color = (0, 170, 255)
        if row["row_type"] == "section_heading":
            color = (0, 220, 0)
        elif row["row_type"] == "other_text":
            color = (80, 80, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image,
            row["row_index"],
            (max(2, left), max(16, top - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one azioni page into candidate structured rows.")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--full-backup-words", type=Path, default=DEFAULT_FULL_BACKUP_WORDS)
    parser.add_argument("--numeric-backup-words", type=Path, default=DEFAULT_NUMERIC_BACKUP_WORDS)
    parser.add_argument("--price-backup-words", type=Path, default=DEFAULT_PRICE_BACKUP_WORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body-top", type=int, default=235)
    parser.add_argument("--body-bottom", type=int, default=2990)
    parser.add_argument("--row-method", choices=["center", "issuer_anchor"], default="center")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read image: {args.image}")
    h, w = image.shape
    body_y0 = int(h * 0.075)

    words = read_csv(args.words)
    full_backup: list[dict[str, str]] = []
    numeric_backup: list[dict[str, str]] = []
    price_backup: list[dict[str, str]] = []
    if args.full_backup_words.exists():
        full_backup = read_csv(args.full_backup_words)
    if args.numeric_backup_words.exists():
        numeric_backup = shift_backup_words(read_csv(args.numeric_backup_words), x_offset=0, y_offset=body_y0)
    if args.price_backup_words.exists():
        price_backup = shift_backup_words(read_csv(args.price_backup_words), x_offset=int(w * 0.78), y_offset=body_y0)

    columns, detected_rules = build_columns(args.image, args.body_top, args.body_bottom, words)
    removed_row_words, median_word_height, max_word_height = row_grouping_diagnostics(
        words,
        args.body_top,
        args.body_bottom,
    )
    groups = group_rows(
        words,
        body_top=args.body_top,
        body_bottom=args.body_bottom,
        columns=columns,
        method=args.row_method,
    )
    parsed = parse_groups(groups, columns, full_backup, numeric_backup, price_backup, args.body_top, args.body_bottom)

    column_names = [name for name, _, _ in columns]
    fields = [
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
    ] + column_names + [f"{name}_source" for name in column_names] + [f"{name}_backup_hint" for name in column_names] + [f"{name}_backup_source" for name in column_names]
    write_csv(args.output_dir / "structured_rows_one_page.csv", parsed, fields)
    write_csv(
        args.output_dir / "candidate_security_rows_one_page.csv",
        [row for row in parsed if row["row_type"] == "candidate_security_row"],
        fields,
    )
    write_csv(
        args.output_dir / "detected_columns_one_page.csv",
        [{"column": name, "x0": str(start), "x1": str(end)} for name, start, end in columns],
        ["column", "x0", "x1"],
    )
    write_csv(
        args.output_dir / "detected_vertical_rules_one_page.csv",
        [{"x": str(x)} for x in detected_rules],
        ["x"],
    )
    write_csv(
        args.output_dir / "row_grouping_removed_words.csv",
        removed_row_words,
        ["text", "conf", "left", "top", "right", "bottom", "width", "height", "reason"],
    )
    draw_overlay(args.image, parsed, columns, detected_rules, args.output_dir / "structured_row_column_overlay.jpg")

    readme = [
        "# One-page structured azioni parse",
        "",
        "This is a prototype output, not the final dataset.",
        "",
        f"Input words: `{args.words}`",
        f"Input image: `{args.image}`",
        f"Full backup words: `{args.full_backup_words}`",
        f"Numeric backup words: `{args.numeric_backup_words}`",
        f"Price backup words: `{args.price_backup_words}`",
        f"Body range: y={args.body_top} to y={args.body_bottom}",
        f"Row grouping method: `{args.row_method}`",
        "",
        "Outputs:",
        "- `structured_rows_one_page.csv`: all grouped rows, including section headings.",
        "- `candidate_security_rows_one_page.csv`: rows that look like security observations.",
        "- `structured_row_column_overlay.jpg`: visual check of row grouping and column bands.",
        "- `detected_columns_one_page.csv`: dynamic column bands snapped to detected vertical table rules.",
        "- `detected_vertical_rules_one_page.csv`: raw detected vertical rule x-positions.",
        "- `row_grouping_removed_words.csv`: OCR artifacts removed before row clustering.",
        "",
        "Changes in this version:",
        "- Very tall OCR tokens that span multiple printed rows are removed before row grouping.",
        "- Column bands are snapped to detected printed vertical rules instead of being fully fixed.",
        "- Section headings use fuzzy matching for common OCR misspellings.",
        "- Full-page PSM 11, numeric-column, and price/quantity backup OCR fill missing numeric cells and leave backup hints for non-empty cells.",
        "",
        "Known limitations:",
        "- Some printed rows are still merged or split when OCR words fall on different baselines. `--row-method issuer_anchor` is available as an experimental alternative.",
        "- Column bands still use the expected azioni schema as a fallback when a printed rule is missed.",
        "- OCR text is raw; numeric normalization and issuer-name cleanup come later.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Detected vertical rules: {len(detected_rules)}")
    print(f"Median OCR word height: {median_word_height:.1f}; max row word height: {max_word_height}")
    print(f"Removed row-spanning artifacts: {len(removed_row_words)}")
    print(f"Row grouping method: {args.row_method}")
    print(f"Grouped rows: {len(parsed)}")
    print(f"Candidate security rows: {sum(row['row_type'] == 'candidate_security_row' for row in parsed)}")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
