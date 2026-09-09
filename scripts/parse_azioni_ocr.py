"""Create review/debug tables from azioni OCR words.

These outputs are intermediate diagnostics, not the final research dataset.
They keep the OCR coordinates visible so the next parser can be improved
without pretending that noisy OCR already forms clean stock-level rows.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_OCR_DIR = Path("output/ocr_1950_azioni")
DEFAULT_OUTPUT = Path("output/parsed_1950_azioni")


COLUMNS = [
    ("capitale", 0.000, 0.105),
    ("numero_azioni", 0.105, 0.195),
    ("valore_nominale", 0.195, 0.255),
    ("godimento", 0.255, 0.335),
    ("cedola_data", 0.335, 0.395),
    ("importo_lordo", 0.395, 0.435),
    ("importo_acconto", 0.435, 0.485),
    ("importo_saldo", 0.485, 0.535),
    ("numero", 0.535, 0.580),
    ("prezzo_compenso", 0.580, 0.655),
    ("titolo", 0.655, 0.800),
    ("prezzi_minimi", 0.800, 0.870),
    ("prezzi_massimi", 0.870, 0.930),
    ("prezzi_chiusura", 0.930, 0.975),
    ("quantitativi", 0.975, 1.000),
]


def load_words(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def numeric_conf(row: dict[str, str]) -> float:
    try:
        return float(row["conf"])
    except ValueError:
        return -1.0


def clean_display_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # The OCR often invents table-border pipes/brackets. Keep the original
    # text in ocr_words.csv; make review files easier to scan.
    text = re.sub(r"[|{}\[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spreadsheet_safe(text: str) -> str:
    text = clean_display_text(text)
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def looks_like_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 3:
        return True
    letters_or_digits = sum(ch.isalnum() for ch in compact)
    return letters_or_digits < max(2, len(compact) * 0.25)


def has_data_signal(text: str) -> bool:
    return bool(re.search(r"\d", text) or re.search(r"[A-ZÀ-Ü][a-zà-ü]{2,}", text))


def page_width(words: list[dict[str, str]]) -> int:
    return max(int(row["right"]) for row in words) if words else 1


def word_mid_y(row: dict[str, str]) -> int:
    return (int(row["top"]) + int(row["bottom"])) // 2


def word_mid_x(row: dict[str, str]) -> int:
    return (int(row["left"]) + int(row["right"])) // 2


def group_rows(words: list[dict[str, str]], gap: int = 16) -> list[list[dict[str, str]]]:
    words = sorted(words, key=lambda row: (word_mid_y(row), int(row["left"])))
    groups: list[list[dict[str, str]]] = []
    centers: list[int] = []
    for word in words:
        center = word_mid_y(word)
        if not groups or abs(center - centers[-1]) > gap:
            groups.append([word])
            centers.append(center)
        else:
            groups[-1].append(word)
            centers[-1] = int(round(sum(word_mid_y(item) for item in groups[-1]) / len(groups[-1])))
    return groups


def assign_column(row: dict[str, str], width: int) -> str:
    x = word_mid_x(row) / float(width)
    for name, start, end in COLUMNS:
        if start <= x < end:
            return name
    return COLUMNS[-1][0]


def parse_rows(words: list[dict[str, str]]) -> list[dict[str, str]]:
    by_page: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for word in words:
        # Drop very low-confidence isolated noise, but keep zero-confidence
        # tokens out of the structured parse.
        if numeric_conf(word) <= 0:
            continue
        by_page[(word["source_file"], word["side"], word["zone_crop_file"])].append(word)

    parsed: list[dict[str, str]] = []
    for (source_file, side, crop_file), page_words in by_page.items():
        width = page_width(page_words)
        row_groups = group_rows(page_words)
        data_row_index = 0
        for group in row_groups:
            top = min(int(word["top"]) for word in group)
            bottom = max(int(word["bottom"]) for word in group)

            # Skip the broad title/header area only for the row parse. The OCR
            # words remain available in ocr_words.csv.
            if top < 170:
                continue

            cells: dict[str, list[dict[str, str]]] = {name: [] for name, _, _ in COLUMNS}
            for word in group:
                cells[assign_column(word, width)].append(word)

            row_text = " ".join(word["text"] for word in sorted(group, key=lambda row: int(row["left"])))
            if looks_like_noise(row_text):
                continue

            data_row_index += 1
            out = {
                "source_file": source_file,
                "side": side,
                "zone_crop_file": crop_file,
                "row_index": str(data_row_index),
                "top": str(top),
                "bottom": str(bottom),
                "raw_text": spreadsheet_safe(row_text),
            }
            for name, _, _ in COLUMNS:
                cell_words = sorted(cells[name], key=lambda row: int(row["left"]))
                out[name] = spreadsheet_safe(" ".join(word["text"] for word in cell_words))
            parsed.append(out)
    return parsed


def build_review_lines(words: list[dict[str, str]]) -> list[dict[str, str]]:
    by_page: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for word in words:
        if numeric_conf(word) <= 0:
            continue
        by_page[(word["source_file"], word["side"], word["zone_crop_file"])].append(word)

    review: list[dict[str, str]] = []
    for (source_file, side, crop_file), page_words in by_page.items():
        for idx, group in enumerate(group_rows(page_words), start=1):
            group = sorted(group, key=lambda row: int(row["left"]))
            text = " ".join(word["text"] for word in group)
            if looks_like_noise(text) or not has_data_signal(text):
                continue
            confs = [numeric_conf(word) for word in group if numeric_conf(word) > 0]
            top = min(int(word["top"]) for word in group)
            bottom = max(int(word["bottom"]) for word in group)
            left = min(int(word["left"]) for word in group)
            right = max(int(word["right"]) for word in group)
            review.append(
                {
                    "source_file": source_file,
                    "side": side,
                    "zone_crop_file": crop_file,
                    "line_index": str(idx),
                    "top": str(top),
                    "bottom": str(bottom),
                    "left": str(left),
                    "right": str(right),
                    "mean_conf": f"{(sum(confs) / len(confs)):.2f}" if confs else "",
                    "line_type": classify_line(text, top),
                    "review_text": spreadsheet_safe(text),
                }
            )
    return review


def classify_line(text: str, top: int) -> str:
    cleaned = clean_display_text(text).lower()
    if top < 190 or any(term in cleaned for term in ("titoli azionari", "capitale", "godimento")):
        return "header_or_column_label"
    if any(term in cleaned for term in ("finanziari", "assicurativi", "trasporti", "tessili", "minerari", "meccanici")):
        return "section_heading"
    if re.search(r"\d", cleaned):
        return "candidate_data_line"
    return "text_line"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review/debug tables from azioni OCR words.")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    words = load_words(args.ocr_dir / "ocr_words.csv")
    parsed = parse_rows(words)
    review = build_review_lines(words)

    parsed_fields = [
        "source_file",
        "side",
        "zone_crop_file",
        "row_index",
        "top",
        "bottom",
        "raw_text",
    ] + [name for name, _, _ in COLUMNS]
    review_fields = [
        "source_file",
        "side",
        "zone_crop_file",
        "line_index",
        "top",
        "bottom",
        "left",
        "right",
        "mean_conf",
        "line_type",
        "review_text",
    ]

    write_csv(args.output_dir / "parsed_rows_debug.csv", parsed, parsed_fields)
    write_csv(args.output_dir / "review_lines.csv", review, review_fields)
    write_tsv(args.output_dir / "review_lines.tsv", review, review_fields)
    print(f"Parsed {len(parsed)} debug row candidates")
    print(f"Built {len(review)} review lines")
    print(f"Wrote: {args.output_dir / 'parsed_rows_debug.csv'}")
    print(f"Wrote: {args.output_dir / 'review_lines.csv'}")
    print(f"Wrote: {args.output_dir / 'review_lines.tsv'}")


if __name__ == "__main__":
    main()
