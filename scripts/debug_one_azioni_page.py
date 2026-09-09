"""Build a focused OCR debug package for one azioni table crop."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OCR_DIR = Path("output/ocr_1950_azioni")
DEFAULT_AZIONI_DIR = Path("output/preprocess_1950_azioni")
DEFAULT_OUTPUT_DIR = Path("output/debug_one_page")


NOISE_RE = re.compile(r"^[\W_]+$", re.UNICODE)
RULELIKE_RE = re.compile(r"^[\s_\-—–|{}\[\]().,;:!]+$", re.UNICODE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def is_noise_word(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    return bool(NOISE_RE.match(text) or RULELIKE_RE.match(text))


def numeric_conf(row: dict[str, str]) -> float:
    try:
        return float(row["conf"])
    except ValueError:
        return -1.0


def mid_y(row: dict[str, str]) -> int:
    return (int(row["top"]) + int(row["bottom"])) // 2


def mid_x(row: dict[str, str]) -> int:
    return (int(row["left"]) + int(row["right"])) // 2


def useful_word(row: dict[str, str]) -> bool:
    text = row["text"].strip()
    if is_noise_word(text):
        return False
    if numeric_conf(row) < 15:
        return False
    height = int(row["bottom"]) - int(row["top"])
    width = int(row["right"]) - int(row["left"])
    if width <= 1 or height <= 1:
        return False
    return True


def group_coordinate_rows(words: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    kept = [row for row in words if useful_word(row)]
    heights = [int(row["bottom"]) - int(row["top"]) for row in kept]
    median_height = float(np.median(heights)) if heights else 20.0
    row_gap = max(10, int(round(median_height * 0.85)))

    groups: list[list[dict[str, str]]] = []
    centers: list[int] = []
    for word in sorted(kept, key=lambda row: (mid_y(row), int(row["left"]))):
        center = mid_y(word)
        if not groups or abs(center - centers[-1]) > row_gap:
            groups.append([word])
            centers.append(center)
        else:
            groups[-1].append(word)
            centers[-1] = int(round(sum(mid_y(item) for item in groups[-1]) / len(groups[-1])))
    return groups


def coordinate_row_records(words: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, group in enumerate(group_coordinate_rows(words), start=1):
        group = sorted(group, key=lambda row: int(row["left"]))
        text = " ".join(row["text"] for row in group)
        records.append(
            {
                "row_index": str(index),
                "top": str(min(int(row["top"]) for row in group)),
                "bottom": str(max(int(row["bottom"]) for row in group)),
                "left": str(min(int(row["left"]) for row in group)),
                "right": str(max(int(row["right"]) for row in group)),
                "word_count": str(len(group)),
                "mean_conf": f"{sum(numeric_conf(row) for row in group) / len(group):.2f}",
                "text": text,
            }
        )
    return records


def filter_rows(rows: list[dict[str, str]], crop_file: str) -> list[dict[str, str]]:
    normalized = crop_file.replace("/", "\\")
    return [row for row in rows if row.get("zone_crop_file", "").replace("/", "\\") == normalized]


def draw_word_overlay(image_path: Path, words: list[dict[str, str]], output_path: Path, noise_only: bool) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")

    for word in words:
        noise = is_noise_word(word["text"])
        if noise_only and not noise:
            continue
        left = int(word["left"])
        top = int(word["top"])
        right = int(word["right"])
        bottom = int(word["bottom"])
        color = (0, 0, 255) if noise else (255, 80, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        if noise:
            label = word["text"][:10]
            cv2.putText(
                image,
                label,
                (left, max(18, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_line_overlay(image_path: Path, lines: list[dict[str, str]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")

    for line in lines:
        left = int(line["left"])
        top = int(line["top"])
        right = int(line["right"])
        bottom = int(line["bottom"])
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_coordinate_row_overlay(image_path: Path, rows: list[dict[str, str]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")

    for row in rows:
        left = int(row["left"])
        top = int(row["top"])
        right = int(row["right"])
        bottom = int(row["bottom"])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 160, 255), 2)
        cv2.putText(
            image,
            row["row_index"],
            (left, max(18, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one-page OCR debug files.")
    parser.add_argument("--crop-file", default=r"table_crops\1950-01-05_3_left_table.jpg")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--azioni-dir", type=Path, default=DEFAULT_AZIONI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    crop_file = args.crop_file.replace("/", "\\")
    stem = Path(crop_file).stem
    out_dir = args.output_dir / stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_path = args.azioni_dir / crop_file
    if not crop_path.exists():
        raise FileNotFoundError(f"Table crop not found: {crop_path}")

    words = filter_rows(read_csv(args.ocr_dir / "ocr_words.csv"), crop_file)
    lines = filter_rows(read_csv(args.ocr_dir / "ocr_lines.csv"), crop_file)
    if not words:
        raise SystemExit(f"No OCR words found for {crop_file}. Run scripts/ocr_azioni_tables.py first.")

    word_fields = list(words[0].keys())
    line_fields = list(lines[0].keys()) if lines else []
    noise_words = [row for row in words if is_noise_word(row["text"])]
    coordinate_rows = coordinate_row_records(words)

    shutil.copy2(crop_path, out_dir / "table_crop.jpg")
    preprocessed = args.ocr_dir / "preprocessed_table_crops" / f"{stem}_ocr.png"
    if preprocessed.exists():
        shutil.copy2(preprocessed, out_dir / "ocr_preprocessed.png")

    write_csv(out_dir / "ocr_words_one_page.csv", words, word_fields)
    write_csv(out_dir / "ocr_words_one_page.tsv", words, word_fields, delimiter="\t")
    if lines:
        write_csv(out_dir / "ocr_lines_one_page.csv", lines, line_fields)
        write_csv(out_dir / "ocr_lines_one_page.tsv", lines, line_fields, delimiter="\t")
    write_csv(out_dir / "punctuation_and_rule_noise_words.csv", noise_words, word_fields)
    write_csv(out_dir / "punctuation_and_rule_noise_words.tsv", noise_words, word_fields, delimiter="\t")
    coordinate_fields = ["row_index", "top", "bottom", "left", "right", "word_count", "mean_conf", "text"]
    write_csv(out_dir / "coordinate_rows_one_page.csv", coordinate_rows, coordinate_fields)
    write_csv(out_dir / "coordinate_rows_one_page.tsv", coordinate_rows, coordinate_fields, delimiter="\t")

    draw_word_overlay(crop_path, words, out_dir / "word_overlay_blue_text_red_noise.jpg", noise_only=False)
    draw_word_overlay(crop_path, words, out_dir / "noise_overlay_red_only.jpg", noise_only=True)
    if lines:
        draw_line_overlay(crop_path, lines, out_dir / "line_overlay.jpg")
    draw_coordinate_row_overlay(crop_path, coordinate_rows, out_dir / "coordinate_row_overlay.jpg")

    print(f"Debug page: {crop_file}")
    print(f"Words: {len(words)}")
    print(f"Lines: {len(lines)}")
    print(f"Punctuation/rule-like OCR words: {len(noise_words)}")
    print(f"Coordinate-grouped rows: {len(coordinate_rows)}")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
