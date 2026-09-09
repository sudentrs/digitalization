"""OCR conservative azioni table crops and keep word coordinates.

This stage deliberately works after zone extraction. It does not try to solve
table structure directly; it creates coordinate-rich OCR outputs that later
row/column parsers can use.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_AZIONI_DIR = Path("output/preprocess_1950_azioni")
DEFAULT_OUTPUT = Path("output/ocr_1950_azioni")
DEFAULT_TESSDATA = Path("tools/tessdata")


def find_tesseract(explicit_path: str | None) -> str:
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    found = shutil.which("tesseract")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit(
        "Could not find tesseract.exe. Install Tesseract OCR or pass "
        "--tesseract \"C:\\Path\\to\\tesseract.exe\"."
    )


def available_languages(tesseract: str, tessdata_dir: Path | None) -> set[str]:
    cmd = [tesseract, "--list-langs"]
    if tessdata_dir is not None:
        cmd.extend(["--tessdata-dir", str(tessdata_dir)])
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    langs: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and not line.lower().startswith("list of available"):
            langs.add(line)
    return langs


def validate_languages(tesseract: str, lang: str, tessdata_dir: Path | None) -> None:
    requested = [part for part in lang.split("+") if part]
    available = available_languages(tesseract, tessdata_dir)
    missing = [part for part in requested if part not in available]
    if missing:
        raise SystemExit(
            "Tesseract is installed, but these language packs are missing: "
            + ", ".join(missing)
            + f". Available: {', '.join(sorted(available))}"
        )


def load_table_zones(azioni_dir: Path, limit: int | None) -> list[dict[str, str]]:
    manifest_path = azioni_dir / "azioni_zone_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as fp:
        rows = [row for row in csv.DictReader(fp) if row["zone_type"] == "azioni_table"]
    if limit is not None:
        rows = rows[:limit]
    return rows


def preprocess_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    target_width = 2600
    scale = 1.0
    if w < target_width:
        scale = target_width / float(w)
        gray = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    kernel_size = max(51, (min(gray.shape) // 18) | 1)
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    normalized = cv2.divide(gray, background, scale=245)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    return enhanced, scale


def run_tesseract_tsv(
    tesseract: str,
    image_path: Path,
    lang: str,
    psm: int,
    tessdata_dir: Path | None,
) -> str:
    cmd = [
        tesseract,
        str(image_path),
        "stdout",
    ]
    if tessdata_dir is not None:
        cmd.extend(["--tessdata-dir", str(tessdata_dir)])
    cmd.extend(
        [
        "-l",
        lang,
        "--oem",
        "1",
        "--psm",
        str(psm),
        "-c",
        "tessedit_create_tsv=1",
        ]
    )
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def parse_tsv(tsv_text: str, scale: float, zone_row: dict[str, str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    # Tesseract TSV should have exactly 12 columns. A few historical-table
    # pages can produce control characters that confuse csv.DictReader and
    # swallow many TSV rows into one "text" field, so parse defensively.
    expected_fields = [
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ]
    for raw_line in tsv_text.splitlines():
        if not raw_line or raw_line.startswith("level\t"):
            continue
        parts = raw_line.split("\t", 11)
        if len(parts) != len(expected_fields):
            continue
        row = dict(zip(expected_fields, parts))
        if row["level"] != "5":
            continue
        text = row["text"].strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf", "-1"))
            left_raw = int(row["left"])
            top_raw = int(row["top"])
            width_raw = int(row["width"])
            height_raw = int(row["height"])
        except ValueError:
            continue
        if conf < 0:
            continue

        left = int(round(left_raw / scale))
        top = int(round(top_raw / scale))
        width = int(round(width_raw / scale))
        height = int(round(height_raw / scale))
        parsed.append(
            {
                "source_file": zone_row["source_file"],
                "date": zone_row["date"],
                "picture": zone_row["picture"],
                "side": zone_row["side"],
                "zone_crop_file": zone_row["zone_crop_file"],
                "block_num": row["block_num"],
                "par_num": row["par_num"],
                "line_num": row["line_num"],
                "word_num": row["word_num"],
                "conf": f"{conf:.2f}",
                "left": str(left),
                "top": str(top),
                "right": str(left + width),
                "bottom": str(top + height),
                "width": str(width),
                "height": str(height),
                "text": text,
            }
        )
    return parsed


def line_rows(words: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for word in words:
        key = (
            word["source_file"],
            word["side"],
            word["block_num"],
            word["par_num"],
            word["line_num"],
            word["zone_crop_file"],
        )
        grouped[key].append(word)

    lines: list[dict[str, str]] = []
    for key, group in grouped.items():
        group.sort(key=lambda row: (int(row["left"]), int(row["top"])))
        source_file, side, block_num, par_num, line_num, crop_file = key
        left = min(int(row["left"]) for row in group)
        top = min(int(row["top"]) for row in group)
        right = max(int(row["right"]) for row in group)
        bottom = max(int(row["bottom"]) for row in group)
        confs = [float(row["conf"]) for row in group]
        lines.append(
            {
                "source_file": source_file,
                "side": side,
                "zone_crop_file": crop_file,
                "block_num": block_num,
                "par_num": par_num,
                "line_num": line_num,
                "left": str(left),
                "top": str(top),
                "right": str(right),
                "bottom": str(bottom),
                "mean_conf": f"{sum(confs) / len(confs):.2f}",
                "text": " ".join(row["text"] for row in group),
            }
        )
    return sorted(lines, key=lambda row: (row["source_file"], row["side"], int(row["top"]), int(row["left"])))


def column_candidates(words: list[dict[str, str]]) -> list[dict[str, str]]:
    by_page: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for word in words:
        by_page[(word["source_file"], word["side"], word["zone_crop_file"])].append(word)

    candidates: list[dict[str, str]] = []
    for (source_file, side, crop_file), group in by_page.items():
        centers = sorted((int(row["left"]) + int(row["right"])) // 2 for row in group)
        if not centers:
            continue
        clusters: list[list[int]] = []
        for center in centers:
            if not clusters or center - clusters[-1][-1] > 35:
                clusters.append([center])
            else:
                clusters[-1].append(center)
        for index, cluster in enumerate(clusters, start=1):
            if len(cluster) < 4:
                continue
            candidates.append(
                {
                    "source_file": source_file,
                    "side": side,
                    "zone_crop_file": crop_file,
                    "candidate_col": str(index),
                    "x_center_min": str(min(cluster)),
                    "x_center_max": str(max(cluster)),
                    "x_center_mean": f"{sum(cluster) / len(cluster):.1f}",
                    "word_count": str(len(cluster)),
                }
            )
    return candidates


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process(
    azioni_dir: Path,
    output_dir: Path,
    tesseract_path: str | None,
    tessdata_dir: Path | None,
    lang: str,
    psm: int,
    limit: int | None,
) -> None:
    tesseract = find_tesseract(tesseract_path)
    if tessdata_dir is not None and not tessdata_dir.exists():
        raise SystemExit(f"Tessdata directory does not exist: {tessdata_dir}")
    validate_languages(tesseract, lang, tessdata_dir)

    preprocessed_dir = output_dir / "preprocessed_table_crops"
    raw_tsv_dir = output_dir / "raw_tsv"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    raw_tsv_dir.mkdir(parents=True, exist_ok=True)

    words: list[dict[str, str]] = []
    table_zones = load_table_zones(azioni_dir, limit)
    for zone in table_zones:
        crop_path = azioni_dir / zone["zone_crop_file"]
        image = cv2.imread(str(crop_path))
        if image is None:
            raise OSError(f"Could not read table crop: {crop_path}")

        stem = Path(zone["zone_crop_file"]).stem
        processed, scale = preprocess_for_ocr(image)
        processed_path = preprocessed_dir / f"{stem}_ocr.png"
        cv2.imwrite(str(processed_path), processed)

        tsv_text = run_tesseract_tsv(tesseract, processed_path, lang=lang, psm=psm, tessdata_dir=tessdata_dir)
        (raw_tsv_dir / f"{stem}.tsv").write_text(tsv_text, encoding="utf-8")
        words.extend(parse_tsv(tsv_text, scale=scale, zone_row=zone))

    word_fields = [
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
    write_csv(output_dir / "ocr_words.csv", words, word_fields)

    lines = line_rows(words)
    write_csv(
        output_dir / "ocr_lines.csv",
        lines,
        [
            "source_file",
            "side",
            "zone_crop_file",
            "block_num",
            "par_num",
            "line_num",
            "left",
            "top",
            "right",
            "bottom",
            "mean_conf",
            "text",
        ],
    )

    columns = column_candidates(words)
    write_csv(
        output_dir / "column_candidates.csv",
        columns,
        [
            "source_file",
            "side",
            "zone_crop_file",
            "candidate_col",
            "x_center_min",
            "x_center_max",
            "x_center_mean",
            "word_count",
        ],
    )

    print(f"Processed {len(table_zones)} table crops")
    print(f"Wrote words: {output_dir / 'ocr_words.csv'}")
    print(f"Wrote lines: {output_dir / 'ocr_lines.csv'}")
    print(f"Wrote column candidates: {output_dir / 'column_candidates.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR regular_azioni table crops with word coordinates.")
    parser.add_argument("--azioni-dir", type=Path, default=DEFAULT_AZIONI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tesseract", default=None, help="Optional full path to tesseract.exe")
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="ita+eng", help="Tesseract language setting, e.g. ita+eng or eng")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    process(args.azioni_dir, args.output_dir, args.tesseract, args.tessdata_dir, args.lang, args.psm, args.limit)


if __name__ == "__main__":
    main()
