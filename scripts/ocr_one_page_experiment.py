"""Run controlled OCR experiments on one azioni table crop.

The aim is to compare preprocessing and Tesseract settings before scaling a
choice to the full collection.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ocr_azioni_tables import DEFAULT_TESSDATA, find_tesseract, parse_tsv, preprocess_for_ocr, validate_languages


DEFAULT_CROP = Path("output/preprocess_1950_azioni/table_crops/1950-01-05_3_left_table.jpg")
DEFAULT_OUTPUT = Path("output/ocr_one_page_experiment")


NOISE_RE = re.compile(r"^[\W_]+$", re.UNICODE)
ITALIAN_TERMS = [
    "Finanziari",
    "Assicurazioni",
    "Trasporti",
    "Tessili",
    "Minerari",
    "Metallurgici",
    "Ansaldo",
    "FIAT",
    "Dalmine",
    "Montecatini",
    "Westinghouse",
]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_noise_token(text: str) -> bool:
    text = text.strip()
    return not text or bool(NOISE_RE.match(text))


def remove_table_rules(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy with long horizontal/vertical rules whitened out."""
    if len(gray.shape) != 2:
        raise ValueError("remove_table_rules expects a grayscale image")

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, w // 35), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 45)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    rules = cv2.bitwise_or(horizontal, vertical)
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    cleaned = gray.copy()
    cleaned[rules > 0] = 255
    return cleaned, rules


def run_tesseract(
    tesseract: str,
    image_path: Path,
    psm: int,
    lang: str,
    tessdata_dir: Path | None,
) -> str:
    cmd = [tesseract, str(image_path), "stdout"]
    if tessdata_dir is not None:
        cmd.extend(["--tessdata-dir", str(tessdata_dir)])
    cmd.extend(["-l", lang, "--oem", "1", "--psm", str(psm), "-c", "tessedit_create_tsv=1"])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout


def words_to_text(words: list[dict[str, str]]) -> str:
    ordered = sorted(words, key=lambda row: (int(row["top"]), int(row["left"])))
    return " ".join(row["text"] for row in ordered)


def summarize_words(variant: str, psm: int, words: list[dict[str, str]]) -> dict[str, str]:
    confs = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
    texts = [row["text"] for row in words]
    noise = [text for text in texts if is_noise_token(text)]
    numeric = [text for text in texts if re.search(r"\d", text)]
    alpha = [text for text in texts if re.search(r"[A-Za-zÀ-ÿ]", text)]
    text_blob = " ".join(texts).lower()
    hits = [term for term in ITALIAN_TERMS if term.lower() in text_blob]
    return {
        "variant": variant,
        "psm": str(psm),
        "word_count": str(len(words)),
        "mean_conf": f"{(sum(confs) / len(confs)):.2f}" if confs else "",
        "noise_token_count": str(len(noise)),
        "noise_token_share": f"{(len(noise) / len(words)):.3f}" if words else "",
        "numeric_token_count": str(len(numeric)),
        "alpha_token_count": str(len(alpha)),
        "known_term_hits": str(len(hits)),
        "known_terms": "; ".join(hits),
    }


def make_variants(crop_path: Path, out_dir: Path) -> dict[str, Path]:
    image = cv2.imread(str(crop_path))
    if image is None:
        raise OSError(f"Could not read crop: {crop_path}")

    preprocessed, _ = preprocess_for_ocr(image)
    line_removed, rule_mask = remove_table_rules(preprocessed)

    h, w = preprocessed.shape
    body_y0 = int(h * 0.075)
    footer_y0 = int(h * 0.90)
    body = line_removed[body_y0:footer_y0, :]

    variants: dict[str, np.ndarray] = {
        "original_full": preprocessed,
        "line_removed_full": line_removed,
        "body_line_removed": body,
        "numeric_columns_body": body[:, : int(w * 0.52)],
        "issuer_column_body": body[:, int(w * 0.48) : int(w * 0.80)],
        "price_quantity_columns_body": body[:, int(w * 0.78) :],
    }

    variant_paths: dict[str, Path] = {}
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / "detected_rule_mask.png"), rule_mask)
    for name, img in variants.items():
        path = image_dir / f"{name}.png"
        cv2.imwrite(str(path), img)
        variant_paths[name] = path
    return variant_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR variants on one page.")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tesseract", default=None)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="ita+eng")
    parser.add_argument("--psm", type=int, nargs="+", default=[6, 11, 12])
    args = parser.parse_args()

    tesseract = find_tesseract(args.tesseract)
    validate_languages(tesseract, args.lang, args.tessdata_dir)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_paths = make_variants(args.crop, args.output_dir)
    zone_row = {
        "source_file": args.crop.name,
        "date": "",
        "picture": "",
        "side": "",
        "zone_crop_file": str(args.crop),
    }

    summary: list[dict[str, str]] = []
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
    for variant, image_path in variant_paths.items():
        for psm in args.psm:
            stem = f"{variant}_psm{psm}"
            tsv_text = run_tesseract(tesseract, image_path, psm=psm, lang=args.lang, tessdata_dir=args.tessdata_dir)
            (args.output_dir / "raw_tsv" / f"{stem}.tsv").parent.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "raw_tsv" / f"{stem}.tsv").write_text(tsv_text, encoding="utf-8")

            words = parse_tsv(tsv_text, scale=1.0, zone_row=zone_row)
            write_csv(args.output_dir / "words" / f"{stem}_words.csv", words, word_fields)
            (args.output_dir / "text" / f"{stem}.txt").parent.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "text" / f"{stem}.txt").write_text(words_to_text(words), encoding="utf-8")
            summary.append(summarize_words(variant, psm, words))

    summary_fields = [
        "variant",
        "psm",
        "word_count",
        "mean_conf",
        "noise_token_count",
        "noise_token_share",
        "numeric_token_count",
        "alpha_token_count",
        "known_term_hits",
        "known_terms",
    ]
    write_csv(args.output_dir / "ocr_experiment_summary.csv", summary, summary_fields)

    md_lines = [
        "# One-page OCR experiment",
        "",
        f"Crop: `{args.crop}`",
        "",
        "The summary metrics are diagnostics, not ground truth. Prefer variants with a lower noise share, readable known terms, and enough numeric tokens.",
        "",
        "| Variant | PSM | Words | Mean confidence | Noise share | Numeric tokens | Known term hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        md_lines.append(
            f"| {row['variant']} | {row['psm']} | {row['word_count']} | {row['mean_conf']} | "
            f"{row['noise_token_share']} | {row['numeric_token_count']} | {row['known_term_hits']} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote experiment outputs to: {args.output_dir}")
    print(f"Summary: {args.output_dir / 'ocr_experiment_summary.csv'}")


if __name__ == "__main__":
    main()
