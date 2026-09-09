from __future__ import annotations

import csv
import html
import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "output" / "batch_30_azioni"

HIGH_FLAG_REASONS = ("missing_capital", "missing_nominal", "missing_issuer")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_flags(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def high_reason(row: dict[str, str]) -> list[str]:
    flags = split_flags(row.get("validation_flags", ""))
    return [flag for flag in HIGH_FLAG_REASONS if flag in flags]


def problem_category(row: dict[str, str], structured: dict[str, str] | None) -> str:
    flags = set(split_flags(row.get("validation_flags", "")))
    role = row.get("row_role", "")
    raw = row.get("raw_row_text", "")
    top = float((structured or {}).get("top") or 0)
    bottom = float((structured or {}).get("bottom") or 0)
    page_h = float((structured or {}).get("_image_height") or 1)
    lower_page = bottom / page_h > 0.78 if page_h else False

    if "missing_issuer" in flags and role == "security_continuation_row":
        return "continuation row needs parent context"
    if lower_page and ({"missing_capital", "missing_nominal", "missing_shares"} & flags):
        return "bottom-row crowding / identity OCR loss"
    if "missing_nominal" in flags and any(ch.isdigit() for ch in raw):
        return "nominal validation or glued identity cells"
    if "missing_capital" in flags and not row.get("capital_subscribed"):
        return "left identity-column OCR loss"
    if "missing_issuer" in flags:
        return "issuer text missing or continuation semantics"
    return "general high-priority identity issue"


def draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, canvas.width, 33], fill=(245, 247, 250))
    draw.text((10, 7), label, fill=(180, 0, 0), font=font)
    return canvas


def copy_context_images(batch: Path, out_dir: Path, sample_id: str) -> dict[str, str]:
    sample_dir = batch / "samples" / sample_id
    copied = {}
    for name in ["structured_overlay.jpg", "images/original_full.png"]:
        src = sample_dir / name
        if src.exists():
            dest = out_dir / "context_images" / sample_id / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied[name] = str(dest.relative_to(out_dir)).replace("\\", "/")
    return copied


def build(batch: Path):
    cleaned_path = batch / "combined_cleaned_candidate_rows.csv"
    structured_path = batch / "combined_structured_rows.csv"
    out_dir = batch / "high_priority_review_packet"
    snippets_dir = out_dir / "row_snippets"

    out_dir.mkdir(parents=True, exist_ok=True)
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "context_images").mkdir(parents=True, exist_ok=True)

    cleaned = read_csv(cleaned_path)
    structured_rows = read_csv(structured_path)
    structured_by_key = {
        (row.get("sample_id", ""), row.get("row_index", "")): row
        for row in structured_rows
    }

    high_rows = [row for row in cleaned if high_reason(row)]

    rows_out: list[dict[str, str]] = []
    category_counts = Counter()
    sample_counts = Counter()
    section_counts = Counter()
    context_cache: dict[str, dict[str, str]] = {}

    for idx, row in enumerate(high_rows, start=1):
        sample_id = row["sample_id"]
        row_index = row["row_index"]
        structured = structured_by_key.get((sample_id, row_index))
        sample_dir = batch / "samples" / sample_id
        image_path = sample_dir / "images" / "original_full.png"
        snippet_rel = ""

        if image_path.exists() and structured:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                structured["_image_height"] = str(im.height)
                top = int(float(structured.get("top") or 0))
                bottom = int(float(structured.get("bottom") or top + 40))
                pad = max(36, int((bottom - top) * 1.2))
                crop_top = max(0, top - pad)
                crop_bottom = min(im.height, bottom + pad)
                crop = im.crop((0, crop_top, im.width, crop_bottom))
                draw = ImageDraw.Draw(crop)
                y1 = top - crop_top
                y2 = bottom - crop_top
                draw.rectangle([2, y1, im.width - 3, y2], outline=(255, 165, 0), width=4)
                label = f"{sample_id} row {row_index} | {', '.join(high_reason(row))}"
                crop = draw_label(crop, label)
                snippet_name = f"{idx:03d}_{sample_id}_row_{row_index}.jpg"
                snippet_path = snippets_dir / snippet_name
                crop.save(snippet_path, quality=92)
                snippet_rel = str(snippet_path.relative_to(out_dir)).replace("\\", "/")

        if sample_id not in context_cache:
            context_cache[sample_id] = copy_context_images(batch, out_dir, sample_id)

        category = problem_category(row, structured)
        category_counts[category] += 1
        sample_counts[sample_id] += 1
        section_counts[row.get("section", "") or "(blank section)"] += 1

        rows_out.append(
            {
                "sample_id": sample_id,
                "row_index": row_index,
                "section": row.get("section", ""),
                "row_role": row.get("row_role", ""),
                "priority_reason": ";".join(high_reason(row)),
                "problem_category": category,
                "issuer_name_clean": row.get("issuer_name_clean", ""),
                "capital_subscribed": row.get("capital_subscribed", ""),
                "shares_outstanding": row.get("shares_outstanding", ""),
                "nominal_value": row.get("nominal_value", ""),
                "price_close": row.get("price_close", ""),
                "quantity_traded": row.get("quantity_traded", ""),
                "validation_flags": row.get("validation_flags", ""),
                "raw_row_text": row.get("raw_row_text", ""),
                "snippet": snippet_rel,
                "overlay": context_cache[sample_id].get("structured_overlay.jpg", ""),
                "original_full": context_cache[sample_id].get("images/original_full.png", ""),
            }
        )

    out_csv = out_dir / "high_priority_review_packet_index.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "row_index",
            "section",
            "row_role",
            "priority_reason",
            "problem_category",
            "issuer_name_clean",
            "capital_subscribed",
            "shares_outstanding",
            "nominal_value",
            "price_close",
            "quantity_traded",
            "validation_flags",
            "raw_row_text",
            "snippet",
            "overlay",
            "original_full",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary_md = out_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# High-Priority Review Packet\n\n")
        f.write(f"Rows included: {len(rows_out)}\n\n")
        f.write("## Problem categories\n\n")
        for name, count in category_counts.most_common():
            f.write(f"- {name}: {count}\n")
        f.write("\n## Top samples\n\n")
        for name, count in sample_counts.most_common(15):
            f.write(f"- {name}: {count}\n")
        f.write("\n## Top sections\n\n")
        for name, count in section_counts.most_common(15):
            f.write(f"- {name}: {count}\n")

    html_path = out_dir / "index.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>High-Priority Azioni Review Packet</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1d1d1d}"
            "h1{margin-bottom:4px} .meta{color:#555;margin-bottom:24px}"
            ".summary{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0 24px}"
            ".box{background:white;border:1px solid #ddd;border-radius:6px;padding:12px 14px;min-width:220px}"
            ".row{background:white;border:1px solid #d8d8d8;border-radius:6px;margin:16px 0;padding:14px}"
            ".row h3{margin:0 0 8px;color:#0b4f8a}.fields{font-size:13px;line-height:1.45}"
            ".flag{color:#9b1c1c;font-weight:bold}.raw{font-family:Consolas,monospace;background:#f2f4f7;padding:8px;white-space:pre-wrap}"
            "img{max-width:100%;height:auto;border:1px solid #ccc;margin-top:10px}"
            "a{color:#0b4f8a}"
            "</style></head><body>"
        )
        f.write("<h1>High-Priority Azioni Review Packet</h1>")
        f.write(f"<div class='meta'>{len(rows_out)} rows with missing issuer, capital, or nominal value.</div>")
        f.write("<div class='summary'>")
        f.write("<div class='box'><b>Top problem categories</b><br>")
        for name, count in category_counts.most_common(6):
            f.write(f"{html.escape(name)}: {count}<br>")
        f.write("</div><div class='box'><b>Top samples</b><br>")
        for name, count in sample_counts.most_common(6):
            f.write(f"{html.escape(name)}: {count}<br>")
        f.write("</div></div>")

        for row in rows_out:
            f.write("<div class='row'>")
            title = f"{row['sample_id']} row {row['row_index']} - {row['problem_category']}"
            f.write(f"<h3>{html.escape(title)}</h3>")
            f.write("<div class='fields'>")
            f.write(f"<b>Section:</b> {html.escape(row['section'])} &nbsp; ")
            f.write(f"<b>Role:</b> {html.escape(row['row_role'])}<br>")
            f.write(f"<b>Priority:</b> <span class='flag'>{html.escape(row['priority_reason'])}</span><br>")
            f.write(
                "<b>Parsed:</b> "
                f"issuer={html.escape(row['issuer_name_clean'])}; "
                f"capital={html.escape(row['capital_subscribed'])}; "
                f"shares={html.escape(row['shares_outstanding'])}; "
                f"nominal={html.escape(row['nominal_value'])}; "
                f"close={html.escape(row['price_close'])}; "
                f"quantity={html.escape(row['quantity_traded'])}<br>"
            )
            f.write(f"<b>Flags:</b> {html.escape(row['validation_flags'])}<br>")
            f.write(f"<div class='raw'>{html.escape(row['raw_row_text'])}</div>")
            if row["overlay"]:
                f.write(f"<a href='{row['overlay']}'>structured overlay</a> ")
            if row["original_full"]:
                f.write(f"<a href='{row['original_full']}'>table crop</a>")
            f.write("</div>")
            if row["snippet"]:
                f.write(f"<img src='{row['snippet']}' alt='row snippet'>")
            f.write("</div>")
        f.write("</body></html>")

    print(f"rows={len(rows_out)}")
    print(out_csv)
    print(summary_md)
    print(html_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a high-priority visual review packet.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    build(args.batch_dir)


if __name__ == "__main__":
    main()
