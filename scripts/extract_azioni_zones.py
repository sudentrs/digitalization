"""Extract table zones for pages classified as regular_azioni.

This is the first layout-specific pass. It only handles pages headed
TITOLI AZIONARI, because those pages repeat a stable financial-table layout.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from preprocess_scans import (
    LayoutZone,
    binarize_ink,
    crop_half_pages,
    estimate_content_bbox,
    normalize_page_for_analysis,
    parse_scan_name,
    resize_for_preview,
    scale_zone,
    write_image,
)


DEFAULT_LABELS = Path("output/preprocess_1950/page_type_labels.csv")
DEFAULT_INPUT = Path("1950")
DEFAULT_OUTPUT = Path("output/preprocess_1950_azioni")


def cluster_indices(indices: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(indices[0])
    last = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current - last <= max_gap:
            last = current
        else:
            clusters.append((start, last + 1))
            start = current
            last = current
    clusters.append((start, last + 1))
    return clusters


def span_from_clusters(
    clusters: list[tuple[int, int]],
    min_value: int,
    max_value: int,
) -> tuple[int, int] | None:
    usable = [(start, end) for start, end in clusters if end >= min_value and start <= max_value]
    if not usable:
        return None
    return min(start for start, _ in usable), max(end for _, end in usable)


def projection_span(
    score: np.ndarray,
    min_value: int,
    max_value: int,
    percentile: float,
    min_threshold: float,
    max_gap: int,
) -> tuple[int, int] | None:
    threshold = max(min_threshold, float(np.percentile(score, percentile)))
    clusters = cluster_indices(np.flatnonzero(score >= threshold), max_gap=max_gap)
    return span_from_clusters(clusters, min_value=min_value, max_value=max_value)


def strongest_horizontal_cluster(
    clusters: list[tuple[int, int]],
    score: np.ndarray,
    min_value: int,
    max_value: int,
) -> tuple[int, int] | None:
    candidates = [(start, end) for start, end in clusters if end >= min_value and start <= max_value]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(score[item[0] : item[1]].max()))


def scale_bbox_to_analysis(
    bbox: tuple[int, int, int, int],
    scale_to_page: float,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(w, int(round(x0 / scale_to_page)))),
        max(0, min(h, int(round(y0 / scale_to_page)))),
        max(0, min(w, int(round(x1 / scale_to_page)))),
        max(0, min(h, int(round(y1 / scale_to_page)))),
    )


def detect_azioni_table_zone(
    normalized_gray: np.ndarray,
    content_bbox: tuple[int, int, int, int] | None = None,
) -> list[LayoutZone]:
    """Detect the main ruled stock table on a normalized regular_azioni page."""
    h, w = normalized_gray.shape
    binary = binarize_ink(normalized_gray)

    margin_x = max(12, int(w * 0.025))
    margin_y = max(12, int(h * 0.020))
    work = binary.copy()
    work[:, :margin_x] = 0
    work[:, w - margin_x :] = 0
    work[:margin_y, :] = 0
    work[h - margin_y :, :] = 0

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(90, w // 7), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(110, h // 8)))
    header_vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 25)))
    horizontal = cv2.morphologyEx(work, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(work, cv2.MORPH_OPEN, vertical_kernel)
    header_vertical = cv2.morphologyEx(work, cv2.MORPH_OPEN, header_vertical_kernel)

    row_score = (horizontal > 0).mean(axis=1)
    col_score = (vertical > 0).mean(axis=0)
    vertical_row_score = (vertical > 0).mean(axis=1)
    header_vertical_row_score = (header_vertical > 0).mean(axis=1)

    row_threshold = max(0.012, float(row_score.max()) * 0.10)
    col_threshold = max(0.012, float(col_score.max()) * 0.08)
    vertical_row_threshold = max(0.006, float(vertical_row_score.max()) * 0.08)
    row_clusters = cluster_indices(np.flatnonzero(row_score >= row_threshold), max_gap=max(8, h // 180))
    col_clusters = cluster_indices(np.flatnonzero(col_score >= col_threshold), max_gap=max(8, w // 180))
    vertical_row_clusters = cluster_indices(
        np.flatnonzero(vertical_row_score >= vertical_row_threshold),
        max_gap=max(18, h // 95),
    )
    header_vertical_row_threshold = max(0.004, float(header_vertical_row_score.max()) * 0.08)
    header_vertical_row_clusters = cluster_indices(
        np.flatnonzero(header_vertical_row_score >= header_vertical_row_threshold),
        max_gap=max(12, h // 120),
    )

    y_span = span_from_clusters(row_clusters, min_value=int(h * 0.05), max_value=int(h * 0.90))
    if y_span is None or (y_span[1] - y_span[0]) < h * 0.35:
        y_span = span_from_clusters(
            vertical_row_clusters,
            min_value=int(h * 0.05),
            max_value=int(h * 0.90),
        )
    if y_span is None or (y_span[1] - y_span[0]) < h * 0.45:
        starts = [start for start, _ in row_clusters + vertical_row_clusters if int(h * 0.04) <= start <= int(h * 0.35)]
        fallback_y0 = min(starts) if starts else int(h * 0.08)
        y_span = (fallback_y0, int(h * 0.86))
    top_candidates = [
        start
        for start, _ in row_clusters + vertical_row_clusters
        if int(h * 0.04) <= start <= int(h * 0.28)
    ]
    if top_candidates:
        y_span = (min(y_span[0], min(top_candidates)), y_span[1])

    x_span = None
    if y_span is not None:
        y_probe0, y_probe1 = y_span
        horizontal_col_score = (horizontal[y_probe0:y_probe1, :] > 0).mean(axis=0)
        horizontal_threshold = max(0.002, float(horizontal_col_score.max()) * 0.04)
        horizontal_clusters = cluster_indices(
            np.flatnonzero(horizontal_col_score >= horizontal_threshold),
            max_gap=max(14, w // 140),
        )
        x_span = span_from_clusters(horizontal_clusters, min_value=int(w * 0.01), max_value=int(w * 0.99))
    if x_span is None:
        x_span = span_from_clusters(col_clusters, min_value=int(w * 0.02), max_value=int(w * 0.98))
    ink_span = None
    if y_span is not None:
        y_probe0, y_probe1 = y_span
        ink_col_score = (work[y_probe0:y_probe1, :] > 0).mean(axis=0)
        smooth = max(9, w // 180)
        kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
        ink_col_score = np.convolve(ink_col_score, kernel, mode="same")
        ink_span = projection_span(
            ink_col_score,
            min_value=int(w * 0.005),
            max_value=int(w * 0.995),
            percentile=62,
            min_threshold=0.004,
            max_gap=max(36, w // 36),
        )
    if y_span is not None and (x_span is None or (x_span[1] - x_span[0]) < w * 0.55):
        if ink_span is not None:
            x_span = ink_span
    elif x_span is not None and ink_span is not None:
        left_extra = max(0, x_span[0] - ink_span[0])
        right_extra = max(0, ink_span[1] - x_span[1])
        if left_extra <= w * 0.10 and right_extra <= w * 0.10:
            x_span = (min(x_span[0], ink_span[0]), max(x_span[1], ink_span[1]))

    zones: list[LayoutZone] = []
    if y_span is None or x_span is None:
        return zones

    x0, x1 = x_span
    y0, y1 = y_span

    if content_bbox is None:
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    else:
        cx0, cy0, cx1, cy1 = content_bbox

    # The title is above the ruled table and the footnote block is below it.
    # Pad slightly so OCR will not lose border-adjacent digits.
    pad_x = max(18, int(w * 0.010))
    pad_y = max(18, int(h * 0.010))
    x0 = max(cx0, x0 - pad_x)
    x1 = min(cx1, x1 + pad_x)
    y0 = max(cy0, y0 - pad_y)
    y1 = min(cy1, y1 + pad_y)

    # Tighten semantic table boundaries. The table top should include the
    # column-label row, so use short vertical header-grid lines rather than
    # horizontal separators, which often sit below the labels.
    header_spans = [
        (start, end)
        for start, end in header_vertical_row_clusters
        if (end - start) > h * 0.35 and start < h * 0.25
    ]
    if header_spans:
        header_top = min(start for start, _ in header_spans)
        ruled_top = max(cy0, header_top - pad_y)
        top_border_candidates = [
            (start, end)
            for start, end in row_clusters
            if int(h * 0.045) <= start <= int(h * 0.160)
            and start <= header_top
            and (header_top - start) <= int(h * 0.120)
        ]
        if top_border_candidates:
            border_start = min(start for start, _ in top_border_candidates)
            ruled_top = max(cy0, border_start - max(pad_y // 2, 6))
        if ruled_top < h * 0.055:
            ruled_top = int(h * 0.060)
        if ruled_top <= int(h * 0.25):
            y0 = ruled_top

    bottom_rule = strongest_horizontal_cluster(
        row_clusters,
        row_score,
        min_value=int(h * 0.700),
        max_value=int(h * 0.900),
    )
    if bottom_rule is not None:
        y1 = min(y1, bottom_rule[1])

    table_ink_span = None
    if y1 > y0:
        table_ink_col_score = (work[y0:y1, :] > 0).mean(axis=0)
        smooth = max(9, w // 180)
        kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
        table_ink_col_score = np.convolve(table_ink_col_score, kernel, mode="same")
        table_ink_span = projection_span(
            table_ink_col_score,
            min_value=int(w * 0.005),
            max_value=int(w * 0.995),
            percentile=62,
            min_threshold=0.004,
            max_gap=max(36, w // 36),
        )
    if table_ink_span is not None:
        ink_x0, ink_x1 = table_ink_span
        # The OCR table crop should be conservative: losing a side column is
        # much worse than including a narrow page margin. Use the ink span
        # inside the final table band as the primary x extent.
        x0 = max(cx0, min(x0, ink_x0 - max(pad_x, int(w * 0.015))))
        x1 = min(cx1, max(x1, ink_x1 + max(pad_x, int(w * 0.015))))

    # Final guardrail for regular azioni pages: the financial table normally
    # spans almost the whole printed page. Rectangular crops cannot follow
    # curvature, so prefer a full-width OCR crop over clipped columns.
    x0 = max(cx0, min(x0, int(w * 0.030)))
    x1 = min(cx1, max(x1, int(w * 0.970)))
    y0 = max(cy0, min(y0, int(h * 0.075)))

    table_width = x1 - x0
    table_height = y1 - y0
    if table_width < w * 0.55 or table_height < h * 0.45:
        return zones
    if table_width > w * 0.995 or table_height > h * 0.970:
        return zones

    zones.append(LayoutZone("azioni_table", (x0, y0, x1, y1), 0.82))

    if y0 > h * 0.04:
        zones.append(LayoutZone("azioni_header", (x0, cy0, x1, y0), 0.55))
    if y1 < h * 0.94:
        footer_y0 = max(0, y1)
        zones.append(LayoutZone("azioni_footer", (x0, footer_y0, x1, cy1), 0.55))

    return zones


def repair_azioni_zone_geometry(zones: list[LayoutZone], shape: tuple[int, int]) -> list[LayoutZone]:
    """Correct plausible-looking azioni zones that still lost table columns."""
    table = next((zone for zone in zones if zone.zone_type == "azioni_table"), None)
    if table is None:
        return zones

    h, w = shape
    x0, y0, x1, y1 = table.bbox

    # Regular stock pages span almost the full printed page. On some tilted or
    # low-contrast pages, line projection locks onto the inner columns and cuts
    # off the capital/share columns on the left.
    repaired_x0 = x0
    repaired_x1 = x1
    repaired_y1 = y1
    if x0 > w * 0.10 or (x1 - x0) < w * 0.78:
        repaired_x0 = min(x0, int(w * 0.035))

    # A detected horizontal rule in the middle of the table can be mistaken for
    # the table bottom. Regular azioni listings normally run close to the
    # footnote block, so repair obvious premature endings.
    if y1 < h * 0.72:
        repaired_y1 = int(h * 0.855)

    if repaired_x0 == x0 and repaired_x1 == x1 and repaired_y1 == y1:
        return zones

    repaired: list[LayoutZone] = []
    for zone in zones:
        zx0, zy0, zx1, zy1 = zone.bbox
        if zone.zone_type == "azioni_table":
            repaired.append(LayoutZone(zone.zone_type, (repaired_x0, zy0, repaired_x1, repaired_y1), zone.confidence))
        elif zone.zone_type == "azioni_header":
            repaired.append(LayoutZone(zone.zone_type, (repaired_x0, zy0, repaired_x1, zy1), zone.confidence))
        elif zone.zone_type == "azioni_footer":
            footer_y0 = max(0, repaired_y1)
            repaired.append(LayoutZone(zone.zone_type, (repaired_x0, footer_y0, repaired_x1, zy1), zone.confidence))
        else:
            repaired.append(zone)
    return repaired


def fallback_azioni_zones(
    shape: tuple[int, int],
    content_bbox: tuple[int, int, int, int],
) -> list[LayoutZone]:
    """Fallback for confirmed stock pages with weak or broken ruling."""
    h, w = shape
    _ = content_bbox
    cx0, cy0, cx1, cy1 = int(w * 0.035), int(h * 0.025), int(w * 0.965), int(h * 0.965)
    y0 = int(h * 0.085)
    y1 = int(h * 0.855)
    footer_y0 = max(cy0, y1)

    zones = [LayoutZone("azioni_table", (cx0, y0, cx1, y1), 0.55)]
    if y0 > cy0:
        zones.append(LayoutZone("azioni_header", (cx0, cy0, cx1, y0), 0.45))
    if y1 < cy1:
        zones.append(LayoutZone("azioni_footer", (cx0, footer_y0, cx1, cy1), 0.45))
    return zones


def annotate(normalized_gray: np.ndarray, zones: list[LayoutZone], label: str) -> np.ndarray:
    preview = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    colors = {
        "azioni_table": (255, 120, 0),
        "azioni_header": (0, 180, 255),
        "azioni_footer": (220, 80, 220),
    }
    for zone in zones:
        x0, y0, x1, y1 = zone.bbox
        color = colors.get(zone.zone_type, (0, 255, 0))
        thickness = 6 if zone.zone_type == "azioni_table" else 3
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, thickness)
    cv2.putText(preview, label, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 255), 4)
    return resize_for_preview(preview, width=1200)


def load_azioni_targets(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    return [row for row in rows if row["page_type"] == "regular_azioni"]


def process(input_dir: Path, labels_path: Path, output_dir: Path, limit: int | None) -> None:
    targets = load_azioni_targets(labels_path)
    if limit is not None:
        targets = targets[:limit]

    page_dir = output_dir / "page_crops"
    normalized_dir = output_dir / "normalized_pages"
    table_dir = output_dir / "table_crops"
    header_dir = output_dir / "header_crops"
    footer_dir = output_dir / "footer_crops"
    preview_dir = output_dir / "zone_previews"
    manifest_path = output_dir / "azioni_zone_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    for generated_dir in [page_dir, normalized_dir, table_dir, header_dir, footer_dir, preview_dir]:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    if manifest_path.exists():
        manifest_path.unlink()

    cache: dict[str, dict[str, object]] = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "source_file",
                "date",
                "picture",
                "side",
                "page_type",
                "page_crop_file",
                "normalized_file",
                "zone_preview_file",
                "zone_type",
                "zone_crop_file",
                "x0",
                "y0",
                "x1",
                "y1",
                "confidence",
                "analysis_scale_to_page",
            ],
        )
        writer.writeheader()

        for target in targets:
            source_file = target["source_file"]
            side = target["side"]
            source_path = input_dir / source_file
            if source_file not in cache:
                image = cv2.imread(str(source_path))
                if image is None:
                    raise OSError(f"Could not read image: {source_path}")
                cache[source_file] = {
                    page.side: page for page in crop_half_pages(image, gutter_overlap=40, deskew=True)
                }

            page = cache[source_file][side]
            stem = f"{Path(source_file).stem}_{side}"
            page_path = page_dir / f"{stem}.jpg"
            normalized_path = normalized_dir / f"{stem}.jpg"
            preview_path = preview_dir / f"{stem}_azioni_zones.jpg"

            write_image(page_path, page.crop)
            normalized, scale_to_page = normalize_page_for_analysis(page.crop)
            write_image(normalized_path, normalized)

            content_x0, content_y0, content_x1, content_y1, _ = estimate_content_bbox(page.crop)
            content_bbox = scale_bbox_to_analysis(
                (content_x0, content_y0, content_x1, content_y1),
                scale_to_page,
                normalized.shape,
            )
            preview_zones = detect_azioni_table_zone(normalized, content_bbox)
            if not any(zone.zone_type == "azioni_table" for zone in preview_zones):
                preview_zones = fallback_azioni_zones(normalized.shape, content_bbox)
            preview_zones = repair_azioni_zone_geometry(preview_zones, normalized.shape)
            page_zones = [scale_zone(zone, scale_to_page, page.crop.shape[:2]) for zone in preview_zones]
            write_image(preview_path, annotate(normalized, preview_zones, stem))

            meta = parse_scan_name(Path(source_file))
            for zone in page_zones:
                x0, y0, x1, y1 = zone.bbox
                zone_crop_path = ""
                if zone.zone_type == "azioni_table":
                    crop = page.crop[y0:y1, x0:x1]
                    crop_path = table_dir / f"{stem}_table.jpg"
                    write_image(crop_path, crop)
                    zone_crop_path = str(crop_path.relative_to(output_dir))
                elif zone.zone_type == "azioni_header":
                    crop = page.crop[y0:y1, x0:x1]
                    crop_path = header_dir / f"{stem}_header.jpg"
                    write_image(crop_path, crop)
                    zone_crop_path = str(crop_path.relative_to(output_dir))
                elif zone.zone_type == "azioni_footer":
                    crop = page.crop[y0:y1, x0:x1]
                    crop_path = footer_dir / f"{stem}_footer.jpg"
                    write_image(crop_path, crop)
                    zone_crop_path = str(crop_path.relative_to(output_dir))
                writer.writerow(
                    {
                        "source_file": source_file,
                        "date": meta["date"],
                        "picture": meta["picture"],
                        "side": side,
                        "page_type": target["page_type"],
                        "page_crop_file": str(page_path.relative_to(output_dir)),
                        "normalized_file": str(normalized_path.relative_to(output_dir)),
                        "zone_preview_file": str(preview_path.relative_to(output_dir)),
                        "zone_type": zone.zone_type,
                        "zone_crop_file": zone_crop_path,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "confidence": f"{zone.confidence:.4f}",
                        "analysis_scale_to_page": f"{scale_to_page:.6f}",
                    }
                )

    print(f"Processed {len(targets)} regular_azioni page sides")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote previews: {preview_dir}")
    print(f"Wrote table crops: {table_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract regular_azioni table zones.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    process(args.input_dir, args.labels, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
