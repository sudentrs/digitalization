"""Preprocess Bocconi exchange-listing JPEG scans.

This first-pass script is deliberately conservative: it does not overwrite the
source scans and it keeps diagnostics so page-splitting failures are easy to
spot before OCR/table extraction is added.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FILENAME_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{2})_(?P<pic>\d+)\.JPG$", re.I)


@dataclass
class PageCrop:
    side: str
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    gutter_x: int
    warp_applied: bool
    content_ratio: float
    mean_gray: float
    dark_ratio: float
    inner_dark_ratio: float
    horizontal_line_score: float
    vertical_line_score: float
    text_component_count: int
    deskew_angle: float


@dataclass
class LayoutZone:
    zone_type: str
    bbox: tuple[int, int, int, int]
    confidence: float


def parse_scan_name(path: Path) -> dict[str, str | int]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected scan filename: {path.name}")
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    pic = int(match.group("pic"))
    return {
        "year": year,
        "month": month,
        "day": day,
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "picture": pic,
    }


def resize_for_preview(image: np.ndarray, width: int = 1600) -> np.ndarray:
    height, current_width = image.shape[:2]
    scale = width / current_width
    return cv2.resize(image, (width, int(height * scale)), interpolation=cv2.INTER_AREA)


def resize_for_analysis(image: np.ndarray, max_width: int = 1400) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def make_page_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # The table/background is dark and pages are light. Otsu isolates paper from
    # black surroundings, while morphology connects page regions across shadows.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def estimate_content_bbox(image: np.ndarray) -> tuple[int, int, int, int, float]:
    """Find the likely paper/content bounding box inside a half-spread image."""
    mask = make_page_mask(image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    if not contours:
        return 0, 0, w, h, 0.0

    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    area_ratio = float(cv2.contourArea(contour)) / float(w * h)

    # Keep page crops deliberately generous. Losing a left/right table column is
    # worse for OCR than carrying some page margin or dark background forward.
    pad_x = max(80, int(w * 0.04))
    pad_y = max(40, int(h * 0.015))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    return x0, y0, x1, y1, area_ratio


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def warp_page_if_possible(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Perspective-correct a page crop when a stable page outline is found."""
    mask = make_page_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    contour = max(contours, key=cv2.contourArea)
    h, w = mask.shape
    area_ratio = cv2.contourArea(contour) / float(w * h)
    if area_ratio < 0.45:
        return image, False

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approx) != 4:
        return image, False

    rect = order_points(approx.reshape(4, 2).astype(np.float32))
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))
    if max_width < 500 or max_height < 800:
        return image, False

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)
    return warped, True


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(245, 245, 245))


def binarize_ink(gray: np.ndarray) -> np.ndarray:
    normalized = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )


def line_structure_metrics(gray: np.ndarray) -> tuple[float, float, int]:
    binary = binarize_ink(gray)
    h, w = binary.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 18), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 18)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_score = float(np.mean(horizontal > 0))
    vertical_score = float(np.mean(vertical > 0))

    line_mask = cv2.bitwise_or(horizontal, vertical)
    text_only = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(text_only, connectivity=8)
    count = 0
    page_area = h * w
    for i in range(1, components):
        x, y, bw, bh, area = stats[i]
        if area < 5 or area > page_area * 0.01:
            continue
        if bw > w * 0.35 or bh > h * 0.08:
            continue
        count += 1

    return horizontal_score, vertical_score, count


def normalize_page_for_analysis(image: np.ndarray, max_width: int = 1800) -> tuple[np.ndarray, float]:
    """Correct slow background shading while keeping a grayscale OCR image."""
    h, w = image.shape[:2]
    scale_to_page = 1.0
    if w > max_width:
        scale = max_width / float(w)
        image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        scale_to_page = 1.0 / scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    kernel_size = max(51, (min(h, w) // 18) | 1)
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    normalized = cv2.divide(gray, background, scale=245)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized), scale_to_page


def scale_zone(zone: LayoutZone, scale: float, page_shape: tuple[int, int]) -> LayoutZone:
    ph, pw = page_shape
    x0, y0, x1, y1 = zone.bbox
    scaled = (
        max(0, min(pw, int(round(x0 * scale)))),
        max(0, min(ph, int(round(y0 * scale)))),
        max(0, min(pw, int(round(x1 * scale)))),
        max(0, min(ph, int(round(y1 * scale)))),
    )
    return LayoutZone(zone.zone_type, scaled, zone.confidence)


def bbox_from_mask(mask: np.ndarray, min_area_ratio: float = 0.0005) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < h * w * min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return None

    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return x0, y0, x1, y1


def line_component_boxes(mask: np.ndarray, min_area_ratio: float = 0.00003) -> list[tuple[int, int, int, int, int]]:
    components, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    h, w = mask.shape
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, components):
        x, y, bw, bh, area = stats[i]
        if area < h * w * min_area_ratio:
            continue
        boxes.append((x, y, x + bw, y + bh, int(area)))
    return boxes


def select_table_bbox(line_mask: np.ndarray, content_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Select the main ruled table block, excluding page edges/background."""
    h, w = line_mask.shape
    cx0, cy0, cx1, cy1 = content_bbox
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for x0, y0, x1, y1, area in line_component_boxes(line_mask):
        bw = x1 - x0
        bh = y1 - y0
        if bw < w * 0.35 or bh < h * 0.35:
            continue
        if bw > w * 0.92 and bh > h * 0.92:
            continue

        ix0 = max(x0, cx0)
        iy0 = max(y0, cy0)
        ix1 = min(x1, cx1)
        iy1 = min(y1, cy1)
        overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        box_area = max(1, bw * bh)
        overlap_ratio = overlap / float(box_area)
        if overlap_ratio < 0.55:
            continue

        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        content_center_x = (cx0 + cx1) / 2.0
        content_center_y = (cy0 + cy1) / 2.0
        center_penalty = (
            abs(center_x - content_center_x) / max(1.0, w)
            + abs(center_y - content_center_y) / max(1.0, h)
        )
        score = overlap_ratio + min(0.5, area / float(h * w)) - center_penalty
        candidates.append((score, (x0, y0, x1, y1)))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def span_from_projection(score: np.ndarray, min_threshold: float, peak_fraction: float) -> tuple[int, int] | None:
    if score.size == 0:
        return None
    peak = float(score.max())
    if peak < min_threshold:
        return None
    threshold = max(min_threshold, peak * peak_fraction)
    active = np.flatnonzero(score >= threshold)
    if active.size == 0:
        return None
    return int(active[0]), int(active[-1] + 1)


def projection_table_bbox(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    content_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Estimate table bounds from horizontal and vertical rule projections."""
    h, w = horizontal.shape
    cx0, cy0, cx1, cy1 = content_bbox
    x0_clip = max(0, int(cx0 + 0.01 * w))
    x1_clip = min(w, int(cx1 - 0.01 * w))
    y0_clip = max(0, int(cy0 + 0.01 * h))
    y1_clip = min(h, int(cy1 - 0.01 * h))
    if x1_clip <= x0_clip or y1_clip <= y0_clip:
        return None

    horizontal_roi = horizontal[y0_clip:y1_clip, x0_clip:x1_clip]
    row_score = (horizontal_roi > 0).mean(axis=1)

    smooth_rows = max(7, h // 260)
    row_kernel = np.ones(smooth_rows, dtype=np.float32) / float(smooth_rows)
    row_score = np.convolve(row_score, row_kernel, mode="same")

    edge_rows = max(3, int(row_score.size * 0.05))
    row_score[:edge_rows] = 0
    row_score[-edge_rows:] = 0
    y_span = span_from_projection(row_score, min_threshold=0.025, peak_fraction=0.10)
    if y_span is None:
        return None

    y0 = y0_clip + y_span[0]
    y1 = y0_clip + y_span[1]
    table_horizontal = horizontal[y0:y1, x0_clip:x1_clip]
    horizontal_col_score = (table_horizontal > 0).mean(axis=0)
    smooth_cols = max(9, w // 220)
    col_kernel = np.ones(smooth_cols, dtype=np.float32) / float(smooth_cols)
    horizontal_col_score = np.convolve(horizontal_col_score, col_kernel, mode="same")
    edge_cols = max(3, int(horizontal_col_score.size * 0.03))
    horizontal_col_score[:edge_cols] = 0
    horizontal_col_score[-edge_cols:] = 0
    horizontal_x_span = span_from_projection(horizontal_col_score, min_threshold=0.003, peak_fraction=0.08)

    vertical_roi = vertical[y0:y1, x0_clip:x1_clip]
    vertical_col_score = (vertical_roi > 0).mean(axis=0)
    vertical_col_score = np.convolve(vertical_col_score, col_kernel, mode="same")
    vertical_col_score[:edge_cols] = 0
    vertical_col_score[-edge_cols:] = 0
    vertical_x_span = span_from_projection(vertical_col_score, min_threshold=0.008, peak_fraction=0.03)

    if horizontal_x_span is None and vertical_x_span is None:
        return None

    if horizontal_x_span is not None and vertical_x_span is not None:
        x0 = x0_clip + min(horizontal_x_span[0], vertical_x_span[0])
        vertical_width = vertical_x_span[1] - vertical_x_span[0]
        if vertical_width >= w * 0.35:
            x1 = x0_clip + vertical_x_span[1]
        else:
            x1 = x0_clip + max(horizontal_x_span[1], vertical_x_span[1])
    else:
        x_span = horizontal_x_span or vertical_x_span
        x0 = x0_clip + x_span[0]
        x1 = x0_clip + x_span[1]
    bw = x1 - x0
    bh = y1 - y0
    if bw < w * 0.35 or bh < h * 0.30:
        return None
    if bw > w * 0.94 or bh > h * 0.94:
        return None
    return x0, y0, x1, y1


def split_table_bbox_by_gaps(binary: np.ndarray, table_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Split a broad table area into large side-by-side blocks when obvious gaps exist."""
    x0, y0, x1, y1 = table_bbox
    roi = binary[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    h, w = roi.shape
    if w < 600:
        return []

    ink_by_col = (roi > 0).mean(axis=0)
    smooth = max(15, w // 90)
    kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
    ink_by_col = np.convolve(ink_by_col, kernel, mode="same")

    gap_threshold = max(0.002, float(np.percentile(ink_by_col, 18)))
    min_gap_width = max(35, int(w * 0.035))
    min_part_width = max(220, int(w * 0.22))
    gap_mask = ink_by_col <= gap_threshold

    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_gap in enumerate(gap_mask):
        if is_gap and start is None:
            start = i
        elif not is_gap and start is not None:
            if i - start >= min_gap_width:
                gaps.append((start, i))
            start = None
    if start is not None and len(gap_mask) - start >= min_gap_width:
        gaps.append((start, len(gap_mask)))

    split_points: list[int] = []
    for gap_start, gap_end in gaps:
        point = (gap_start + gap_end) // 2
        if point >= min_part_width and (w - point) >= min_part_width:
            split_points.append(point)

    if not split_points:
        return []

    bounds = [0] + sorted(set(split_points)) + [w]
    parts: list[tuple[int, int, int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left < min_part_width:
            continue
        parts.append((x0 + left, y0, x0 + right, y1))

    return parts if len(parts) > 1 else []


def pad_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = bbox
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def detect_layout_zones(normalized_gray: np.ndarray) -> list[LayoutZone]:
    """Find coarse header/table/footer zones for visual QA and later OCR."""
    binary = binarize_ink(normalized_gray)
    h, w = binary.shape
    margin_x = max(8, int(w * 0.025))
    margin_y = max(8, int(h * 0.020))
    binary_work = binary.copy()
    binary_work[:, :margin_x] = 0
    binary_work[:, w - margin_x :] = 0
    binary_work[:margin_y, :] = 0
    binary_work[h - margin_y :, :] = 0

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, w // 10), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, h // 12)))
    horizontal = cv2.morphologyEx(binary_work, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary_work, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal, vertical)

    content_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 80), max(8, h // 180)))
    content_mask = cv2.morphologyEx(binary_work, cv2.MORPH_CLOSE, content_kernel)
    content_bbox = bbox_from_mask(content_mask, min_area_ratio=0.0002)

    zones: list[LayoutZone] = []
    if content_bbox is None:
        return zones

    content_bbox = pad_bbox(content_bbox, 10, binary.shape)
    cx0, cy0, cx1, cy1 = content_bbox
    zones.append(LayoutZone("content", content_bbox, 0.50))
    content_parts = split_table_bbox_by_gaps(binary_work, content_bbox)
    for index, part_bbox in enumerate(content_parts, start=1):
        zones.append(LayoutZone(f"content_part_{index}", pad_bbox(part_bbox, 10, binary.shape), 0.45))
    table_candidate = projection_table_bbox(horizontal, vertical, content_bbox)
    if table_candidate is None:
        table_candidate = select_table_bbox(line_mask, content_bbox)

    if table_candidate is not None:
        table_pad = max(18, int(min(h, w) * 0.012))
        table_bbox = pad_bbox(table_candidate, table_pad, binary.shape)
        tx0, ty0, tx1, ty1 = table_bbox
        table_area = (tx1 - tx0) * (ty1 - ty0)
        confidence = min(0.95, 0.45 + table_area / float(h * w))
        zones.append(LayoutZone("table", table_bbox, confidence))
        table_parts = split_table_bbox_by_gaps(binary_work, table_bbox)
        for index, part_bbox in enumerate(table_parts, start=1):
            zones.append(LayoutZone(f"table_part_{index}", pad_bbox(part_bbox, table_pad, binary.shape), 0.70))

        if ty0 - cy0 > max(35, h * 0.025):
            zones.append(LayoutZone("header", (cx0, cy0, cx1, ty0), 0.60))
        if cy1 - ty1 > max(35, h * 0.025):
            zones.append(LayoutZone("footer", (cx0, ty1, cx1, cy1), 0.60))
    else:
        zones.append(LayoutZone("text_block", content_bbox, 0.35))

    return zones


def annotate_zone_preview(page: PageCrop, normalized_gray: np.ndarray, zones: list[LayoutZone]) -> np.ndarray:
    preview = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    colors = {
        "content": (160, 160, 160),
        "content_part": (80, 200, 120),
        "table": (255, 120, 0),
        "table_part": (40, 220, 255),
        "header": (0, 180, 255),
        "footer": (220, 80, 220),
        "text_block": (255, 120, 0),
        "blank_page": (0, 165, 255),
    }
    for zone in zones:
        x0, y0, x1, y1 = zone.bbox
        if zone.zone_type.startswith("table_part_"):
            color_key = "table_part"
        elif zone.zone_type.startswith("content_part_"):
            color_key = "content_part"
        else:
            color_key = zone.zone_type
        color = colors.get(color_key, (0, 255, 0))
        thickness = 6 if zone.zone_type in {"table", "text_block"} else 3
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, thickness)
        label = f"{zone.zone_type} {zone.confidence:.2f}"
        cv2.putText(preview, label, (x0 + 12, max(40, y0 + 34)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.putText(preview, page.side, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (40, 40, 255), 4)
    return resize_for_preview(preview, width=1200)


def estimate_deskew_angle(gray: np.ndarray) -> float:
    binary = binarize_ink(gray)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    h, w = gray.shape
    min_line_length = max(80, w // 8)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=min_line_length, maxLineGap=20)
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if -8 <= angle <= 8:
            angles.append(float(angle))

    if len(angles) < 4:
        return 0.0
    return float(np.median(angles))


def detect_gutter_x(image: np.ndarray) -> int:
    """Estimate the book gutter so left/right crops do not overlap.

    The gutter is usually a darker vertical band near the image center, but
    printed table columns can be dark too. We therefore combine darkness, a
    "paper occupancy" gap score, and an ink-density penalty so dense table
    columns are less likely to be mistaken for the fold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_start = int(w * 0.45)
    center_end = int(w * 0.55)
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    center_strip = gray[y0:y1, center_start:center_end]
    paper_strip = make_page_mask(image)[y0:y1, center_start:center_end]

    darkness = 255.0 - center_strip.mean(axis=0)
    paper_occupancy = (paper_strip > 0).mean(axis=0)
    gap_score = 1.0 - paper_occupancy
    ink = binarize_ink(center_strip)
    ink_density = (ink > 0).mean(axis=0)
    window = max(25, int(w * 0.006))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    darkness = np.convolve(darkness, kernel, mode="same")
    gap_score = np.convolve(gap_score, kernel, mode="same")
    ink_density = np.convolve(ink_density, kernel, mode="same")

    darkness_range = np.ptp(darkness)
    gap_range = np.ptp(gap_score)
    ink_range = np.ptp(ink_density)
    darkness_norm = (darkness - darkness.min()) / darkness_range if darkness_range else np.zeros_like(darkness)
    gap_norm = (gap_score - gap_score.min()) / gap_range if gap_range else np.zeros_like(gap_score)
    ink_norm = (ink_density - ink_density.min()) / ink_range if ink_range else np.zeros_like(ink_density)
    x_positions = np.arange(center_start, center_end)
    center_penalty = np.abs(x_positions - (w / 2.0)) / float(center_end - center_start)
    combined = 0.45 * darkness_norm + 0.20 * gap_norm + 0.35 * (1.0 - ink_norm) - 0.40 * center_penalty
    return center_start + int(np.argmax(combined))


def crop_half_pages(image: np.ndarray, gutter_overlap: int, deskew: bool) -> list[PageCrop]:
    h, w = image.shape[:2]
    gutter_x = detect_gutter_x(image)
    left_end = min(w, gutter_x + gutter_overlap)
    right_start = max(0, gutter_x - gutter_overlap)
    halves = {
        "left": (0, left_end),
        "right": (right_start, w),
    }
    crops: list[PageCrop] = []

    for side, (x_start, x_end) in halves.items():
        half = image[:, x_start:x_end]
        x0, y0, x1, y1, content_ratio = estimate_content_bbox(half)
        rough_crop = half[y0:y1, x0:x1]
        crop, warp_applied = warp_page_if_possible(rough_crop)
        deskew_angle = 0.0
        if deskew:
            analysis_crop = resize_for_analysis(crop)
            gray_for_angle = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)
            deskew_angle = estimate_deskew_angle(gray_for_angle)
            if 0.2 <= abs(deskew_angle) <= 5.0:
                crop = rotate_image(crop, deskew_angle)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        analysis_gray = cv2.cvtColor(resize_for_analysis(crop), cv2.COLOR_BGR2GRAY)
        ch, cw = analysis_gray.shape
        margin_y = int(ch * 0.10)
        margin_x = int(cw * 0.10)
        inner = analysis_gray[margin_y : ch - margin_y, margin_x : cw - margin_x]
        inner_dark_ratio = float(np.mean(inner < 90)) if inner.size else float(np.mean(analysis_gray < 90))
        if inner_dark_ratio < 0.10:
            horizontal_line_score, vertical_line_score, text_component_count = line_structure_metrics(
                inner if inner.size else analysis_gray
            )
        else:
            horizontal_line_score = 0.0
            vertical_line_score = 0.0
            text_component_count = 0
        crops.append(
            PageCrop(
                side=side,
                crop=crop,
                bbox=(x_start + x0, y0, x_start + x1, y1),
                gutter_x=gutter_x,
                warp_applied=warp_applied,
                content_ratio=content_ratio,
                mean_gray=float(np.mean(gray)),
                dark_ratio=float(np.mean(gray < 90)),
                inner_dark_ratio=inner_dark_ratio,
                horizontal_line_score=horizontal_line_score,
                vertical_line_score=vertical_line_score,
                text_component_count=text_component_count,
                deskew_angle=deskew_angle,
            )
        )

    return crops


def annotate_preview(image: np.ndarray, pages: list[PageCrop], label: str) -> np.ndarray:
    preview = image.copy()
    if pages:
        gutter_x = pages[0].gutter_x
        cv2.line(preview, (gutter_x, 0), (gutter_x, preview.shape[0]), (0, 0, 255), 6)
    for page in pages:
        x0, y0, x1, y1 = page.bbox
        color = (60, 220, 60) if page.content_ratio > 0.35 else (0, 165, 255)
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, 8)
        warp = " warped" if page.warp_applied else ""
        text = (
            f"{page.side}{warp} inner_dark={page.inner_dark_ratio:.2f} "
            f"h={page.horizontal_line_score:.3f} v={page.vertical_line_score:.3f} "
            f"angle={page.deskew_angle:.2f}"
        )
        cv2.putText(preview, text, (x0 + 20, max(50, y0 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(preview, label, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 200, 255), 5)
    return resize_for_preview(preview)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise OSError(f"Could not write image: {path}")


def process(
    input_dir: Path,
    output_dir: Path,
    limit: int | None,
    start_index: int,
    end_index: int | None,
    gutter_overlap: int,
    write_crops: bool,
    deskew: bool,
) -> None:
    scans = sorted(input_dir.glob("*.JPG"))
    if start_index < 0:
        raise ValueError("--start-index must be zero or greater")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    scans = scans[start_index:end_index]
    if limit is not None:
        scans = scans[:limit]

    crops_dir = output_dir / "page_crops"
    normalized_dir = output_dir / "normalized_pages"
    previews_dir = output_dir / "previews"
    zone_previews_dir = output_dir / "zone_previews"
    manifest_path = output_dir / "preprocess_manifest.csv"
    zone_manifest_path = output_dir / "zone_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as fp, zone_manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as zone_fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "source_file",
                "date",
                "picture",
                "side",
                "crop_file",
                "x0",
                "y0",
                "x1",
                "y1",
                "gutter_x",
                "warp_applied",
                "content_ratio",
                "mean_gray",
                "dark_ratio",
                "inner_dark_ratio",
                "horizontal_line_score",
                "vertical_line_score",
                "text_component_count",
                "deskew_angle",
                "flag",
            ],
        )
        writer.writeheader()
        zone_writer = csv.DictWriter(
            zone_fp,
            fieldnames=[
                "source_file",
                "date",
                "picture",
                "side",
                "zone_type",
                "zone_file",
                "normalized_file",
                "x0",
                "y0",
                "x1",
                "y1",
                "confidence",
                "analysis_scale_to_page",
            ],
        )
        zone_writer.writeheader()

        for scan in scans:
            meta = parse_scan_name(scan)
            image = cv2.imread(str(scan))
            if image is None:
                raise OSError(f"Could not read image: {scan}")

            pages = crop_half_pages(image, gutter_overlap, deskew)
            preview = annotate_preview(image, pages, f"{scan.name}")
            write_image(previews_dir / f"{scan.stem}_preview.jpg", preview)

            for page in pages:
                crop_name = f"{scan.stem}_{page.side}.jpg"
                crop_path = crops_dir / crop_name
                normalized_path = normalized_dir / crop_name
                zone_preview_path = zone_previews_dir / f"{scan.stem}_{page.side}_zones.jpg"
                if write_crops:
                    write_image(crop_path, page.crop)
                flag = ""
                line_score = page.horizontal_line_score + page.vertical_line_score
                if page.content_ratio < 0.30:
                    flag = "low_content_or_blank"
                elif page.inner_dark_ratio < 0.005 and line_score < 0.002:
                    flag = "possible_blank_page"
                elif page.inner_dark_ratio < 0.08:
                    flag = "low_text_density_review"
                elif page.inner_dark_ratio > 0.38:
                    flag = "possible_occlusion_or_shadow"
                normalized, analysis_scale_to_page = normalize_page_for_analysis(page.crop)
                if flag in {"low_content_or_blank", "possible_blank_page"}:
                    nh, nw = normalized.shape
                    zones_for_preview = [LayoutZone("blank_page", (0, 0, nw, nh), 0.80)]
                else:
                    zones_for_preview = detect_layout_zones(normalized)
                zones = [
                    scale_zone(zone, analysis_scale_to_page, page.crop.shape[:2])
                    for zone in zones_for_preview
                ]
                write_image(normalized_path, normalized)
                zone_preview = annotate_zone_preview(page, normalized, zones_for_preview)
                write_image(zone_preview_path, zone_preview)

                x0, y0, x1, y1 = page.bbox
                writer.writerow(
                    {
                        "source_file": scan.name,
                        "date": meta["date"],
                        "picture": meta["picture"],
                        "side": page.side,
                        "crop_file": str(crop_path.relative_to(output_dir)),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "gutter_x": page.gutter_x,
                        "warp_applied": page.warp_applied,
                        "content_ratio": f"{page.content_ratio:.4f}",
                        "mean_gray": f"{page.mean_gray:.2f}",
                        "dark_ratio": f"{page.dark_ratio:.4f}",
                        "inner_dark_ratio": f"{page.inner_dark_ratio:.4f}",
                        "horizontal_line_score": f"{page.horizontal_line_score:.6f}",
                        "vertical_line_score": f"{page.vertical_line_score:.6f}",
                        "text_component_count": page.text_component_count,
                        "deskew_angle": f"{page.deskew_angle:.4f}",
                        "flag": flag,
                    }
                )
                for zone in zones:
                    zx0, zy0, zx1, zy1 = zone.bbox
                    zone_writer.writerow(
                        {
                            "source_file": scan.name,
                            "date": meta["date"],
                            "picture": meta["picture"],
                            "side": page.side,
                            "zone_type": zone.zone_type,
                            "zone_file": str(zone_preview_path.relative_to(output_dir)),
                            "normalized_file": str(normalized_path.relative_to(output_dir)),
                            "x0": zx0,
                            "y0": zy0,
                            "x1": zx1,
                            "y1": zy1,
                            "confidence": f"{zone.confidence:.4f}",
                            "analysis_scale_to_page": f"{analysis_scale_to_page:.6f}",
                        }
                    )

    print(f"Processed {len(scans)} scans")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote zone manifest: {zone_manifest_path}")
    print(f"Wrote previews: {previews_dir}")
    print(f"Wrote normalized pages: {normalized_dir}")
    print(f"Wrote zone previews: {zone_previews_dir}")
    if write_crops:
        print(f"Wrote page crops: {crops_dir}")
    else:
        print("Skipped full-resolution page crops; rerun with --write-crops to create them.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Bocconi JPEG scans with OpenCV.")
    parser.add_argument("--input-dir", default="1950", type=Path)
    parser.add_argument("--output-dir", default=Path("output") / "preprocess_1950", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N scans.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this sorted scan index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this sorted scan index.")
    parser.add_argument(
        "--gutter-overlap",
        type=int,
        default=40,
        help=(
            "Pixels included on both sides of the detected gutter. "
            "This prevents losing columns near the fold; overlap can be resolved later."
        ),
    )
    parser.add_argument(
        "--gutter-gap",
        dest="gutter_overlap",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--write-crops",
        action="store_true",
        help="Write full-resolution cropped page images. Omit while tuning to save disk space.",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Estimate and apply deskew rotation. Slower; best used for OCR crop generation.",
    )
    args = parser.parse_args()

    process(
        args.input_dir,
        args.output_dir,
        args.limit,
        args.start_index,
        args.end_index,
        args.gutter_overlap,
        args.write_crops,
        args.deskew,
    )


if __name__ == "__main__":
    main()
