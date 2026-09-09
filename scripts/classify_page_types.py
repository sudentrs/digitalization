"""Create first-pass page-type labels for the 1950 Bocconi scan sample.

The labels are intentionally coarse. They are meant to choose the next layout
strategy, not to extract data directly. Pages marked with medium confidence
should be reviewed before they are used as training data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_MANIFEST = Path("output/preprocess_1950/preprocess_manifest.csv")
DEFAULT_OUTPUT = Path("output/preprocess_1950/page_type_labels.csv")


PAGE_TYPE_DESCRIPTIONS = {
    "cover_valori_stato": "cover/title page with VALORI DI STATO table",
    "dividendi_in_pagamento": "dividend payment list",
    "regular_obbligazioni": "regular bonds page headed OBBLIGAZIONI",
    "regular_azioni": "regular equities page headed TITOLI AZIONARI",
    "comunicati_notizie": "notices page headed COMUNICATI E NOTIZIE",
    "special_market_lists": "mixed special lists such as MERCATO RISTRETTO, ALTRE PIAZZE, PROSSIME ASSEMBLEE, or call-order lists",
    "advertisement_notice": "advertisement or long company notice, often next to COMUNICATI E NOTIZIE",
    "blank_or_divider": "blank folder/divider page",
    "unknown_review": "not confidently classified",
}


# Overrides from visual inspection of the previews/contact sheets.
# Key format: (date, picture, side).
OVERRIDES: dict[tuple[str, int, str], tuple[str, str, str]] = {
    ("1950-01-05", 1, "left"): (
        "special_market_lists",
        "high",
        "Headings include ALCUNE QUOTAZIONI DI ALTRE PIAZZE and MERCATO RISTRETTO.",
    ),
    ("1950-03-17", 1, "left"): (
        "special_market_lists",
        "high",
        "Special quotation/list page before VALORI DI STATO cover.",
    ),
    ("1950-03-17", 5, "left"): (
        "special_market_lists",
        "high",
        "Headings include ALCUNE QUOTAZIONI DI ALTRE PIAZZE and MERCATO RISTRETTO.",
    ),
    ("1950-03-17", 5, "right"): (
        "blank_or_divider",
        "high",
        "Blank divider/folder page with small typed label.",
    ),
    ("1950-03-31", 1, "left"): (
        "special_market_lists",
        "medium",
        "Looks like special lists rather than the usual dividend table.",
    ),
    ("1950-04-03", 1, "left"): (
        "special_market_lists",
        "medium",
        "Looks like special lists rather than the usual dividend table.",
    ),
    ("1950-04-14", 1, "left"): (
        "special_market_lists",
        "medium",
        "Special lists visible before cover page.",
    ),
    ("1950-04-21", 1, "left"): (
        "special_market_lists",
        "high",
        "Headed PROSSIME ASSEMBLEE before VALORI DI STATO cover.",
    ),
    ("1950-04-28", 1, "left"): (
        "special_market_lists",
        "high",
        "Headed PROSSIME ASSEMBLEE / call-order list.",
    ),
    ("1950-05-19", 3, "left"): (
        "comunicati_notizie",
        "high",
        "Headed COMUNICATI E NOTIZIE; contains notices and small market summary blocks, not a regular TITOLI AZIONARI page.",
    ),
    ("1950-05-19", 3, "right"): (
        "special_market_lists",
        "high",
        "Headed CRONACHE SOCIETARIE and IL LISTINO SECONDO L'ORDINE DI CHIAMATA; not a regular TITOLI AZIONARI page.",
    ),
    ("1950-06-16", 5, "left"): (
        "special_market_lists",
        "medium",
        "Special list page before VALORI DI STATO cover.",
    ),
    ("1950-06-30", 4, "left"): (
        "regular_azioni",
        "high",
        "Extra TITOLI AZIONARI spread.",
    ),
    ("1950-06-30", 4, "right"): (
        "regular_azioni",
        "high",
        "Extra TITOLI AZIONARI spread.",
    ),
    ("1950-06-30", 5, "right"): (
        "blank_or_divider",
        "high",
        "Blank page.",
    ),
}


def default_label(picture: int, side: str) -> tuple[str, str, str]:
    if picture in {1, 5}:
        if side == "left":
            return (
                "dividendi_in_pagamento",
                "high",
                "Recurring left-side dividend payment list.",
            )
        return (
            "cover_valori_stato",
            "high",
            "Recurring right-side LISTINO UFFICIALE / VALORI DI STATO cover page.",
        )

    if picture == 2:
        return (
            "regular_obbligazioni",
            "high",
            "Recurring OBBLIGAZIONI spread.",
        )

    if picture == 3:
        return (
            "regular_azioni",
            "high",
            "Recurring TITOLI AZIONARI spread.",
        )

    if picture == 4:
        if side == "left":
            return (
                "comunicati_notizie",
                "high",
                "Recurring COMUNICATI E NOTIZIE page.",
            )
        return (
            "special_market_lists",
            "medium",
            "Right side usually contains special lists, assemblies, notices, or call-order tables.",
        )

    return ("unknown_review", "low", "No recurring rule for this picture number.")


def classify(date: str, picture: int, side: str) -> tuple[str, str, str]:
    return OVERRIDES.get((date, picture, side), default_label(picture, side))


def write_labels(manifest_path: Path, output_path: Path) -> Counter[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    with manifest_path.open(newline="", encoding="utf-8") as f_in, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = [
            "source_file",
            "date",
            "picture",
            "side",
            "page_type",
            "confidence",
            "needs_review",
            "description",
            "notes",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            date = row["date"]
            picture = int(row["picture"])
            side = row["side"]
            page_type, confidence, notes = classify(date, picture, side)
            counts[page_type] += 1
            writer.writerow(
                {
                    "source_file": row["source_file"],
                    "date": date,
                    "picture": picture,
                    "side": side,
                    "page_type": page_type,
                    "confidence": confidence,
                    "needs_review": "yes" if confidence != "high" else "no",
                    "description": PAGE_TYPE_DESCRIPTIONS[page_type],
                    "notes": notes,
                }
            )

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    counts = write_labels(args.manifest, args.output)
    print(f"Wrote {args.output}")
    for page_type, count in sorted(counts.items()):
        print(f"{page_type}: {count}")


if __name__ == "__main__":
    main()
