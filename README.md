# Bocconi financial table digitization

Code for extracting financial tables from the 1950 Bocconi scans. The pipeline splits facing pages, locates tables, runs Tesseract OCR, groups words into rows and columns, and flags records for review.

This repository contains the source code and setup instructions. Scans, generated datasets, and the final presentation are shared separately. See the [data guide](docs/data.md) for the required files and their directory structure.

## Code layout

| Location | Contents |
| --- | --- |
| `scripts/` | Preprocessing, OCR, parsing, cleaning, and review scripts. See the [script guide](scripts/README.md). |
| `requirements.txt` | Python dependencies for the OCR pipeline. |
| `docs/data.md` | Instructions for using the separately supplied data. |

The main azioni workflow is `preprocess_scans.py` → `classify_page_types.py` → `extract_azioni_zones.py` → `batch_test_azioni_sample.py`. Other page types use `batch_process_page_types.py` and `parse_page_types_structured.py`; bond records have a separate cleaner in `clean_obbligazioni_records.py`.

## Setup

Use Python 3.10 or later. Run these commands in PowerShell from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install Tesseract separately. Put its executable on `PATH`, or pass `--tesseract` to the OCR scripts. Put the Italian and English language files (`ita.traineddata` and `eng.traineddata`) in `tools/tessdata/`. Alternatively, pass `--tessdata-dir` with the path to an existing language-data directory.

Place the source JPEG scans in `1950/` before running preprocessing. These files and the OCR language models are not downloaded by the scripts. A fresh clone contains no research data.

## Running the azioni pipeline

With the scans in place, generate the page manifest, page labels, and table crops:

```powershell
python scripts/preprocess_scans.py
python scripts/classify_page_types.py
python scripts/extract_azioni_zones.py
```

Then run OCR, coordinate-based parsing, and cleaning on two table crops:

```powershell
python scripts/batch_test_azioni_sample.py --sample-size 2 --output-dir output/local_azioni_check
```

If the separate data package already supplies `output/preprocess_1950_azioni/table_crops/`, you can start with the batch command. Increase `--sample-size` for a larger run. Use `--help` on a script to inspect its options.

The preprocessing commands write to their default folders under `output/`. Use explicit output paths when comparing new runs with saved results, and pass the corresponding manifest and label paths to later stages. Avoid `--overwrite` on saved benchmarks.

## Review and limitations

The separately supplied azioni benchmark covers 54 page sides and 3,278 candidate rows. Manual review addressed the remaining flagged identity fields; this is not a measurement of accuracy for every field. The bond parser remains a prototype with unresolved validation flags.

Review and correction scripts preserve the automatic output alongside correction records. Keep source scans and page-level OCR evidence with any dataset used for analysis. The [data guide](docs/data.md) identifies the saved benchmark files.

The optional JavaScript workbook builders require `@oai/artifact-tool` from the original authoring environment. They are not required for the Python pipeline or CSV review. Report-generation helpers and presentation build sources are not included in this code repository.

## Sharing

Git excludes source scans, generated outputs, language models, reports, PDFs, Word files, local environments, and ZIP archives. These files remain available locally. Share the code repository link and supply the data or final presentation separately when requested.

`scripts/package_submission.py` is a helper for creating the full local research ZIP. It includes large data files and is separate from the code-only Git workflow. See the [data guide](docs/data.md#local-archive) before using it.
