# Data supplied separately

The Git repository contains the code. Source scans, generated results, documents, and OCR language models are excluded through `.gitignore` and remain in the local project folder.

| Local path | Purpose |
| --- | --- |
| `1950/` | Original JPEG scans, with filenames such as `1950-01-05_1.JPG`. |
| `output/` | Generated page crops, manifests, OCR words and coordinates, parsed records, review images, and manual corrections. |
| `tools/tessdata/` | `ita.traineddata` and `eng.traineddata` used by Tesseract. |
| `bocconi_final_presentation.pdf` | Final presentation, sent separately. |

To use the existing data package, extract its `1950/`, `output/`, and `tools/tessdata/` folders into the repository root. Keep their names and directory structure intact. Alternatively, place the original scans in `1950/` and regenerate the outputs using the commands in the main README. Supply Italian and English language files in `tools/tessdata/`, or pass `--tessdata-dir` to the OCR scripts to use an existing installation's language directory.

The full local data package contains about 5.3 GB of scans and generated outputs before compression. Share it through a separate file-transfer or university storage link when needed. No data download link is configured in this repository.

## Existing benchmark results

The separately supplied azioni benchmark is under `output/batch_100_azioni/`. Despite the historical folder name, it covers 54 page sides and 3,278 candidate rows. Read `final_benchmark_summary.md` in that folder for the review history.

- `combined_cleaned_candidate_rows.csv`: automatic output.
- `manual_corrections.csv`: final manual corrections.
- `corrected/combined_cleaned_candidate_rows_corrected.csv`: corrected benchmark.
- `samples/`: page-level OCR and layout evidence.

These files belong together. The corrected CSV alone does not preserve the full review trail. The review addressed flagged identity fields; it does not establish accuracy for every extracted value.

The bond results are under `output/batch_obbligazioni_full/`, `output/parsed_obbligazioni_full/`, and `output/cleaned_obbligazioni_full/`. They remain provisional and contain unresolved validation flags.

## Local archive

`python scripts/package_submission.py` creates `submission/digitalization.zip` from the local research materials. Its selection is separate from Git's ignore rules: it deliberately includes the scans and saved outputs. It requires `bocconi_final_presentation.pdf` and refuses to replace an existing ZIP. The existing archive is a snapshot; editing the repository does not update it.
