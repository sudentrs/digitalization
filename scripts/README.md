# Script guide

Run scripts from the repository root. Python command-line scripts provide `--help`.

| Stage | Scripts |
| --- | --- |
| Page preparation | `preprocess_scans.py`, `classify_page_types.py`, `extract_azioni_zones.py` |
| Azioni OCR and parsing | `ocr_azioni_tables.py`, `parse_one_azioni_page_structured.py`, `clean_one_azioni_page.py` |
| Azioni batch runs | `batch_test_azioni_sample.py` |
| Other page types | `batch_process_page_types.py`, `parse_page_types_structured.py` |
| Bond validation | `clean_obbligazioni_records.py` |
| Review and corrections | `build_column_diagnostics.py`, `build_high_priority_review_packet.py`, `build_manual_corrections_template.py`, `apply_manual_corrections.py` |
| Single-page experiments | `ocr_one_page_experiment.py`, `debug_one_azioni_page.py`, `parse_azioni_ocr.py` |
| Submission archive | `package_submission.py` |


`parse_azioni_ocr.py` is an earlier text-based experiment. For the azioni benchmark, use the coordinate-based parser through `batch_test_azioni_sample.py`.

Several scripts import helpers from neighbouring files, so keep the Python scripts together. Batch names and default output paths reflect the experiments for which they were written; supply explicit paths for a new run.

`package_submission.py` packages the separate local data submission. It requires the final presentation PDF and local research files; it is not a code-only GitHub export.
