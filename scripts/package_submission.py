"""Package the research files without local environments or generated previews."""

from pathlib import Path
import os
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    "venv", "node_modules", "__pycache__", "submission",
}
# The submission uses the final PDF; earlier reports stay in the repository.
EXCLUDED_PATHS = {"reports"}
EXCLUDED_FILES = {
    "scripts/build_followup_ocr_report.py",
    "scripts/build_supervisor_methods_doc.py",
    "scripts/make_supervisor_visual_map.py",
    "requirements-reports.txt",
}


def submission_files():
    for directory, folders, files in os.walk(ROOT, followlinks=False):
        parent = Path(directory)
        folders[:] = sorted(
            name for name in folders
            if name not in EXCLUDED_DIRS
            and not name.startswith(".")
            and (parent / name).relative_to(ROOT).as_posix() not in EXCLUDED_PATHS
            and not (parent / name).is_symlink()
        )
        for name in sorted(files):
            source = parent / name
            if source.relative_to(ROOT).as_posix() in EXCLUDED_FILES:
                continue
            if (source.is_symlink() or name.startswith("~$")
                    or name in {".DS_Store", "Thumbs.db"}
                    or name.endswith((".pyc", ".pyo", ".pptx.inspect.ndjson"))):
                continue
            yield source


def main():
    if not (ROOT / "bocconi_final_presentation.pdf").is_file():
        raise FileNotFoundError("Missing required submission file: bocconi_final_presentation.pdf")
    destination = ROOT / "submission" / "digitalization.zip"
    destination.parent.mkdir(exist_ok=True)
    # Exclusive creation prevents replacing an earlier submission by accident.
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        count = 0
        for source in submission_files():
            archive.write(source, Path("digitalization") / source.relative_to(ROOT))
            count += 1
    print(f"Wrote {count} files to {destination}")


if __name__ == "__main__":
    main()
