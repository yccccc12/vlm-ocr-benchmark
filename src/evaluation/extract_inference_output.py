"""
Extract per-engine prediction zips from inference_output/ into outputs/,
matching the layout evaluation scripts expect (and that outputs_/ shows
as a worked reference):

    outputs/<task>/<model>/<sample_id>/...
    outputs/table_by_level/<model>/<level_x>/<sample_id>/...

Each zip in inference_output/ is expected to bundle handwritten_en,
handwritten_zh, and table (mandatory) plus computational-cost logs;
table_by_level is optional -- a zip that doesn't include it is not an
error, it's just skipped for that model. Different notebooks zip these
under different top-level prefixes (e.g. "content/output/...",
"output/...", "content/MonkeyOCR/output/..."). Rather than relying on a
fixed prefix, this script locates the known task folder name inside each
zip entry's path and keeps everything from there onward -- so per-model
prediction file layouts (which vary by engine; see eval_handwritten_en.py's
extract_pred_text) are preserved exactly.

Computational logs (log.csv, log.json, log_summary.txt, Computational/...)
are skipped -- evaluation doesn't use them.

Does not touch inference_output/ or outputs_/ (zips are opened read-only).

Tesseract has no zip -- it writes straight to outputs/ when you run
src/engines/tesseract.py.

Run: python src/evaluation/extract_inference_output.py
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = ROOT / "inference_output"
OUTPUTS_DIR = ROOT / "outputs"

MANDATORY_TASKS = {"handwritten_en", "handwritten_zh", "table"}
OPTIONAL_TASKS = {"table_by_level"}
TASK_DIRS = MANDATORY_TASKS | OPTIONAL_TASKS

# zip filename (without .zip) -> model key used under outputs/<task>/<model>/
ZIP_TO_MODEL = {
    "baidu_unlimited_ocr_output": "baidu_ocr",
    "deepseek-OCR_output": "deepseekOCR",
    "deepseek-OCR2_output": "deepseekOCR2",
    "dots_mocr_output": "dots_mocr",
    "glm_ocr_output": "glm_ocr",
    "mineru_output": "mineru",
    "monkey_ocr_output": "monkey_ocr",
    "paddle_ocr_vl_1_5_output": "paddle_vl_1.5",
    "paddle_ocr_vl_1_6_output": "paddle_vl_1.6",
}


def extract_zip(zip_path: Path, model: str, outputs_dir: Path, tasks: set[str],
                 dry_run: bool) -> tuple[dict[str, int], int]:
    counts = {t: 0 for t in tasks}
    skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            parts = info.filename.split("/")
            task = None
            task_idx = None
            for i, part in enumerate(parts):
                if part in TASK_DIRS:
                    task, task_idx = part, i
                    break

            if task is None or task not in tasks:
                skipped += 1
                continue

            rel_parts = parts[task_idx + 1:]
            if not rel_parts:
                continue

            dest = outputs_dir / task / model / Path(*rel_parts)
            counts[task] += 1

            if dry_run:
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                out.write(src.read())

    return counts, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract inference_output/*.zip into outputs/, matching the outputs_/ layout."
    )
    parser.add_argument("--inference-dir", default=str(INFERENCE_DIR),
                         help="Folder containing the per-engine *.zip files.")
    parser.add_argument("--outputs-dir", default=str(OUTPUTS_DIR),
                         help="Destination root (default: outputs/).")
    parser.add_argument("--tasks", default=",".join(sorted(TASK_DIRS)),
                         help="Comma-separated subset of tasks to extract.")
    parser.add_argument("--dry-run", action="store_true",
                         help="List what would be extracted without writing any files.")
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    outputs_dir = Path(args.outputs_dir)
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    unknown = tasks - TASK_DIRS
    if unknown:
        parser.error(f"Unknown task(s): {sorted(unknown)}. Valid: {sorted(TASK_DIRS)}")

    zips = sorted(inference_dir.glob("*.zip"))
    if not zips:
        parser.error(f"No zip files found in {inference_dir}")

    print(f"Found {len(zips)} zip(s) in {inference_dir}")
    print(f"{'[DRY RUN] Would extract' if args.dry_run else 'Extracting'} into {outputs_dir}\n")

    seen_models = set()
    mandatory_gaps: list[str] = []

    for zip_path in zips:
        model = ZIP_TO_MODEL.get(zip_path.stem)
        if model is None:
            print(f"[SKIP] {zip_path.name}: no entry in ZIP_TO_MODEL for this filename.")
            continue
        seen_models.add(model)

        print(f"[{model}] {zip_path.name}")
        counts, skipped = extract_zip(zip_path, model, outputs_dir, tasks, args.dry_run)
        for task in sorted(counts):
            if counts[task] > 0:
                print(f"    {task}: {counts[task]} files")
            elif task in OPTIONAL_TASKS:
                print(f"    {task}: not included in this zip (optional) -- skipping")
            else:
                print(f"    {task}: 0 files [ERROR] this task is required")
                mandatory_gaps.append(f"{model}/{task}")
        print(f"    (skipped {skipped} non-task entries, e.g. computational logs)")

    missing = set(ZIP_TO_MODEL.values()) - seen_models
    if missing:
        print(f"\n[WARNING] No zip found for: {sorted(missing)}")

    print(f"\n{'[DRY RUN] Done.' if args.dry_run else 'Done.'}")
    print("Reminder: tesseract has no zip -- run `python src/engines/tesseract.py` to populate")
    print("outputs/handwritten_en/tesseract/ and outputs/handwritten_zh/tesseract/.")

    if mandatory_gaps:
        print(f"\n[ERROR] Missing mandatory data for: {mandatory_gaps}")
        sys.exit(1)


if __name__ == "__main__":
    main()
