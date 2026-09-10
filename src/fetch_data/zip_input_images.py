"""
Package data/raw/<task> images -- images only, no ground truth -- into
data/data.zip, ready to upload as the input zip for the src/engines/
notebooks.

Mirrors the layout of the existing example bundle (unzip data/data.zip to
see it):

    handwritten_en/<id>.jpg
    handwritten_zh/<id>.jpg
    table/<id>.jpg
    table_by_level/<level_x>/<id>.jpg   (optional)

handwritten_en, handwritten_zh, and table are mandatory: each must exist
under data/raw/ (run src/fetch_data/fetch_all.py, or the individual fetch
scripts, first) or this script aborts. table_by_level is optional -- if
data/raw/table_by_level wasn't fetched (e.g. select_table.py was skipped),
it's simply left out of the zip.

Run: python src/fetch_data/zip_input_images.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "data.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MANDATORY_TASKS = ["handwritten_en", "handwritten_zh", "table"]


def _find_image_dir(task_dir: Path) -> Path | None:
    for name in ("images", "img"):
        candidate = task_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def collect_flat_task(task: str) -> list[tuple[Path, str]]:
    """handwritten_en / handwritten_zh / table -> [(file, "<task>/<name>")]."""
    image_dir = _find_image_dir(RAW_DIR / task)
    if image_dir is None:
        return []
    return [(f, f"{task}/{f.name}") for f in _list_images(image_dir)]


def collect_table_by_level() -> list[tuple[Path, str]]:
    """table_by_level/<level>/img/* -> [(file, "table_by_level/<level>/<name>")]."""
    tbl_dir = RAW_DIR / "table_by_level"
    if not tbl_dir.is_dir():
        return []

    entries: list[tuple[Path, str]] = []
    for level_dir in sorted(tbl_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        image_dir = _find_image_dir(level_dir)
        if image_dir is None:
            continue
        entries.extend(
            (f, f"table_by_level/{level_dir.name}/{f.name}")
            for f in _list_images(image_dir)
        )
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Zip data/raw/<task> images into data/data.zip for engine notebook upload."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output zip path (default: data/data.zip).")
    parser.add_argument("--dry-run", action="store_true", help="List what would be zipped without writing.")
    args = parser.parse_args()

    out_path = Path(args.out)

    entries: list[tuple[Path, str]] = []
    missing_mandatory = []

    for task in MANDATORY_TASKS:
        task_entries = collect_flat_task(task)
        if not task_entries:
            missing_mandatory.append(task)
        else:
            entries.extend(task_entries)
        print(f"{task}: {len(task_entries)} image(s)")

    if missing_mandatory:
        parser.error(
            f"Missing/empty required task(s) under {RAW_DIR}: {missing_mandatory}. "
            f"Run the corresponding src/fetch_data/*.py script(s) first."
        )

    tbl_entries = collect_table_by_level()
    if tbl_entries:
        entries.extend(tbl_entries)
        print(f"table_by_level: {len(tbl_entries)} image(s)")
    else:
        print("table_by_level: not found under data/raw -- skipping (optional).")

    print(f"\n{len(entries)} image(s) total.")

    if args.dry_run:
        print(f"[DRY RUN] Would write {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in entries:
            zf.write(src, arcname)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
