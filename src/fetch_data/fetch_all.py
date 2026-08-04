"""
Runs the full fetch_data pipeline in order:
  table.py -> select_table.py -> handwritten_en.py -> handwritten_zh.py

select_table.py depends on table.py's output (data/raw/table/images, gt),
so order matters. Each script is run as its own subprocess (they execute as
top-level module code, not via a main() function).

Run from the project root: python src/fetch_data/fetch_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "table.py",
    "select_table.py",
    "handwritten_en.py",
    "handwritten_zh.py",
]


def main():
    for name in SCRIPTS:
        print(f"=== Running {name} ===")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / name)])
        if result.returncode != 0:
            print(f"{name} failed (exit code {result.returncode}), stopping.")
            sys.exit(1)

    print("All fetch_data scripts completed successfully.")


if __name__ == "__main__":
    main()
