"""Validate and render handouts. Use --full for checks before publication."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run the tests and all notebooks before rendering.",
    )
    args = parser.parse_args(argv)
    quarto = shutil.which("quarto")
    if not quarto:
        parser.exit(
            1, "Quarto is not installed. Install Quarto to render PDF/DOCX handouts.\n"
        )
    validation = [sys.executable, str(ROOT / "scripts/validate_materials.py")]
    if args.full:
        validation.append("--full")
    try:
        subprocess.run(validation, cwd=ROOT, check=True)
        subprocess.run([quarto, "render", "--to", "all"], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        parser.exit(
            error.returncode, "Build stopped because a check or render failed.\n"
        )
    print(f'Handouts rendered beside their QMD sources under {ROOT / "hands-on"}.')
    print("Review changed PDF/DOCX layouts before publishing.")


if __name__ == "__main__":
    main()
