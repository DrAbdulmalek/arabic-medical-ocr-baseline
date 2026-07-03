#!/usr/bin/env python3
"""
Data Pipeline Setup — Creates the 14-day pipeline directory structure.

Usage:
    python setup_data.py                    # create in ./data/
    python setup_data.py --root /path/to/   # custom root
"""

import argparse
from pathlib import Path


DIRECTORIES = {
    "scans_printed_raw": "Raw scanned printed documents (before any processing)",
    "scans_printed_fixed": "Scanner-fixer processed printed scans (after deskew, crop, enhance)",
    "handwriting_raw": "Raw handwritten medical documents/images",
    "handwriting_labeled": "Handwriting images with verified text labels (ground truth pairs)",
    "ground_truth_verified": "Verified ground truth text files (authoritative)",
    "ocr_corrections": "HITL correction pairs: (original_ocr, corrected_text)",
    "training_exports": "Final export-ready datasets (JSONL, HF format, CSV)",
    "reports": "Benchmark reports, CER/WER statistics, pipeline logs",
    "previews": "Before/after preview images from scanner-fixer",
    "quarantine": "Images that failed processing or need manual review",
}

METADATA_CSV = """filename,type,language,source,quality,has_ground_truth,notes
example_prescription_001.png,scan_printed,ar,hospital_x,high,y,Diabetes medication list
example_lab_result_002.jpg,scan_printed,ar,clinic_y,medium,y,Liver function test
example_handwriting_003.png,handwriting,ar,doctor_z,low,n,Unknown doctor notes - needs labeling
"""

README_TEMPLATE = """# {title}

{description}

## Contents
Files in this directory follow the naming convention:
`{{source}}_{{document_type}}_{{page_number}}.{{ext}}`

Example: `hospital_prescription_001.png`
"""


def setup_pipeline(root: Path) -> None:
    """Create the full data pipeline directory structure."""
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Create subdirectories with READMEs
    for dirname, description in DIRECTORIES.items():
        dir_path = data_root / dirname
        dir_path.mkdir(parents=True, exist_ok=True)

        readme_path = dir_path / "README.md"
        if not readme_path.exists():
            title = dirname.replace("_", " ").title()
            readme_path.write_text(
                README_TEMPLATE.format(title=title, description=description),
                encoding="utf-8",
            )

    # Create metadata template
    meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        meta_path.write_text(METADATA_CSV.strip(), encoding="utf-8")

    # Create .gitkeep in empty dirs
    for d in data_root.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            (d / ".gitkeep").touch()

    print(f"Pipeline structure created at: {data_root}")
    print(f"Subdirectories: {len(DIRECTORIES)}")
    print(f"Metadata template: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Set up data pipeline directory structure")
    parser.add_argument("--root", "-r", default=".", help="Root directory (default: current)")
    args = parser.parse_args()

    setup_pipeline(Path(args.root))


if __name__ == "__main__":
    main()