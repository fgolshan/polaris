#!/usr/bin/env python3
"""
expand_templates.py

Usage:
  python3 expand_templates.py /path/to/TEMPLATE_DIR /path/to/OUTPUT_DIR 5

What it does:
- Finds all *subdirectories* in TEMPLATE_DIR whose names end with "template"
  (e.g., "pca_template").
- For each such subdirectory, creates N copies under OUTPUT_DIR named like:
  <base>_1, <base>_2, ..., <base>_N, where <base> is the folder name with the
  trailing "template" removed (and any trailing underscore stripped).
- Copies the full contents (files and subfolders) of each template folder into
  each newly created output folder.
"""

import argparse
import sys
from pathlib import Path
import shutil

def copy_contents(src: Path, dst: Path) -> None:
    """Copy *contents* of src directory (both files and subdirectories) into dst."""
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            # Copy whole subdirectory; dirs_exist_ok allows merging into existing dirs
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            # Copy single file with metadata
            shutil.copy2(item, target)

def normalize_base(name: str) -> str:
    """
    Remove the trailing 'template' from name, then strip a trailing underscore if present.
    Example: 'pca_template' -> 'pca'
             'mytemplate'    -> 'my'
    """
    if not name.endswith("template"):
        return name  # Shouldn't happen given our filter, but be safe
    base = name[: -len("template")]
    # Avoid ending up with 'pca__1' when the name was 'pca_template'
    if base.endswith("_"):
        base = base[:-1]
    return base

def main():
    parser = argparse.ArgumentParser(description="Expand template subfolders into numbered copies.")
    parser.add_argument("template_dir", type=Path, help="Path to the directory containing template subfolders")
    parser.add_argument("output_dir", type=Path, help="Path to the directory where numbered folders will be created")
    parser.add_argument("n", type=int, help="Number of copies to create for each template subfolder")
    args = parser.parse_args()

    template_dir: Path = args.template_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    n: int = args.n

    if n <= 0:
        print("Error: n must be a positive integer.", file=sys.stderr)
        sys.exit(2)

    if not template_dir.exists() or not template_dir.is_dir():
        print(f"Error: template_dir does not exist or is not a directory: {template_dir}", file=sys.stderr)
        sys.exit(2)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subdirectories whose names end with "template"
    template_subdirs = [d for d in template_dir.iterdir() if d.is_dir() and d.name.endswith("template")]

    if not template_subdirs:
        print(f"Warning: No subdirectories ending with 'template' found in {template_dir}")
        sys.exit(0)

    for tmpl in template_subdirs:
        base = normalize_base(tmpl.name)

        for k in range(1, n + 1):
            dest = output_dir / f"{base}_{k}"
            dest.mkdir(parents=True, exist_ok=True)
            copy_contents(tmpl, dest)
            print(f"Created: {dest}")

    print("Done.")

if __name__ == "__main__":
    main()
