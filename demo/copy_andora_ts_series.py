#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT_DIR = Path("/data/res/test-andorra-ts")
SERIES_DIR = ROOT_DIR / "series"
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}-\d{5}")
SINGLE_DATE_NO_SUFFIX_FILES = {
    "dualpol_rgb": "dualpol_rgb.tif",
}
SINGLE_DATE_NO_FLAT_FILES = {
}
BI_DATE_FILES = {
    "amp_vv": "amp_prm_vv.tif",
    "coh_vv": "coh_vv.tif",
    "phi_vv": "phi_vv.tif",
    "change_vv": "change_vv_abs_gray.tif"
}


def extract_timestamps(folder_name: str) -> list[str]:
    return DATE_PATTERN.findall(folder_name)


def classify_folder(folder_name: str, timestamps: list[str]) -> dict[str, str]:
    if len(timestamps) >= 2:
        return BI_DATE_FILES

    if len(timestamps) == 1:
        if folder_name.endswith("_no_flat"):
            return SINGLE_DATE_NO_FLAT_FILES
        return SINGLE_DATE_NO_SUFFIX_FILES

    return {}


def copy_series_files() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {ROOT_DIR}")

    SERIES_DIR.mkdir(exist_ok=True)

    copied = 0
    skipped = 0

    for folder in sorted(ROOT_DIR.iterdir()):
        if not folder.is_dir() or folder == SERIES_DIR:
            continue

        timestamps = extract_timestamps(folder.name)
        if not timestamps:
            print(f"Skipping {folder.name}: no timestamp found")
            skipped += 1
            continue

        timestamp = timestamps[0]
        files_to_copy = classify_folder(folder.name, timestamps)
        if not files_to_copy:
            print(f"Skipping {folder.name}: no matching folder rule")
            skipped += 1
            continue

        for output_stem, source_name in files_to_copy.items():
            source_file = folder / source_name
            if not source_file.is_file():
                print(f"Skipping {folder.name}: missing {source_name}")
                skipped += 1
                continue

            destination = SERIES_DIR / f"{output_stem}_{timestamp}.tif"
            shutil.copy2(source_file, destination)
            print(f"Copied {source_file} -> {destination}")
            copied += 1

    print(f"Done. Copied {copied} file(s), skipped {skipped} item(s).")


if __name__ == "__main__":
    copy_series_files()
