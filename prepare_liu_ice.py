"""Download the official LiU-ICE archive if needed and cache engineered windows."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from engine_lab.liu_ice import SOURCE_URL, build_feature_windows, summarize_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "liu_ice"
RAW = DATA / "raw" / "trainingdata"
ARCHIVE = DATA / "trainingdata.zip"
FEATURES = DATA / "processed_features.csv.gz"


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract zip members only when their resolved paths stay under destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
            target = (destination / Path(*name.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        zipped.extractall(destination)


def main() -> None:
    if not list(RAW.glob("*.csv")):
        if not ARCHIVE.exists():
            print(f"Downloading official LiU-ICE training data from {SOURCE_URL}")
            urllib.request.urlretrieve(SOURCE_URL, ARCHIVE)
        safe_extract(ARCHIVE, DATA / "raw")
    windows = build_feature_windows(RAW)
    windows.to_csv(FEATURES, index=False, compression="gzip")
    print(json.dumps({"feature_file": str(FEATURES), **summarize_dataset(windows)}, indent=2))


if __name__ == "__main__":
    main()
