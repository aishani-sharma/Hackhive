"""Fetch and verify the 70-channel Marine Engine Fault dataset from Zenodo."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from engine_lab.marine import load_runs, summarize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "marine_engine"
RAW = DATA / "raw"
ARCHIVE = DATA / "Marine_Engine_Fault_Data_v1.zip"
DOWNLOAD = "https://zenodo.org/api/records/19857425/files/Marine_Engine_Fault_Data_v1.zip/content"


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            target = (destination / Path(*name.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def main() -> None:
    if not list(RAW.glob("Marine_Engine_Fault_Data/*/*.csv")):
        if not ARCHIVE.exists():
            print(f"Downloading from {DOWNLOAD}")
            urllib.request.urlretrieve(DOWNLOAD, ARCHIVE)
        safe_extract(ARCHIVE, RAW)
    rows = load_runs()
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
