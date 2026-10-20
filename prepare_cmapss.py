"""Download and safely extract NASA's public C-MAPSS FD001 prognostics data."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import zipfile

from engine_lab.cmapss import DATA_DIR

ARCHIVE = DATA_DIR / "CMAPSSData.zip"
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
REQUIRED = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size == 0:
        print("Downloading NASA C-MAPSS archive…")
        partial = ARCHIVE.with_suffix(".zip.part")
        try:
            with urlopen(URL, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if partial.stat().st_size == 0:
                raise ValueError("The NASA download returned an empty archive.")
            partial.replace(ARCHIVE)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    if not all((DATA_DIR / name).is_file() for name in REQUIRED):
        safe_extract(ARCHIVE, DATA_DIR)
    candidates = list(DATA_DIR.rglob("train_FD001.txt"))
    if not candidates:
        raise FileNotFoundError("The archive did not contain train_FD001.txt.")
    source = candidates[0].parent
    for name in REQUIRED:
        if not (DATA_DIR / name).is_file():
            (source / name).replace(DATA_DIR / name)
    print(f"NASA FD001 files are ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
