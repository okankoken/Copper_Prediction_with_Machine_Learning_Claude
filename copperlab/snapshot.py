"""Copy the raw inputs from the source project into this project (read-only on the source).

The copy is what every run uses, so a run is reproducible even if the source
project changes later. Each file is recorded with size and SHA256.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone

from . import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def take_snapshot() -> dict:
    src_raw = C.SOURCE_PROJECT / "data" / "raw"
    entries = []
    for rel in C.RAW_FILES:
        src = src_raw / rel
        dst = C.DATA_RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({"file": f"raw/{rel}", "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    for rel in C.REFERENCE_FILES:
        src = C.SOURCE_PROJECT / rel
        if src.exists():
            dst = C.DATA_REF / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            entries.append({"file": f"reference/{src.name}", "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    manifest = {
        "taken_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_project": str(C.SOURCE_PROJECT),
        "files": entries,
    }
    (C.ROOT / "data" / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
