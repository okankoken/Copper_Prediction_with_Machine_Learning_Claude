"""Copy the raw inputs from the source project into this project (read-only on the source).

The source project (Copper_Prediction_with_Machine_Learning, which refreshes LME and macro
data with Airflow) can be
  * a local folder   : COPPER_SOURCE_PROJECT=/path/to/project (default ~/Copper_Prediction_with_Machine_Learning)
  * on another host  : COPPER_SOURCE_SSH=user@host:path        (default on a PC when the local folder is missing:
                       train@127.0.0.1:Copper_Prediction_with_Machine_Learning, i.e. the VirtualBox VM)
Remote files are fetched with one `ssh ... tar` call. The copy is what every run uses, so a run is
reproducible even if the source changes later. Each file is recorded with size and SHA256.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

from . import config as C

DEFAULT_SSH = "train@127.0.0.1:Copper_Prediction_with_Machine_Learning"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source() -> tuple[str, str]:
    """('local', path) or ('ssh', 'user@host:path')."""
    if os.environ.get("COPPER_SOURCE_SSH"):
        return "ssh", os.environ["COPPER_SOURCE_SSH"]
    if C.SOURCE_PROJECT.exists():
        return "local", str(C.SOURCE_PROJECT)
    return "ssh", DEFAULT_SSH


def _fetch_ssh(spec: str, rels: list[str]) -> None:
    host, path = spec.split(":", 1)
    files = " ".join(f"data/raw/{r}" for r in rels)
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host,
                          f"cd {path} && tar cf - {files}"], capture_output=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"could not read source data over SSH from {spec}: {out.stderr.decode(errors='ignore')}")
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        for m in tar.getmembers():
            if m.isfile():
                rel = m.name.split("data/raw/", 1)[1]
                dst = C.DATA_RAW / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tar.extractfile(m).read())


def take_snapshot() -> dict:
    kind, where = source()
    rels = list(C.RAW_FILES)
    if kind == "ssh":
        _fetch_ssh(where, rels)
    else:
        for rel in rels:
            dst = C.DATA_RAW / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(C.SOURCE_PROJECT / "data" / "raw" / rel, dst)
    entries = [{"file": f"raw/{rel}", "bytes": (C.DATA_RAW / rel).stat().st_size, "sha256": sha256(C.DATA_RAW / rel)}
               for rel in rels]
    manifest = {
        "taken_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_project": where,
        "source_kind": kind,
        "files": entries,
    }
    (C.ROOT / "data" / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
