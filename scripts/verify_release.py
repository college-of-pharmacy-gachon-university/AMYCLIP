#!/usr/bin/env python3
"""Verify checkpoint hashes and strict compatibility with the public model."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amyclip.model import load_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "checkpoints" / "MANIFEST.json").read_text())
    for record in manifest["checkpoints"]:
        path = ROOT / "checkpoints" / record["file"]
        observed = sha256(path)
        if observed != record["sha256"]:
            raise SystemExit(f"FAIL hash: {path.name}")
        model = load_checkpoint(path)
        if not math.isclose(float(model.temperature), 12.5, rel_tol=0.0, abs_tol=1e-5):
            raise SystemExit(f"FAIL temperature: {path.name}")
        print(f"PASS {path.name} sha256={observed}")
    print("All released checkpoints load strictly into amyclip.model.AMYCLIP")


if __name__ == "__main__":
    main()
