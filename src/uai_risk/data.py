"""Load the numeric-only snapshot and verify its canonical JSON hash."""

import gzip
import hashlib
import json
from pathlib import Path

from .core import ClaimTuple


def load_splits(directory: Path) -> dict:
    # Hash the decompressed bytes, so gzip metadata does not affect integrity.
    manifest = json.loads((directory / "manifest.json").read_text())
    record = manifest["splits"]
    payload = gzip.decompress((directory / record["file"]).read_bytes())
    if hashlib.sha256(payload).hexdigest() != record["json_sha256"]:
        raise ValueError("Split snapshot hash mismatch")
    orders = json.loads(payload)
    for key, order in orders.items():
        # Keys such as "300:42" encode dataset size and random seed. Values are
        # frozen permutations of positions in the sorted query list.
        size, _ = map(int, key.split(":"))
        if sorted(order) != list(range(size)):
            raise ValueError(f"Not a complete permutation: {key}")
    return orders


def load_data(directory: Path, name: str) -> list[ClaimTuple]:
    # This snapshot contains numbers, not response text or claim provenance.
    # The hash checks consistency with the manifest, not scientific correctness.
    manifest = json.loads((directory / "manifest.json").read_text())
    if name not in manifest["datasets"]:
        raise ValueError(f"Unknown dataset: {name}")
    record = manifest["datasets"][name]
    payload = gzip.decompress((directory / record["file"]).read_bytes())
    if hashlib.sha256(payload).hexdigest() != record["json_sha256"]:
        raise ValueError(f"Snapshot hash mismatch: {name}")
    raw = json.loads(payload)
    tuples = [ClaimTuple(t["id"], t["weak"], t["entropy"], t["labels"]) for t in raw]
    ids = [t.query_id for t in tuples]
    if ids != sorted(set(ids)):
        # Reordering tuples silently would change every frozen split.
        raise ValueError("Tuple IDs must be unique and sorted for the frozen splits")
    if len(tuples) != record["tuples"] or sum(len(t.labels) for t in tuples) != record["claims"]:
        raise ValueError("Snapshot count mismatch")
    return tuples
