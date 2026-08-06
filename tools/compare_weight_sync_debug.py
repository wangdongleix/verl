#!/usr/bin/env python3
"""Compare actor-export and vLLM-receive weight-sync debug records.

The input may be a mixed Ray/vLLM log.  It looks for JSON records emitted by
``verl.utils.debug.weight_sync`` and compares the set of signatures observed
for each parameter name.  Repeated records from data/TP ranks are intentionally
deduplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MARKER = "[weight_sync_debug] "


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            marker_index = line.find(MARKER)
            if marker_index < 0:
                continue
            payload = line[marker_index + len(MARKER) :].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                print(f"warning: cannot parse {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(record, dict) and "stage" in record and "name" in record:
                records.append(record)
    return records


def _signature(record: dict[str, Any]) -> tuple[Any, ...] | None:
    signature = record.get("signature")
    if not isinstance(signature, dict):
        return None
    return (
        tuple(signature.get("shape", [])),
        signature.get("dtype"),
        signature.get("numel"),
        signature.get("stats_mode"),
        signature.get("stats_numel"),
        signature.get("mean"),
        signature.get("min"),
        signature.get("max"),
        signature.get("hash_mode"),
        signature.get("sha256"),
    )


def _by_name(records: list[dict[str, Any]], stage: str) -> dict[str, set[tuple[Any, ...]]]:
    result: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        if record.get("stage") != stage:
            continue
        signature = _signature(record)
        if signature is not None:
            result[str(record["name"])].add(signature)
    return result


def compare(path: Path, source_stage: str, receive_stage: str) -> int:
    records = _load_records(path)
    source = _by_name(records, source_stage)
    receive = _by_name(records, receive_stage)

    missing = sorted(set(source) - set(receive))
    extra = sorted(set(receive) - set(source))
    mismatched = sorted(name for name in set(source) & set(receive) if source[name] != receive[name])

    print(f"log: {path}")
    print(f"records: {len(records)}")
    print(f"{source_stage}: {len(source)} unique names")
    print(f"{receive_stage}: {len(receive)} unique names")
    print(f"missing on {receive_stage}: {len(missing)}")
    print(f"extra on {receive_stage}: {len(extra)}")
    print(f"signature mismatches: {len(mismatched)}")

    if missing:
        print("missing examples:")
        for name in missing[:20]:
            print(f"  {name}")
    if extra:
        print("extra examples:")
        for name in extra[:20]:
            print(f"  {name}")
    if mismatched:
        print("mismatch examples:")
        for name in mismatched[:20]:
            print(f"  {name}")
            print(f"    {source[name]}")
            print(f"    {receive[name]}")

    return 1 if missing or extra or mismatched else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="mixed Ray/vLLM log containing weight_sync_debug records")
    parser.add_argument("--source-stage", default="actor_export")
    parser.add_argument("--receive-stage", default="vllm_receive")
    args = parser.parse_args()
    return compare(args.log, args.source_stage, args.receive_stage)


if __name__ == "__main__":
    raise SystemExit(main())
