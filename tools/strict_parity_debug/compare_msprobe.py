"""Compare train and rollout msprobe ``dump.json`` statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


STAT_KEY_MAP = {
    "Max": "max",
    "Min": "min",
    "Mean": "mean",
    "Norm": "norm",
    "shape": "shape",
    "dtype": "dtype",
}
CHILD_KEY_MAP = {
    "input_args": "input",
    "input_kwargs": "input",
}


@dataclass
class Record:
    name: str
    stats: dict[str, Any]
    source: str
    ordinal: int


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, Mapping):
        for key in ("value", "val", "real", "data"):
            if key in value:
                result = _number(value[key])
                if result is not None:
                    return result
    return None


def _record_stats(node: Mapping[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for source_key, target_key in STAT_KEY_MAP.items():
        if source_key not in node:
            continue
        value = node[source_key]
        if target_key == "shape":
            if isinstance(value, (list, tuple)):
                stats[target_key] = [int(item) if isinstance(item, (int, float)) else item for item in value]
            else:
                stats[target_key] = value
        elif target_key == "dtype":
            stats[target_key] = str(value)
        elif _number(value) is not None:
            stats[target_key] = _number(value)
    return stats


def _child_name(parent: str, key: Any) -> str:
    part = CHILD_KEY_MAP.get(str(key), str(key))
    return f"{parent}.{part}"


def _walk(node: Any, source: str, records: list[Record], name: str) -> None:
    if isinstance(node, Mapping):
        stats = _record_stats(node)
        if stats:
            records.append(Record(name=name, stats=stats, source=source, ordinal=len(records)))
        for key, value in node.items():
            if isinstance(value, (Mapping, list, tuple)):
                _walk(value, source, records, _child_name(name, key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, source, records, _child_name(name, index))


def _read_json_file(path: Path) -> list[Record]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping):
        return []
    records: list[Record] = []
    for operation_name, operation_data in data.items():
        _walk(operation_data, str(path), records, str(operation_name))
    return records


def _input_files(path: str | os.PathLike[str]) -> list[Path]:
    target = Path(path).expanduser()
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)
    files = [item for item in target.rglob("dump.json") if item.is_file()]
    return sorted(files)


def _load_records(
    path: str | os.PathLike[str], name_map: Mapping[str, str] | None = None
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    grouped: dict[str, list[Record]] = {}
    files = _input_files(path)
    files_with_records = 0
    for file_path in files:
        records = _read_json_file(file_path)
        files_with_records += bool(records)
        for record in records:
            name = name_map.get(record.name, record.name) if name_map else record.name
            grouped.setdefault(name, []).append(Record(name, record.stats, record.source, record.ordinal))
    return grouped, {
        "path": str(Path(path).expanduser()),
        "dump_files": len(files),
        "files_with_records": files_with_records,
        "file_examples": [str(file_path) for file_path in files[:3]],
    }


def load_records(path: str | os.PathLike[str], name_map: Mapping[str, str] | None = None) -> dict[str, list[Record]]:
    return _load_records(path, name_map)[0]


def _load_name_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("name map must be a JSON object: source_name -> canonical_name")
    return {str(key): str(item) for key, item in value.items()}


def _same_value(left: Any, right: Any, atol: float, rtol: float) -> tuple[bool, dict[str, Any] | None]:
    left_number, right_number = _number(left), _number(right)
    if left_number is not None and right_number is not None:
        if math.isnan(left_number) or math.isnan(right_number):
            equal = math.isnan(left_number) and math.isnan(right_number)
            return equal, None if equal else {"train": left, "rollout": right}
        diff = abs(left_number - right_number)
        tolerance = atol + rtol * abs(left_number)
        return diff <= tolerance, {
            "train": left,
            "rollout": right,
            "abs_diff": diff,
            "tolerance": tolerance,
        }
    return left == right, {"train": left, "rollout": right}


def compare(
    train: Mapping[str, list[Record]],
    rollout: Mapping[str, list[Record]],
    *,
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> dict[str, Any]:
    train_names, rollout_names = set(train), set(rollout)
    missing = sorted(train_names - rollout_names)
    extra = sorted(rollout_names - train_names)
    mismatches: list[dict[str, Any]] = []
    total_mismatches = 0
    matched_records = 0
    for name in sorted(train_names & rollout_names):
        left_records, right_records = train[name], rollout[name]
        if len(left_records) != len(right_records):
            total_mismatches += 1
            if len(mismatches) < max_mismatches:
                mismatches.append(
                    {
                        "name": name,
                        "kind": "occurrence_count",
                        "train": len(left_records),
                        "rollout": len(right_records),
                    }
                )
        for occurrence, (left, right) in enumerate(zip(left_records, right_records)):
            matched_records += 1
            differences: dict[str, Any] = {}
            for key in sorted(set(left.stats) | set(right.stats)):
                if key not in left.stats or key not in right.stats:
                    differences[key] = {"train": left.stats.get(key), "rollout": right.stats.get(key)}
                    continue
                equal, detail = _same_value(left.stats[key], right.stats[key], atol, rtol)
                if not equal and detail is not None:
                    differences[key] = detail
            if differences and len(mismatches) < max_mismatches:
                total_mismatches += 1
                mismatches.append(
                    {
                        "name": name,
                        "occurrence": occurrence,
                        "kind": "statistics",
                        "differences": differences,
                        "train_source": left.source,
                        "rollout_source": right.source,
                    }
                )
            elif differences:
                total_mismatches += 1
    return {
        "train_unique_names": len(train_names),
        "rollout_unique_names": len(rollout_names),
        "train_records": sum(len(items) for items in train.values()),
        "rollout_records": sum(len(items) for items in rollout.values()),
        "matched_records": matched_records,
        "missing_on_rollout": missing,
        "extra_on_rollout": extra,
        "mismatch_count": total_mismatches,
        "reported_mismatches": len(mismatches),
        "mismatches": mismatches,
        "first_divergence": mismatches[0] if mismatches else None,
        "no_records": not train or not rollout,
        "equal": bool(train) and bool(rollout) and not missing and not extra and total_mismatches == 0,
        "tolerance": {"atol": atol, "rtol": rtol},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="training msprobe dump.json file or parent directory")
    parser.add_argument("--rollout", required=True, help="rollout msprobe dump.json file or parent directory")
    parser.add_argument("--output", default="strict_parity_output/msprobe_compare.json")
    parser.add_argument("--name-map", help="optional JSON object mapping source names to canonical names")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-mismatches", type=int, default=100)
    args = parser.parse_args(list(argv) if argv is not None else None)
    name_map = _load_name_map(args.name_map)
    train, train_input = _load_records(args.train, name_map)
    rollout, rollout_input = _load_records(args.rollout, name_map)
    result = compare(train, rollout, atol=args.atol, rtol=args.rtol, max_mismatches=args.max_mismatches)
    result["inputs"] = {"train": train_input, "rollout": rollout_input}
    if result["no_records"]:
        result["diagnostic"] = (
            "No native msprobe tensor statistics were loaded from one or both inputs; "
            "check dump_files/files_with_records and the selected dump roots."
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["equal"] else 2


if __name__ == "__main__":
    sys.exit(main())
