"""Prepare a separate checkpoint view using the current training model code."""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("target", type=Path)
parser.add_argument("--code", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve(strict=True)
target = args.target
if target.resolve() == source:
    raise ValueError("Model view must be separate from the source checkpoint")
target.mkdir(parents=True, exist_ok=True)
for src in source.iterdir():
    dst = target / src.name
    if src.is_file() and src.suffix == ".safetensors":
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"Unexpected weight link: {dst}")
        elif not dst.exists():
            dst.symlink_to(src.resolve())
    elif src.is_file() and src.suffix in {".json", ".py", ".jinja", ".txt", ".model", ".tiktoken"}:
        if dst.is_symlink():
            raise ValueError(f"Metadata must not be a symlink: {dst}")
        shutil.copy2(src, dst)
for src in args.code.glob("*.py"):
    dst = target / src.name
    if dst.is_symlink():
        raise ValueError(f"Model code must not be a symlink: {dst}")
    shutil.copy2(src, dst)
