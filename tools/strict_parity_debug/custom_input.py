"""Create a strict-parity replay artifact from custom token input.

The artifact is intentionally the same format produced by the training-side
monkey patch.  Exact ``input_ids`` are recommended; ``--text`` is a small
convenience path that uses a local Transformers tokenizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _torch():
    import torch

    return torch


def _load_json_or_numbers(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).expanduser().read_text(encoding="utf-8"))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        path = Path(value).expanduser()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return [int(item) for item in value.replace(",", " ").split() if item]


def _rows(value: Any, name: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON array")
    if all(isinstance(item, (int, float)) for item in value):
        return [[int(item) for item in value]]
    if all(isinstance(item, list) for item in value):
        return [[int(item) for item in row] for row in value]
    raise ValueError(f"{name} must be a 1-D or 2-D integer array")


def _pad(rows: list[list[int]], width: int, value: int, *, dtype: Any):
    torch = _torch()
    result = torch.full((len(rows), width), value, dtype=dtype)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, dtype=dtype)
    return result


def _tokenize_text(args: argparse.Namespace) -> tuple[list[list[int]], list[int], str]:
    if not args.tokenizer:
        raise ValueError("--tokenizer is required with --text/--prompt-text")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.text is not None:
        tokens = tokenizer(args.text, add_special_tokens=True, return_tensors=None)["input_ids"]
        return [list(map(int, tokens))], [args.prompt_length if args.prompt_length is not None else 0], "text"
    if args.prompt_text is None or args.response_text is None:
        raise ValueError("--prompt-text and --response-text must be provided together")
    prompt = tokenizer(args.prompt_text, add_special_tokens=True, return_tensors=None)["input_ids"]
    response = tokenizer(args.response_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    return [list(map(int, prompt + response))], [len(prompt)], "prompt_text+response_text"


def _build_fields(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    source = "custom"
    spec: dict[str, Any] = {}
    if args.spec:
        spec = json.loads(Path(args.spec).expanduser().read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("--spec must contain a JSON object")
        input_ids = spec.get("input_ids")
        if input_ids is None:
            raise ValueError("custom input spec requires input_ids")
        input_rows = _rows(input_ids, "input_ids")
        prompt_lengths = spec.get("prompt_lengths")
        if prompt_lengths is None and spec.get("prompt_length") is not None:
            prompt_lengths = [int(spec["prompt_length"])] * len(input_rows)
        source = "spec"
    elif args.text is not None or args.prompt_text is not None:
        input_rows, prompt_lengths, source = _tokenize_text(args)
    elif args.input_ids is not None:
        input_rows = _rows(_load_json_or_numbers(args.input_ids), "input_ids")
        prompt_lengths = [args.prompt_length] * len(input_rows) if args.prompt_length is not None else None
    else:
        raise ValueError("provide exactly one of --spec, --input-ids, or --text")

    batch_size = len(input_rows)
    width = max(len(row) for row in input_rows)

    attention_value = spec.get("attention_mask") if spec else None
    if attention_value is None and args.attention_mask:
        attention_value = _load_json_or_numbers(args.attention_mask)
    attention_rows = _rows(attention_value, "attention_mask") if attention_value is not None else [
        [1] * len(row) for row in input_rows
    ]
    if len(attention_rows) == 1 and batch_size > 1:
        attention_rows *= batch_size
    if len(attention_rows) != batch_size:
        raise ValueError("attention_mask batch size does not match input_ids")

    if prompt_lengths is None:
        prompt_lengths = spec.get("prompt_lengths") if spec else None
    response_mask_value = spec.get("response_mask") if spec else None
    if response_mask_value is None and args.response_mask:
        response_mask_value = _load_json_or_numbers(args.response_mask)
    if prompt_lengths is None and response_mask_value is not None:
        response_hint = _rows(response_mask_value, "response_mask")
        if len(response_hint) == 1 and batch_size > 1:
            response_hint *= batch_size
        prompt_lengths = [
            next((index for index, item in enumerate(row) if int(item) != 0), len(row)) for row in response_hint
        ]
    if prompt_lengths is None:
        raise ValueError("provide prompt_length(s) or an explicit response_mask")
    if isinstance(prompt_lengths, (int, float)):
        prompt_lengths = [int(prompt_lengths)] * batch_size
    else:
        prompt_lengths = [int(item) for item in prompt_lengths]
    if len(prompt_lengths) == 1 and batch_size > 1:
        prompt_lengths *= batch_size
    if len(prompt_lengths) != batch_size:
        raise ValueError("prompt_lengths batch size does not match input_ids")
    for row, mask, prompt_length in zip(input_rows, attention_rows, prompt_lengths, strict=True):
        valid_length = min(len(row), sum(int(item) != 0 for item in mask))
        if prompt_length < 0 or prompt_length > valid_length:
            raise ValueError(
                f"prompt_length={prompt_length} is outside valid input range [0, {valid_length}]"
            )

    if response_mask_value is not None:
        response_rows = _rows(response_mask_value, "response_mask")
        if len(response_rows) == 1 and batch_size > 1:
            response_rows *= batch_size
    else:
        response_rows = []
        for row, mask, prompt_length in zip(input_rows, attention_rows, prompt_lengths, strict=True):
            valid_length = sum(int(item) != 0 for item in mask[: len(row)])
            response_rows.append([0] * prompt_length + [1] * max(0, valid_length - prompt_length))
    if len(response_rows) != batch_size:
        raise ValueError("response_mask batch size does not match input_ids")

    position_value = spec.get("position_ids") if spec else None
    if position_value is None and args.position_ids:
        position_value = _load_json_or_numbers(args.position_ids)
    position_rows = _rows(position_value, "position_ids") if position_value is not None else [
        list(range(len(row))) for row in input_rows
    ]
    if len(position_rows) == 1 and batch_size > 1:
        position_rows *= batch_size
    if len(position_rows) != batch_size:
        raise ValueError("position_ids batch size does not match input_ids")

    padded_input_ids = _pad(input_rows, width, 0, dtype=torch.int64)
    padded_attention = _pad(attention_rows, width, 0, dtype=torch.int64)
    padded_position = _pad(position_rows, width, 0, dtype=torch.int64)
    padded_response = _pad(response_rows, width, 0, dtype=torch.bool)

    prompts = []
    responses = []
    for row, mask, prompt_length in zip(input_rows, attention_rows, prompt_lengths, strict=True):
        valid_length = min(len(row), sum(int(item) != 0 for item in mask))
        valid_row = row[:valid_length]
        prompts.append(valid_row[:prompt_length])
        responses.append(valid_row[prompt_length:])

    fields = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention,
        "position_ids": padded_position,
        "response_mask": padded_response,
        "loss_mask": padded_response.clone(),
        "prompts": prompts,
        "responses": responses,
        "uid": [f"strict_parity_custom_{index}" for index in range(batch_size)],
    }
    metadata = {
        "backend": "custom_input",
        "stage": "custom_replay_artifact",
        "source": source,
        "batch_size": batch_size,
        "sequence_width": width,
        "prompt_lengths": prompt_lengths,
        "tokenizer": args.tokenizer,
    }
    return fields, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="JSON object containing input_ids and optional masks")
    source.add_argument("--input-ids", help="token IDs, JSON array, or @path/to/input_ids.json")
    source.add_argument("--text", help="text tokenized with --tokenizer")
    source.add_argument("--prompt-text", help="prompt text, used together with --response-text")
    parser.add_argument("--response-text", help="response text, used together with --prompt-text")
    parser.add_argument("--tokenizer", help="local tokenizer/model path for text input")
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--attention-mask")
    parser.add_argument("--position-ids")
    parser.add_argument("--response-mask")
    parser.add_argument("--output", required=True, help="replay.pt output path")
    parser.add_argument("--manifest", help="optional manifest path; defaults beside replay.pt")
    args = parser.parse_args()

    from .replay_io import save_replay, write_manifest

    fields, metadata = _build_fields(args)
    replay_path = Path(args.output).expanduser()
    manifest = save_replay(replay_path, fields, metadata)
    write_manifest(args.manifest or replay_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
