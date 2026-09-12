#!/usr/bin/env python3
"""Convert a raw CountBenchQA-lite parquet shard to verl RL parquet.

The raw shard contains ``image``, ``text``, ``question`` and ``number``.
verl's RLHFDataset expects a chat-style ``prompt`` column and multimodal
images in an ``images`` column, plus the ground-truth answer in
``reward_model``.

This script deliberately does not copy ``text`` into the output prompt:
CountBenchQA captions can contain the answer and would leak the label.
"""

from __future__ import annotations

import argparse
import os

import datasets

INSTRUCTION = (
    " You FIRST think about the reasoning process as an internal monologue and then provide the final answer."
    " The reasoning process MUST BE enclosed within <think> </think> tags."
    " The final answer MUST BE put in \\boxed{}."
)


def convert(input_path: str, output_path: str, split: str = "test") -> None:
    dataset = datasets.load_dataset("parquet", data_files={"data": input_path})["data"]
    required = {"image", "question", "number"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Raw CountBenchQA parquet is missing columns {sorted(missing)}; found {dataset.column_names}")

    def make_example(example: dict, index: int) -> dict:
        question = str(example["question"])
        answer = str(example["number"])
        return {
            "data_source": "countbenchqa_lite",
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>\n{question}{INSTRUCTION}",
                }
            ],
            "images": [example["image"]],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": index,
                "answer": answer,
                "question": question,
            },
        }

    converted = dataset.map(
        make_example,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Converting CountBenchQA-lite to verl format",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    converted.to_parquet(output_path)

    print(f"wrote {len(converted)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw CountBenchQA parquet shard")
    parser.add_argument("--output", required=True, help="verl-format parquet output")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    convert(args.input, args.output, args.split)


if __name__ == "__main__":
    main()
