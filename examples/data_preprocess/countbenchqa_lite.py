# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the CountBenchQA-lite dataset to parquet format for verl RL.

The output schema (images / data_source / prompt / ability / reward_model / extra_info)
matches the geo3k preprocessing script, so the training and reward logic is identical.

Raw schema (from gsarch/countbenchqa_lite, upstream vikhyatk/CountBenchQA):
    image    : dict {'bytes': ..., 'path': ...}   # single image per sample
    text     : str  # caption, may contain the answer in words (NOT used in the prompt)
    question : str  # e.g. "How many labels are there in the image?"
    number   : int  # the ground-truth count
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/countbenchqa_lite", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Fraction of samples held out for the test/val split."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the stratified train/test split.")

    args = parser.parse_args()

    data_source = "countbenchqa_lite"

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset("gsarch/countbenchqa_lite")

    # The lite subset ships a single "test" split, so split it into train/test here.
    # Stratify on the answer column so both splits cover all count values.
    split = dataset["test"].train_test_split(test_size=args.test_ratio, seed=args.seed, stratify_by_column="number")
    train_dataset = split["train"]
    test_dataset = split["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # `text` is a caption that can spell out the answer (e.g. "A set of nine labels ..."),
            # so it must be dropped to avoid leaking the ground truth into the prompt.
            example.pop("text", None)
            question = example.pop("question")
            prompt = "<image>" + question + " " + instruction_following
            answer = str(example.pop("number"))
            image = example.pop("image")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [image],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
