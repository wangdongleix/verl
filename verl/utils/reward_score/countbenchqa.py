# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Reward for CountBenchQA outputs rendered with the Kimi-K3 XTML template."""

import re


_THINK_CLOSE_RE = re.compile(r"<\|close\|>\s*think\s*<\|sep\|>", re.IGNORECASE)
_RESPONSE_OPEN_RE = re.compile(r"^\s*<\|open\|>\s*response\s*<\|sep\|>", re.IGNORECASE)
_RESPONSE_END_RE = re.compile(
    r"<\|close\|>\s*(?:response|message)\s*<\|sep\|>|<\|end_of_msg\|>",
    re.IGNORECASE,
)
_BOXED_INTEGER_RE = re.compile(
    r"\\boxed\s*(?:\{\s*([+-]?\d+)\s*\}|\(\s*([+-]?\d+)\s*\)|([+-]?\d+))",
    re.IGNORECASE,
)
_STRICT_FINAL_RE = re.compile(r".*\\boxed\s*\{\s*[+-]?\d+\s*\}\s*$", re.DOTALL | re.IGNORECASE)


def _parse_response(solution_str: str) -> tuple[str | None, bool]:
    """Return the response channel and whether its opening marker was present."""
    think_close = _THINK_CLOSE_RE.search(solution_str)
    if think_close is None:
        return None, False

    response = solution_str[think_close.end() :]
    response_open = _RESPONSE_OPEN_RE.match(response)
    has_response_open = response_open is not None
    if response_open is not None:
        response = response[response_open.end() :]
    response_end = _RESPONSE_END_RE.search(response)
    if response_end is not None:
        response = response[: response_end.start()]
    return response.strip(), has_response_open


def extract_boxed_integer(response: str) -> int | None:
    """Extract the last boxed integer from a response channel."""
    matches = list(_BOXED_INTEGER_RE.finditer(response))
    if not matches:
        return None

    value = next(group for group in matches[-1].groups() if group is not None)
    return int(value)


def compute_score(
    solution_str: str, ground_truth: str, format_score: float = 0.1
) -> dict[str, float | int | bool | None]:
    """Score integer accuracy and strict final-answer format separately."""
    try:
        expected = int(str(ground_truth).strip())
    except (TypeError, ValueError):
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "predicted": None,
            "expected": None,
            "has_response": False,
        }

    response, has_response_open = _parse_response(str(solution_str))
    predicted = extract_boxed_integer(response) if response is not None else None
    accuracy = float(predicted == expected)
    valid_format = float(
        response is not None
        and has_response_open
        and bool(_STRICT_FINAL_RE.fullmatch(response))
    )
    return {
        "score": (1.0 - format_score) * accuracy + format_score * valid_format,
        "acc": accuracy,
        "format": valid_format,
        "predicted": predicted,
        "expected": expected,
        "has_response": response is not None,
    }
