"""Reward for CountBenchQA outputs rendered with the Kimi K3 XTML template."""

from __future__ import annotations

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
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_STRICT_FINAL_RE = re.compile(r".*\\boxed\s*\{\s*[+-]?\d+\s*\}\s*$", re.DOTALL | re.IGNORECASE)


def extract_response(solution_str: str) -> str | None:
    """Return the first response channel, never an answer from the think channel."""
    think_close = _THINK_CLOSE_RE.search(solution_str)
    if think_close is None:
        return None
    response = solution_str[think_close.end() :]
    response = _RESPONSE_OPEN_RE.sub("", response, count=1)
    response_end = _RESPONSE_END_RE.search(response)
    if response_end is not None:
        response = response[: response_end.start()]
    return response.strip()


def _canonicalize_integer_text(value: object) -> str | None:
    """Return a canonical decimal string without constructing an unbounded int.

    TransferQueue's wire encoder only accepts integers in its 64-bit range.  A
    model response is untrusted text and can contain an arbitrarily long boxed
    integer, so converting it to ``int`` before transport can crash an otherwise
    healthy rollout.  Canonical strings preserve exact equality and auditability
    without depending on Python's integer-string digit limit or a wire integer
    width.
    """
    text = str(value).strip()
    if _INTEGER_RE.fullmatch(text) is None:
        return None

    negative = text.startswith("-")
    digits = text.lstrip("+-").lstrip("0") or "0"
    if digits == "0":
        return "0"
    return f"-{digits}" if negative else digits


def extract_boxed_integer(response: str) -> str | None:
    """Extract the last boxed integer as a canonical decimal string."""
    matches = list(_BOXED_INTEGER_RE.finditer(response))
    if not matches:
        return None
    groups = matches[-1].groups()
    value = next(group for group in groups if group is not None)
    return _canonicalize_integer_text(value)


def compute_score(
    solution_str: str, ground_truth: str, format_score: float = 0.1
) -> dict[str, float | str | bool | None]:
    """Score CountBench and expose auditable components without changing the total."""
    expected = _canonicalize_integer_text(ground_truth)
    if expected is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "predicted": None,
            "expected": None,
            "has_response": False,
        }

    response = extract_response(str(solution_str))
    predicted = extract_boxed_integer(response) if response is not None else None
    accuracy = float(predicted == expected)
    valid_format = float(response is not None and bool(_STRICT_FINAL_RE.fullmatch(response)))
    score = (1.0 - format_score) * accuracy + format_score * valid_format
    return {
        "score": score,
        "acc": accuracy,
        "format": valid_format,
        "predicted": predicted,
        "expected": expected,
        "has_response": response is not None,
    }
