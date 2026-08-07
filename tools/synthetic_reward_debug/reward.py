"""Build deterministic group-varying reward tensors without importing verl."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _split_batch_key(key: str) -> tuple[str, str]:
    fields = key.rsplit("_", 2)
    if len(fields) != 3:
        raise ValueError(f"Unexpected V1 batch key format: {key!r}")
    uid, session_id, output_index = fields
    try:
        int(output_index)
    except ValueError as exc:
        raise ValueError(f"Unexpected V1 batch output index in key: {key!r}") from exc
    return uid, session_id


def _session_sort_key(session_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(session_id))
    except ValueError:
        return (1, session_id)


def build_group_rank_sequence_rewards(
    batch_keys: Sequence[str], active_rows: Sequence[bool], scale: float
) -> list[float]:
    """Assign evenly spaced rewards to sessions within each prompt UID."""
    if len(batch_keys) != len(active_rows):
        raise ValueError(
            f"batch_keys and active_rows must have equal length, got {len(batch_keys)} and {len(active_rows)}"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    if not any(active_rows):
        raise ValueError("Synthetic GRPO reward needs at least one valid response row")

    row_sessions: list[tuple[str, str] | None] = []
    uid_to_sessions: dict[str, set[str]] = defaultdict(set)
    for key, active in zip(batch_keys, active_rows, strict=True):
        if not active:
            row_sessions.append(None)
            continue
        uid, session_id = _split_batch_key(str(key))
        row_sessions.append((uid, session_id))
        uid_to_sessions[uid].add(session_id)

    session_rewards: dict[tuple[str, str], float] = {}
    for uid, session_ids in uid_to_sessions.items():
        ordered_sessions = sorted(session_ids, key=_session_sort_key)
        if len(ordered_sessions) < 2:
            raise ValueError(
                f"Synthetic GRPO reward needs at least two rollout sessions for uid {uid!r}; "
                "set actor_rollout_ref.rollout.n>=2"
            )
        denominator = len(ordered_sessions) - 1
        for rank, session_id in enumerate(ordered_sessions):
            session_rewards[(uid, session_id)] = -scale + (2.0 * scale * rank / denominator)

    return [0.0 if session is None else session_rewards[session] for session in row_sessions]


def build_synthetic_rm_scores(batch_keys: Sequence[str], response_mask: Any, scale: float):
    """Place each sequence reward on its final valid response token."""
    import torch

    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")
    valid_mask = response_mask.bool()
    active_rows = valid_mask.any(dim=-1)
    sequence_rewards = build_group_rank_sequence_rewards(batch_keys, active_rows.cpu().tolist(), scale)
    sequence_rewards = torch.tensor(sequence_rewards, dtype=torch.float32, device=response_mask.device)

    rm_scores = torch.zeros(response_mask.shape, dtype=torch.float32, device=response_mask.device)
    if active_rows.any():
        positions = torch.arange(response_mask.size(1), device=response_mask.device).unsqueeze(0)
        last_valid = torch.where(valid_mask, positions, -1).max(dim=-1).values
        row_indices = active_rows.nonzero(as_tuple=False).squeeze(-1)
        rm_scores[row_indices, last_valid[row_indices]] = sequence_rewards[row_indices]
    return rm_scores, sequence_rewards, active_rows
