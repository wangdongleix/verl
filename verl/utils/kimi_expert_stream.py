# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bounded Kimi expert tiles on the existing named-tensor checkpoint wire."""

TILE_MARKER = ".__verl_expert_tile__."
EXPERT_MARKER = ".__verl_streamed_expert__."
LOCAL_MARKER = ".__verl_packed_local__."


def expert_tile_name(name, expert, rows, columns, row, column, num_experts=1):
    return f"{name}{TILE_MARKER}{expert}.{num_experts}.{rows}.{columns}.{row}.{column}"


def parse_expert_tile(name):
    if TILE_MARKER not in name:
        return None
    base, encoded = name.rsplit(TILE_MARKER, 1)
    values = encoded.split(".")
    if len(values) != 6 or not all(value.isdecimal() for value in values):
        raise ValueError(f"Invalid Kimi expert tile name: {name}")
    expert, num_experts, rows, columns, row, column = map(int, values)
    if min(rows, columns) <= 0 or expert >= num_experts or row >= rows or column >= columns:
        raise ValueError(f"Invalid Kimi expert tile bounds: {name}")
    return base, expert, num_experts, rows, columns, row, column


async def merge_expert_tiles(weights):
    """Reconstruct one expert, never the global packed parameter.

    Tiles for an expert are contiguous in the stream. Validate coverage without
    allocating a per-element mask, and copy out of reusable transport buckets.
    """
    pending, buffer, covered = None, None, 0
    rectangles = []
    progress = {}
    async for name, tensor in weights:
        tile = parse_expert_tile(name)
        if tile is None:
            if pending is not None:
                raise ValueError(f"Incomplete Kimi expert before {name}: {pending}")
            yield name, tensor
            continue
        base, expert, num_experts, rows, columns, row, column = tile
        key = base, expert, rows, columns
        if tensor.ndim != 2 or tensor.numel() == 0:
            raise ValueError(f"Kimi expert tile must be a nonempty matrix: {name}")
        height, width = tensor.shape
        if row + height > rows or column + width > columns:
            raise ValueError(f"Kimi expert tile exceeds matrix bounds: {name}")
        if pending is None:
            expected, total = progress.get(base, (0, num_experts))
            if expert != expected or total != num_experts:
                raise ValueError(f"Missing, duplicate or inconsistent Kimi expert: {name}")
            progress[base] = expected, total
            pending = key
            buffer = tensor.new_empty((1, rows, columns))
        if key != pending or tensor.dtype != buffer.dtype or tensor.device != buffer.device:
            raise ValueError(f"Inconsistent or incomplete Kimi expert tiles: {name}, pending={pending}")
        for previous_row, previous_column, previous_height, previous_width in rectangles:
            if (
                row < previous_row + previous_height
                and previous_row < row + height
                and column < previous_column + previous_width
                and previous_column < column + width
            ):
                raise ValueError(f"Overlapping Kimi expert tile: {name}")
        buffer[0, row : row + height, column : column + width].copy_(tensor)
        rectangles.append((row, column, height, width))
        covered += tensor.numel()
        if covered == rows * columns:
            yield f"{base}{EXPERT_MARKER}{expert}", buffer
            progress[base] = expert + 1, num_experts
            pending, buffer, covered = None, None, 0
            rectangles = []
    if pending is not None:
        raise ValueError(f"Incomplete Kimi expert at end of stream: {pending}")
    if any(received != total for received, total in progress.values()):
        raise ValueError(f"Missing Kimi experts at end of stream: {progress}")
