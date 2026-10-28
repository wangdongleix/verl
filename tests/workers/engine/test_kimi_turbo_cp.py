"""Compare Turbo CP token outputs and averaged parameter gradients with CP=1."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl.utils.model_adapters import kimi_k3_logits_to_input_indices
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def check_case(raw_rows, grids, temperatures, cp_size, *, force_fp32=False, checkpoint_entropy=False):
    rank = dist.get_rank()
    torch.manual_seed(42)
    config = SimpleNamespace(
        model_type="kimi_k3",
        media_placeholder_token_id=31,
        pad_token_id=0,
        vision_config=SimpleNamespace(merge_type="sd2_tpool", merge_kernel_size=[2, 2]),
    )
    ids = torch.nested.as_nested_tensor([torch.tensor(row) for row in raw_rows], layout=torch.jagged)
    dense_ids = torch.nested.to_padded_tensor(ids, 0)
    indices = kimi_k3_logits_to_input_indices(dense_ids, {"grid_thws": torch.tensor(grids)}, config)
    output_indices = indices
    if indices is None:
        indices = torch.arange(dense_ids.shape[1]).expand(len(raw_rows), -1)
    expanded_length = int(indices.max()) + 1
    padded_length = (expanded_length + cp_size - 1) // cp_size * cp_size
    hidden = torch.randn(len(raw_rows), padded_length, 7, dtype=torch.float64)
    weight = torch.randn(7, 32, dtype=torch.float64, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    chunk = padded_length // cp_size
    local_logits = hidden[:, rank * chunk : (rank + 1) * chunk] @ weight
    engine = SimpleNamespace(
        _is_ulysses_enabled=lambda: True,
        model_config=SimpleNamespace(hf_config=config),
        _parallel_state=SimpleNamespace(get_ulysses_rank=lambda: rank),
        ulysses_parallel_group=dist.group.WORLD,
        engine_config=SimpleNamespace(entropy_checkpointing=checkpoint_entropy),
    )
    data = TensorDict({"input_ids": ids}, batch_size=[len(raw_rows)])
    tu.assign_non_tensor(data, use_remove_padding=False, calculate_entropy=True, calculate_sum_pi_squared=True)
    labels = ids.values().roll(-1)
    args = {
        "logits_to_input_indices": output_indices,
        "input_ids_rmpad_rolled": labels,
        "temperature": torch.tensor(temperatures, dtype=torch.float64),
        "temperature_is_one": all(value == 1 for value in temperatures),
    }
    result = FSDPTurboEngineWithLMHead.prepare_model_outputs(
        engine, SimpleNamespace(logits=local_logits), args, data, None if force_fp32 else object()
    )
    full_logits = hidden @ reference_weight
    aligned = full_logits[torch.arange(len(raw_rows))[:, None], indices]
    aligned = aligned / args["temperature"][:, None, None]
    valid = torch.arange(indices.shape[1])[None, :] < ids.offsets().diff()[:, None]
    logits = aligned[valid]
    probs = logits.softmax(-1)
    expected = {
        "log_probs": (logits.float() if force_fp32 else logits).log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1),
        "entropy": -(probs * logits.log_softmax(-1)).sum(-1),
        "sum_pi_squared": probs.square().sum(-1),
    }
    coefficients = torch.linspace(0.1, 1.1, len(labels), dtype=torch.float64)
    loss, reference_loss = 0, 0
    tolerance = 1e-6 if force_fp32 else 1e-10
    for name in expected:
        torch.testing.assert_close(
            result[name].values(), expected[name].to(result[name].dtype), atol=tolerance, rtol=tolerance
        )
        loss = loss + (result[name].values() * coefficients).sum()
        reference_loss = reference_loss + (expected[name] * coefficients).sum()
    loss.backward()
    reference_loss.backward()
    # FSDP's parameter-gradient averaging includes the CP replicas.
    dist.all_reduce(weight.grad)
    weight.grad /= cp_size
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=tolerance * 10, rtol=tolerance * 10)
    torch.testing.assert_close(
        weight - 0.01 * weight.grad,
        reference_weight - 0.01 * reference_weight.grad,
        atol=tolerance * 10,
        rtol=tolerance * 10,
    )


def check_replay(raw_rows, grids, cp_size):
    config = SimpleNamespace(
        model_type="kimi_k3",
        media_placeholder_token_id=31,
        pad_token_id=0,
        vision_config=SimpleNamespace(merge_type="sd2_tpool", merge_kernel_size=[2, 2]),
    )
    raw_width = max(map(len, raw_rows))
    ids = torch.zeros(len(raw_rows), raw_width, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    lengths = []
    image = iter(grids)
    for row, tokens in enumerate(raw_rows):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = True
        length = len(tokens)
        for token in tokens:
            if token == 31:
                _, height, width = next(image)
                length += height * width // 4 - 1
        lengths.append(length)
    fused_width = max(length + raw_width - len(tokens) for length, tokens in zip(lengths, raw_rows, strict=True))
    route_width = max(lengths) - 1
    routes = torch.arange(len(raw_rows) * route_width * 4).reshape(len(raw_rows), route_width, 2, 2)
    weights = routes.float() / 100
    replay_mask = torch.arange(route_width)[None, :] < torch.tensor(lengths)[:, None] - 1
    inputs = dict(
        input_ids=ids,
        attention_mask=mask,
        grid_thws=torch.tensor(grids),
        routed_experts=routes,
        routed_expert_weights=weights,
        router_replay_mask=replay_mask,
    )
    args = {"logits_to_input_indices": kimi_k3_logits_to_input_indices(ids, inputs, config)}
    engine = object.__new__(FSDPTurboEngineWithLMHead)
    engine._is_ulysses_enabled = lambda: True
    engine.model_config = SimpleNamespace(hf_config=config)
    engine.ulysses_parallel_group = dist.group.WORLD
    padded_width = (fused_width + cp_size - 1) // cp_size * cp_size
    expected_ids = torch.zeros(len(raw_rows), padded_width, 2, 2, dtype=torch.long)
    expected_weights = torch.zeros_like(expected_ids, dtype=torch.float32)
    expected_mask = torch.zeros(len(raw_rows), padded_width, dtype=torch.bool)
    left_padding = all(len(tokens) == raw_width for tokens in raw_rows) and bool(grids)
    for row, length in enumerate(lengths):
        start = fused_width - length if left_padding else 0
        expected_ids[row, start : start + length - 1] = routes[row, : length - 1]
        expected_weights[row, start : start + length - 1] = weights[row, : length - 1]
        expected_mask[row, start : start + length - 1] = True
    with patch.object(FSDPEngineWithLMHead, "prepare_model_inputs", return_value=(inputs, args)):
        actual, _ = engine.prepare_model_inputs(None)
    for key, expected in zip(
        ("routed_experts", "routed_expert_weights", "router_replay_mask"),
        (expected_ids, expected_weights, expected_mask),
        strict=True,
    ):
        torch.testing.assert_close(actual[key], expected.chunk(cp_size, dim=1)[dist.get_rank()])


def check_attention_padding(cp_size):
    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import _full_causal_mask_from_local

    local_length = 5
    full_length = local_length * cp_size
    positions = torch.arange(full_length)
    valid_keys = torch.stack((positions >= 3, positions < full_length - 3))
    local_keys = valid_keys.chunk(cp_size, dim=-1)[dist.get_rank()]
    local_causal = torch.ones(local_length, local_length, dtype=torch.bool).tril()
    local_mask = local_causal[None, None] & local_keys[:, None, None, :]
    query = torch.zeros(2, 4, local_length, 2)
    expected = torch.ones(full_length, full_length, dtype=torch.bool).tril()[None, None]
    expected = expected & valid_keys[:, None, None, :]
    for mask in (local_mask, torch.zeros_like(local_mask, dtype=torch.float32).masked_fill(~local_mask, -float("inf"))):
        actual = _full_causal_mask_from_local(mask, query, dist.group.WORLD, seq_dim=2)
        torch.testing.assert_close(actual == 0, expected)


def main():
    # A torch_npu installation sets the checkpoint default to NPU even for CPU tensors.
    torch.utils.checkpoint.DefaultDeviceType.set_device_type("cpu")
    dist.init_process_group("gloo")
    cp_size = dist.get_world_size()
    check_attention_padding(cp_size)
    check_case([[2, 31, 4, 5, 6], [7, 31, 8]], [[1, 4, 6], [1, 4, 4]], [0.8, 1.3], cp_size)
    check_case([[31, 9]], [[1, 8, 8]], [1.0], cp_size)
    check_case([[2, 31, 3, 31, 4], [5, 31, 6, 7, 8]], [[1, 4, 4], [1, 4, 6], [1, 4, 4]], [1.0, 1.0], cp_size)
    check_case([[2, 3, 4], [5]], [], [0.8, 1.0], cp_size, force_fp32=True, checkpoint_entropy=True)
    check_replay([[2, 31, 4, 5, 6], [7, 31, 8]], [[1, 4, 6], [1, 4, 4]], cp_size)
    check_replay([[31, 9], [31, 8]], [[1, 8, 8], [1, 4, 4]], cp_size)
    check_replay([[2, 3, 4], [5]], [], cp_size)
    if dist.get_rank() == 0:
        print(
            f"PASS CP={cp_size}: uneven lengths, multi-image, empty ownership, "
            "outputs, gradients, SGD update, R3 route alignment"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
