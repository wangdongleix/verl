from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.model_forward import _kimi_k3_dynamic_multimodal_length


def _config(**overrides):
    values = {
        "seq_length": 4096,
        "sequence_parallel": True,
        "tensor_model_parallel_size": 4,
        "kimi_multimodal_length_alignment": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dynamic_multimodal_length_uses_bounded_aligned_bucket():
    indices = torch.tensor([[0, 1025]])

    assert _kimi_k3_dynamic_multimodal_length(indices, _config()) == 1280


def test_dynamic_multimodal_length_rejects_invalid_bucket():
    with pytest.raises(ValueError, match="must be positive"):
        _kimi_k3_dynamic_multimodal_length(
            torch.tensor([[0]]),
            _config(kimi_multimodal_length_alignment=0),
        )

    with pytest.raises(ValueError, match="exceeds the configured limit"):
        _kimi_k3_dynamic_multimodal_length(
            torch.tensor([[4095]]),
            _config(kimi_multimodal_length_alignment=257),
        )


def test_kimi_forward_accepts_current_engine_options_and_preserves_jagged_rows():
    from verl.models.mcore.model_forward import kimi_k3_forward_model_engine

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_process = True
            self.config = _config(
                model_type="kimi_k3",
                media_placeholder_token_id=99,
                context_parallel_size=1,
                sequence_parallel=False,
            )

        def forward(self, *, input_ids, attention_mask, position_ids, layout_padding_mask):
            assert attention_mask is None and position_ids is None
            torch.testing.assert_close(layout_padding_mask, torch.tensor([[True, True, False], [True, True, True]]))
            return input_ids.float().unsqueeze(-1).expand(-1, -1, 2)

    ids = torch.nested.as_nested_tensor([torch.tensor([2, 3]), torch.tensor([4, 5, 6])], layout=torch.jagged)
    temperatures = torch.nested.as_nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    output = kimi_k3_forward_model_engine(
        Model(),
        ids,
        {},
        pad_token_id=0,
        data_format="bshd",
        logits_processor=lambda logits, **kwargs: {"log_probs": logits[..., 0]},
        logits_processor_args={"label": ids, "temperature": temperatures},
        router_padding_mask=None,
        mtp_loss_normalization_factor=None,
        pad_to_length_bucket=None,
        local_cp_size=None,
        forced_max_seqlen=None,
    )
    torch.testing.assert_close(output["log_probs"].values(), ids.values().float())
    torch.testing.assert_close(output["log_probs"].offsets(), ids.offsets())


@pytest.mark.parametrize("option", ["router_padding_mask", "mtp_loss_normalization_factor", "pad_to_length_bucket"])
def test_kimi_forward_rejects_unsupported_engine_modes(option):
    from verl.models.mcore.model_forward import kimi_k3_forward_model_engine

    with pytest.raises(NotImplementedError):
        kimi_k3_forward_model_engine(None, None, {}, **{option: 1})


def test_kimi_dynamic_image_forward_restores_raw_positions_and_prepares_replay():
    from unittest.mock import Mock

    from verl.models.mcore.model_forward import kimi_k3_forward_model_engine

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_process = True
            self.config = _config(
                model_type="kimi_k3",
                media_placeholder_token_id=99,
                context_parallel_size=1,
                sequence_parallel=False,
                seq_length=16,
                kimi_dynamic_multimodal_length=True,
                kimi_multimodal_length_alignment=8,
                vision_config=SimpleNamespace(merge_type="sd2_tpool", merge_kernel_size=[2, 2]),
            )

        def forward(self, *, input_ids, multimodal_sequence_length, grid_thws, **kwargs):
            assert multimodal_sequence_length == 8
            torch.testing.assert_close(grid_thws, torch.tensor([[1, 4, 4]]))
            return torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

    ids = torch.nested.as_nested_tensor([torch.tensor([10, 99, 11])], layout=torch.jagged)
    temperatures = torch.nested.as_nested_tensor([torch.ones(3)], layout=torch.jagged)
    replay = Mock()
    output = kimi_k3_forward_model_engine(
        Model(),
        ids,
        {"grid_thws": torch.tensor([[1, 4, 4]]), "pixel_values": torch.zeros(16, 3)},
        pad_token_id=0,
        router_replay_prepare=replay,
        logits_processor=lambda logits, **kwargs: {"log_probs": logits[..., 0]},
        logits_processor_args={"label": ids, "temperature": temperatures},
    )
    replay.assert_called_once_with(max_model_rows=8)
    torch.testing.assert_close(output["log_probs"].values(), torch.tensor([0.0, 4.0, 5.0]))
