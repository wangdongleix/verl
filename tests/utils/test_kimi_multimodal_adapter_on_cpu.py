# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from verl.utils.model_adapters import get_vllm_adapter


@pytest.mark.parametrize("modality", ["image", "vision_chunk"])
def test_kimi_images_match_registered_modality(modality):
    image = Image.new("RGB", (14, 14))
    wrapped = {"type": "image", "image": image}
    adapter = get_vllm_adapter("kimi_k3", supported_mm_limits={modality: None})
    payload = adapter.build_vllm_multimodal_data([image, wrapped])
    assert set(payload) == {modality}
    expected = image if modality == "image" else wrapped
    assert payload[modality] == [expected, expected]
    assert payload[modality][1] is expected
    assert adapter.build_vllm_multimodal_data() == {}
    with pytest.raises(ValueError, match="only supports image"):
        adapter.build_vllm_multimodal_data(video_data=[object()])


@pytest.mark.parametrize(
    ("modality", "placeholder"),
    [
        ("image", "<|kimi_image_placeholder|>"),
        ("vision_chunk", "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"),
    ],
)
def test_kimi_prompt_normalization_matches_vllm_replacement_target(modality, placeholder):
    adapter = get_vllm_adapter("kimi_k3", supported_mm_limits={modality: None})
    tokenizer = SimpleNamespace(
        decode=Mock(
            return_value="before <|media_begin|>image 448x448<|media_content|><|media_pad|><|media_end|> after"
        ),
        encode=Mock(return_value=[101, 102]),
    )
    assert adapter.prepare_vllm_prompt_ids([1, 2], tokenizer, [object()]) == [101, 102]
    tokenizer.encode.assert_called_once_with(f"before {placeholder} after", add_special_tokens=False)


def test_kimi_rejects_unsupported_registered_modalities():
    with pytest.raises(ValueError, match="Unsupported Kimi K3 rollout modalities"):
        get_vllm_adapter("kimi_k3", supported_mm_limits={"video": None})
