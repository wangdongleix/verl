# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from verl.utils.model_adapters import get_vllm_adapter


def test_kimi_images_use_typed_vision_chunks():
    image = Image.new("RGB", (14, 14))
    wrapped = {"type": "image", "image": image}
    adapter = get_vllm_adapter("kimi_k3")
    payload = adapter.build_vllm_multimodal_data([image, wrapped])
    assert set(payload) == {"vision_chunk"}
    assert payload["vision_chunk"] == [wrapped, wrapped]
    assert payload["vision_chunk"][1] is wrapped
    assert adapter.build_vllm_multimodal_data() == {}
    with pytest.raises(ValueError, match="only supports image"):
        adapter.build_vllm_multimodal_data(video_data=[object()])


def test_kimi_prompt_normalization_matches_vllm_image_replacement_target():
    adapter = get_vllm_adapter("kimi_k3")
    tokenizer = SimpleNamespace(
        decode=Mock(
            return_value="before <|media_begin|>image 448x448<|media_content|><|media_pad|><|media_end|> after"
        ),
        encode=Mock(return_value=[101, 102]),
    )
    assert adapter.prepare_vllm_prompt_ids([1, 2], tokenizer, [object()]) == [101, 102]
    tokenizer.encode.assert_called_once_with(
        "before <|media_begin|>image<|media_content|><|media_pad|><|media_end|> after"
    )
