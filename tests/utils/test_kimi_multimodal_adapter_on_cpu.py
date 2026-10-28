# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from verl.utils.model_adapters import KimiK3Adapter, get_vllm_adapter


@pytest.mark.parametrize("text", ["<|kimi_image_placeholder|>", ["<|kimi_image_placeholder|>"]])
def test_kimi_processor_preserves_text_batch_for_image_placeholders(text):
    image = Image.new("RGB", (14, 14))

    def process_images(*, medias, text, **kwargs):
        assert sum(sample.count("<|kimi_image_placeholder|>") for sample in text) == len(medias)
        return {"input_ids": [[1, 2]], "pixel_values": medias}

    processor = Mock(side_effect=process_images)
    adapter = KimiK3Adapter(processor)
    inputs = adapter(text=text, images=[image], return_tensors="pt")
    assert inputs["input_ids"] == [[1, 2]]
    assert inputs["pixel_values"] == [{"type": "image", "image": image}]


def test_kimi_images_use_standard_image_modality():
    image = Image.new("RGB", (14, 14))
    wrapped = {"type": "image", "image": image}
    adapter = get_vllm_adapter("kimi_k3")
    payload = adapter.build_vllm_multimodal_data([image, wrapped])
    assert set(payload) == {"image"}
    assert payload["image"] == [image, image]
    assert payload["image"][1] is image
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
        "before <|kimi_image_placeholder|> after", add_special_tokens=False
    )


@pytest.mark.parametrize("count", [1, 2])
def test_kimi_generated_sample_can_rebuild_image_inputs(count):
    from verl.utils.model_adapters import decode_multimodal_processor_text

    media = "<|media_begin|>image 28x14<|media_content|><|media_pad|><|media_end|>"
    decoded = "<|im_user|>" + media * count + "question<|im_assistant|>answer<|im_end|>"
    tokenizer = SimpleNamespace(decode=Mock(return_value=decoded))
    images = [Image.new("RGB", (28, 14)) for _ in range(count)]

    def process_images(*, medias, text, **kwargs):
        assert text[0].count("<|kimi_image_placeholder|>") == len(medias) == count
        assert "<|media_begin|>" not in text[0]
        assert text[0].endswith("answer<|im_end|>")
        return {"pixel_values": medias}

    adapter = KimiK3Adapter(Mock(side_effect=process_images))
    text = decode_multimodal_processor_text(adapter, tokenizer, [1, 2, 3])
    tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=False)
    assert adapter(text=[text], images=images)["pixel_values"] == [
        {"type": "image", "image": image} for image in images
    ]


def test_other_processors_keep_special_token_stripping():
    from verl.utils.model_adapters import decode_multimodal_processor_text

    tokenizer = SimpleNamespace(decode=Mock(return_value="question answer"))
    assert decode_multimodal_processor_text(object(), tokenizer, [1, 2]) == "question answer"
    tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=True)
