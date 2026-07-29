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

import re


class ProcessorAdapter:
    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        try:
            processor = object.__getattribute__(self, "_processor")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(processor, name)


class KimiK3Adapter(ProcessorAdapter):
    _MEDIA_PROMPT_PATTERN = re.compile(
        r"<\|media_begin\|>image \d+x\d+<\|media_content\|><\|media_pad\|><\|media_end\|>"
    )
    _VLLM_MEDIA_PROMPT = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"

    @staticmethod
    def _validate_media(video_data, audio_data):
        if video_data is not None or audio_data is not None:
            raise ValueError("KimiK3Processor only supports image inputs")

    def __call__(self, *, text, images=None, videos=None, audio=None, **kwargs):
        self._validate_media(videos, audio)
        if isinstance(text, list | tuple):
            if len(text) != 1:
                raise ValueError("KimiK3Processor only supports one text input at a time")
            text = text[0]

        if images:
            medias = [
                image
                if isinstance(image, dict) and image.get("type") == "image"
                else {"type": "image", "image": image}
                for image in images
            ]
            return self._processor(medias=medias, text=text, **kwargs)

        return self._processor.tokenizer(text=text, **kwargs)

    @classmethod
    def prepare_vllm_prompt_ids(cls, prompt_ids: list[int], tokenizer, image_data) -> list[int]:
        if not image_data:
            return prompt_ids

        prompt = tokenizer.decode(prompt_ids)
        vllm_prompt = cls._MEDIA_PROMPT_PATTERN.sub(cls._VLLM_MEDIA_PROMPT, prompt)
        if vllm_prompt == prompt:
            return prompt_ids
        from verl.utils.tokenizer import normalize_token_ids

        return normalize_token_ids(tokenizer.encode(vllm_prompt))

    @classmethod
    def build_vllm_multimodal_data(cls, image_data=None, video_data=None, audio_data=None) -> dict:
        cls._validate_media(video_data, audio_data)
        if image_data is None:
            return {}
        return {
            "vision_chunk": [
                image
                if isinstance(image, dict) and image.get("type") == "image"
                else {"type": "image", "image": image}
                for image in image_data
            ]
        }


_PROCESSOR_ADAPTERS = {
    "KimiK3Processor": KimiK3Adapter,
}
_VLLM_ADAPTERS = {
    "kimi_k3": KimiK3Adapter,
}


def adapt_processor(processor):
    adapter_cls = _PROCESSOR_ADAPTERS.get(processor.__class__.__name__)
    return processor if adapter_cls is None else adapter_cls(processor)


def get_vllm_adapter(model_type):
    return _VLLM_ADAPTERS.get(model_type)
