"""Dependency-light tests for raw multimodal sidecar round trips."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .media_io import has_media_payload, load_media_ref, save_media_for_key
except ImportError:  # unittest discover with this directory as the top level
    from media_io import has_media_payload, load_media_ref, save_media_for_key


class MediaIOTest(unittest.TestCase):
    def test_nested_bytes_and_tuple_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            data = {
                "images": [b"image-bytes"],
                "videos": [(b"frame-bytes", {"fps": 2})],
                "audios": [],
            }
            reference = save_media_for_key(directory, "sample-0", data)
            restored = load_media_ref(reference)
            self.assertEqual(restored, data)
            self.assertTrue(has_media_payload(restored))

    def test_file_path_is_copied_into_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "image.bin"
            source.write_bytes(b"content")
            reference = save_media_for_key(directory, "sample-1", {"images": [str(source)]})
            restored = load_media_ref(reference)
            copied = Path(restored["images"][0])
            self.assertNotEqual(copied, source)
            self.assertEqual(copied.read_bytes(), b"content")


if __name__ == "__main__":
    unittest.main()
