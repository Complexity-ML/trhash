from pathlib import Path

import numpy as np
from PIL import Image

from trhash.decoding import decode
from trhash.metadata import ModelMetadata
from trhash.preprocessing import preprocess, restore_boxes


def _metadata(**overrides) -> ModelMetadata:
    values = {
        "format_version": 2,
        "task": "detection",
        "model_file": "model.onnx",
        "image_size": 32,
        "num_classes": 2,
        "class_names": ("cat", "dog"),
        "grid_sizes": (1,),
        "reg_max": 0,
        "box_encoding": "stride_ltrb_dfl",
        "score_encoding": "quality_class_sigmoid",
        "recommended_confidence": 0.25,
    }
    values.update(overrides)
    return ModelMetadata(**values)


def test_metadata_round_trip(tmp_path: Path):
    metadata = _metadata()

    metadata.save(tmp_path)

    assert ModelMetadata.load(tmp_path) == metadata


def test_numpy_decode_quality_ltrb_prediction():
    distance = np.log(np.expm1(0.25))
    raw = np.array([[distance, distance, distance, distance, -2.0, 4.0]], dtype=np.float32)

    boxes, scores, labels = decode(raw, _metadata(), confidence=0.2, iou=0.45)

    np.testing.assert_allclose(boxes, [[0.25, 0.25, 0.75, 0.75]], atol=1e-6)
    assert scores[0] > 0.98
    assert labels.tolist() == [1]


def test_numpy_decode_dfl_prediction():
    metadata = _metadata(reg_max=2)
    side = np.array([-8.0, 8.0, -8.0], dtype=np.float32)
    raw = np.concatenate((np.tile(side, 4), np.array([-2.0, 4.0], dtype=np.float32)))[None]

    boxes, scores, labels = decode(raw, metadata, confidence=0.2, iou=0.45)

    np.testing.assert_allclose(boxes, [[0.0, 0.0, 1.0, 1.0]], atol=1e-3)
    assert scores[0] > 0.98
    assert labels.tolist() == [1]


def test_numpy_decode_returns_typed_empty_detections():
    raw = np.full((1, 6), -8.0, dtype=np.float32)

    boxes, scores, labels = decode(raw, _metadata(), confidence=0.99, iou=0.45)

    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


def test_letterbox_restore_round_trip():
    metadata = _metadata()
    pixels, geometry = preprocess(Image.new("RGB", (80, 40), "white"), metadata)
    letterboxed_box = np.array([[0.25, 0.375, 0.75, 0.625]], dtype=np.float32)

    restored = restore_boxes(letterboxed_box, metadata, geometry)

    assert pixels.shape == (3, 32, 32)
    np.testing.assert_allclose(restored, [[20, 10, 60, 30]], atol=1e-6)
