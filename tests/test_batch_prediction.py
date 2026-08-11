from pathlib import Path

from PIL import Image

from trhash import Result, Vision


class BatchBackend:
    def __init__(self):
        self.batch_sizes = []

    def predict_batch(self, sources, **_options):
        self.batch_sizes.append(len(sources))
        return [
            Result(
                image=Image.open(source).convert("RGB"),
                boxes=[],
                scores=[],
                labels=[],
                names=("object",),
                source=str(source),
            )
            for source in sources
        ]


def _vision() -> Vision:
    model = Vision.__new__(Vision)
    model.backend = BatchBackend()
    return model


def _images(directory: Path, count: int = 3) -> None:
    directory.mkdir()
    for index in range(count):
        Image.new("RGB", (16, 16), "white").save(directory / f"{index}.jpg")


def test_directory_prediction_uses_real_batches(tmp_path: Path):
    images = tmp_path / "images"
    _images(images)
    model = _vision()

    results = model.predict(images, batch=2)

    assert isinstance(results, list)
    assert len(results) == 3
    assert model.backend.batch_sizes == [2, 1]


def test_stream_prediction_is_lazy_and_single_image_stays_compatible(tmp_path: Path):
    images = tmp_path / "images"
    _images(images)
    model = _vision()

    stream = model.predict(images, batch=2, stream=True)

    assert model.backend.batch_sizes == []
    assert len(list(stream)) == 3
    single = model.predict(images / "0.jpg")
    assert isinstance(single, Result)
