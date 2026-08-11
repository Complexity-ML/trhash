from pathlib import Path

from PIL import Image

from trhash import Result, Vision


class PerfectBackend:
    names = ("object",)

    def predict_batch(self, sources, **_options):
        return [
            Result(
                image=Image.open(source).convert("RGB"),
                boxes=[(25.0, 25.0, 75.0, 75.0)],
                scores=[0.9],
                labels=[0],
                names=self.names,
                source=str(source),
            )
            for source in sources
        ]


class OutOfRangeBackend:
    def predict_batch(self, sources, **_options):
        return [
            Result(
                image=Image.open(source).convert("RGB"),
                boxes=[(25.0, 25.0, 75.0, 75.0)],
                scores=[0.9],
                labels=[1],
                names=("0", "unexpected"),
                source=str(source),
            )
            for source in sources
        ]


def _validation_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (100, 100), "white").save(root / "images" / "val" / f"{index}.jpg")
        (root / "labels" / "val" / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    config = root / "dataset.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: object\n"
    )
    return config


def test_val_reports_perfect_map_precision_and_recall(tmp_path: Path):
    model = Vision.__new__(Vision)
    model.backend = PerfectBackend()

    metrics = model.val(data=_validation_dataset(tmp_path), batch=2)

    assert metrics.map50 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.best_confidence == 0.9
    assert metrics.images == 2
    assert metrics.targets == 2
    assert metrics.per_class_ap50 == {"object": 1.0}


def test_val_ignores_out_of_range_remote_labels_without_key_error(tmp_path: Path):
    model = Vision.__new__(Vision)
    model.backend = OutOfRangeBackend()

    metrics = model.val(data=_validation_dataset(tmp_path), batch=2)

    assert metrics.map50 == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.predictions == 2
