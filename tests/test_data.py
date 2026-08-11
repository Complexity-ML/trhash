from pathlib import Path

from trhash.data import load_dataset, observed_class_count


def test_load_ultralytics_style_dataset(tmp_path: Path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "labels" / "train" / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
    )
    config = tmp_path / "data.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n"
    )

    dataset = load_dataset(config)

    assert dataset.names == ("cat", "dog")
    assert dataset.train_labels == tmp_path / "labels" / "train"
    assert observed_class_count(dataset.train_labels) == 2
