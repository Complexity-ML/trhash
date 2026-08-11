# TR-Hash Vision

The small public SDK for predicting, fine-tuning, and serving AETHORIA AI
TR-Hash Vision models. It deliberately exposes a model-centric API instead of
the research framework internals.

## Python

Install the local runtime (PyPI publication will come later):

```bash
pip install -e ".[local,serve]"
```

```python
from trhash import Vision

model = Vision("AETHORIA-AI/TR-HASH-Vision-0.8M-VOC")

result = model.predict("image.jpg", confidence=0.25)
result.save("prediction.jpg")

checkpoint = model.train(
    data="dataset.yaml",
    epochs=20,
    batch=64,
    device="cuda",
)
```

`train()` transfers the complete detector: vision tower, feature pyramid,
hidden detection heads, box regression, and objectness. Class rows with the
same name are copied automatically; new classes are initialized and trained.

## CLI

The CLI follows the familiar `key=value` model workflow:

```bash
trhash predict model=AETHORIA-AI/TR-HASH-Vision-0.8M-VOC source=image.jpg save=prediction.jpg

trhash train model=AETHORIA-AI/TR-HASH-Vision-0.8M-VOC \
  data=dataset.yaml epochs=20 batch=64 device=cuda

trhash serve model=AETHORIA-AI/TR-HASH-Vision-0.8M-VOC host=0.0.0.0 port=8000
```

## YOLO dataset contract

```yaml
path: /data/animals
train: images/train
val: images/val
names:
  0: cat
  1: dog
```

Labels use normalized YOLO rows: `class_id cx cy width height`. By default,
`labels/train` and `labels/val` are inferred from the corresponding `images`
directories.

## Remote endpoint

The core SDK can call a running TR-Hash endpoint without loading PyTorch:

```python
model = Vision(
    "AETHORIA-AI/TR-HASH-Vision-0.8M-VOC",
    endpoint="https://vision.example.com",
    api_key="...",
)
result = model("image.jpg")
```

The endpoint is bound to one deployed model. Local training and serving remain
optional extras, keeping remote clients small.
