# TR-Hash Vision

The small public SDK for predicting, fine-tuning, and serving AETHORIA AI
TR-Hash Vision models. It deliberately exposes a model-centric API instead of
the research framework internals.

## Python

Install the local runtime (PyPI publication will come later):

```bash
pip install -e ".[local,export,serve]"
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

## Export and production serving

The research framework is used to train and export a checkpoint. Production
inference uses the generated ONNX bundle and does not import PyTorch or
`complexity-framework`.

```bash
trhash export \
  model=artifacts/detector/best \
  runtime=torch \
  device=cpu \
  output=artifacts/trhash-onnx

trhash serve \
  model=artifacts/trhash-onnx \
  runtime=onnx \
  host=0.0.0.0 \
  port=8000
```

The portable bundle contains:

- `model.onnx`: fixed-resolution inference graph with dynamic batching;
- `trhash.json`: classes, feature-grid geometry, preprocessing, calibrated
  confidence, and NMS-free metadata.

Publish it beside the training artifacts on Hugging Face:

```bash
trhash publish \
  bundle=artifacts/trhash-onnx \
  repo=AETHORIA-AI/TR-HASH-Vision-0.8M-VOC \
  private=false
```

The server can then load the repository directly with
`TRHASH_MODEL=AETHORIA-AI/TR-HASH-Vision-0.8M-VOC`.

The server exposes `GET /health`, authenticated `GET /v1/model`, and
authenticated `POST /v1/predict`. Set `TR_HASH_API_KEY` to require
`X-API-Key`.

```bash
curl -H 'X-API-Key: secret' -F file=@image.jpg \
  'http://127.0.0.1:8000/v1/predict?confidence=0.25'
```

Container deployment:

```bash
docker build -t trhash-server .
docker run --rm -p 8000:8000 \
  -e TRHASH_MODEL=/models/trhash \
  -e TR_HASH_API_KEY=secret \
  -v "$PWD/artifacts/trhash-onnx:/models/trhash:ro" \
  trhash-server
```

For NVIDIA production at larger scale, the same ONNX graph can be placed
behind Triton/TensorRT for dynamic batching and GPU scheduling while clients
continue to use the same `Vision(endpoint=...)` API.

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
