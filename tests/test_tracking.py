from PIL import Image

from trhash import ByteTracker, Result, Vision


def test_bytetrack_keeps_id_through_low_confidence_detection():
    tracker = ByteTracker(
        high_threshold=0.5,
        low_threshold=0.1,
        match_iou_threshold=0.3,
        second_match_iou_threshold=0.2,
    )

    first = tracker.update([(10, 10, 30, 30)], [0.9], [0])
    recovered = tracker.update([(11, 10, 31, 30)], [0.2], [0])
    third = tracker.update([(12, 10, 32, 30)], [0.8], [0])

    assert first == [1]
    assert recovered == [1]
    assert third == [1]


def test_bytetrack_is_class_aware():
    tracker = ByteTracker()

    assert tracker.update([(10, 10, 30, 30)], [0.9], [0]) == [1]
    assert tracker.update([(10, 10, 30, 30)], [0.9], [1]) == [2]


class SequenceBackend:
    def __init__(self):
        self.index = 0

    def predict_batch(self, sources, **_options):
        scores = (0.9, 0.2, 0.8)
        results = []
        for image in sources:
            results.append(
                Result(
                    image=image.copy(),
                    boxes=[(10 + self.index, 10, 30 + self.index, 30)],
                    scores=[scores[self.index]],
                    labels=[0],
                    names=("object",),
                )
            )
            self.index += 1
        return results


def test_vision_track_returns_ids_aligned_with_detections():
    model = Vision.__new__(Vision)
    model.backend = SequenceBackend()
    frames = [Image.new("RGB", (48, 48), "white") for _ in range(3)]

    results = model.track(frames, stream=False)

    assert [result.track_ids for result in results] == [[1], [1], [1]]
    assert results[0].to_dict()["detections"][0]["track_id"] == 1
