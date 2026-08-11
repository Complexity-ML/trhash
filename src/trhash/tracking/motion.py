"""Constant-velocity Kalman filter for bounding boxes."""

from __future__ import annotations

import numpy as np


def _measurement(box) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    return np.asarray(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)),
        dtype=np.float64,
    )


class BoxKalman:
    def __init__(self, box) -> None:
        self.mean = np.concatenate((_measurement(box), np.zeros(4, dtype=np.float64)))
        self.covariance = np.diag((10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0))
        self.transition = np.eye(8, dtype=np.float64)
        self.transition[:4, 4:] = np.eye(4, dtype=np.float64)
        self.observation = np.zeros((4, 8), dtype=np.float64)
        self.observation[:, :4] = np.eye(4, dtype=np.float64)

    @property
    def box(self) -> np.ndarray:
        center_x, center_y, width, height = self.mean[:4]
        width, height = max(width, 1e-3), max(height, 1e-3)
        return np.asarray(
            (
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            ),
            dtype=np.float32,
        )

    def predict(self) -> np.ndarray:
        scale = max(float(self.mean[2]), float(self.mean[3]), 1.0)
        process_noise = np.diag(
            (scale * 0.01,) * 4 + (scale * 0.001,) * 4
        ) ** 2
        self.mean = self.transition @ self.mean
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + process_noise
        )
        return self.box

    def update(self, box) -> np.ndarray:
        measurement = _measurement(box)
        scale = max(float(measurement[2]), float(measurement[3]), 1.0)
        noise = np.diag((scale * 0.02,) * 4) ** 2
        projected = self.observation @ self.covariance @ self.observation.T + noise
        gain = np.linalg.solve(
            projected,
            self.observation @ self.covariance,
        ).T
        innovation = measurement - self.observation @ self.mean
        self.mean += gain @ innovation
        self.covariance = (
            np.eye(8, dtype=np.float64) - gain @ self.observation
        ) @ self.covariance
        return self.box
