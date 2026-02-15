"""Support utilities used by the Difix integration."""

from __future__ import annotations

import numpy as np


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a xyzw quaternion."""

    m = mat[:3, :3]
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


class CameraPoseInterpolator:
    """Light-weight helper to match poses for Difix runs."""

    def __init__(self, rotation_weight: float = 1.0, translation_weight: float = 1.0) -> None:
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def _pose_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)

        q1 = _mat_to_quat(pose1)
        q2 = _mat_to_quat(pose2)
        if np.dot(q1, q2) < 0:
            q2 = -q2
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        rotation_dist = 2.0 * np.arccos(dot)

        return self.translation_weight * translation_dist + self.rotation_weight * rotation_dist

    def find_nearest_assignments(self, training_poses: np.ndarray, testing_poses: np.ndarray) -> np.ndarray:
        assignments = []
        for pose in testing_poses:
            distances = [self._pose_distance(train_pose, pose) for train_pose in training_poses]
            assignments.append(int(np.argmin(distances)))
        return np.asarray(assignments, dtype=np.int64)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8 or abs(angle) < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis /= norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def sample_nearby_pose(
    pose: np.ndarray,
    scene_scale: float,
    translation_jitter: float,
    rotation_jitter_deg: float,
) -> np.ndarray:
    pose = pose.astype(np.float64)
    new_pose = pose.copy()

    trans_noise = np.random.normal(scale=translation_jitter * scene_scale, size=3)
    new_pose[:3, 3] = pose[:3, 3] + trans_noise

    axis = np.random.normal(size=3)
    angle = np.deg2rad(rotation_jitter_deg) * np.random.normal()
    rot_noise = _axis_angle_to_matrix(axis, angle)
    new_pose[:3, :3] = rot_noise @ pose[:3, :3]

    return new_pose.astype(np.float32)


__all__ = ["CameraPoseInterpolator", "sample_nearby_pose"]
