from __future__ import annotations

import numpy as np


def build_ego_to_camera(camera_height: float = 1.5) -> np.ndarray:
    """Convert ego coordinates (x forward, y left, z up) to camera coordinates
    (x right, y down, z forward) for a front camera aligned with the ego frame.
    """
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, camera_height],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


def project_points(
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    ego_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_ego_h = to_homogeneous(points_ego)
    points_camera_h = (ego_to_camera @ points_ego_h.T).T
    points_camera = points_camera_h[:, :3]

    valid = points_camera[:, 2] > 1e-6
    pixels = np.full((points_ego.shape[0], 2), np.nan, dtype=np.float64)

    projected = (intrinsic @ points_camera[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, points_camera


def back_project_with_depth(
    pixels: np.ndarray,
    depth_z: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x = (pixels[:, 0] - cx) * depth_z / fx
    y = (pixels[:, 1] - cy) * depth_z / fy
    z = depth_z
    return np.stack([x, y, z], axis=1)


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    intrinsic = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ego_to_camera = build_ego_to_camera(camera_height=1.5)

    points_ego = np.array(
        [
            [10.0, 0.0, 0.0],   # lane point 10m ahead
            [20.0, 0.0, 0.0],   # lane point 20m ahead
            [15.0, -2.0, 0.0],  # object on the right side
            [15.0, 2.0, 0.0],   # object on the left side
            [25.0, 0.0, 1.5],   # point at camera height
        ],
        dtype=np.float64,
    )

    pixels, points_camera = project_points(points_ego, intrinsic, ego_to_camera)

    print("Intrinsic matrix K:")
    print(intrinsic)
    print("\nEgo points [x_forward, y_left, z_up]:")
    print(points_ego)
    print("\nCamera points [x_right, y_down, z_forward]:")
    print(points_camera)
    print("\nProjected pixels [u, v]:")
    print(pixels)

    valid = np.isfinite(pixels[:, 0])
    reconstructed = back_project_with_depth(
        pixels[valid],
        points_camera[valid, 2],
        intrinsic,
    )
    print("\nBack-projected camera points with known depth:")
    print(reconstructed)


if __name__ == "__main__":
    main()
