from __future__ import annotations

import numpy as np


def points_to_bev(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    x_min, x_max = x_range
    y_min, y_max = y_range

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )
    points = points[mask]

    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))
    grid = np.zeros((height, width), dtype=np.int32)

    x_idx = np.floor((x_max - points[:, 0]) / resolution).astype(np.int32)
    y_idx = np.floor((points[:, 1] - y_min) / resolution).astype(np.int32)

    valid = (
        (x_idx >= 0)
        & (x_idx < height)
        & (y_idx >= 0)
        & (y_idx < width)
    )
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]

    for i, j in zip(x_idx, y_idx):
        grid[i, j] += 1

    return grid


def make_box_points(
    x_center: float,
    y_center: float,
    length: float,
    width: float,
    step: float = 0.5,
) -> np.ndarray:
    xs = np.arange(x_center - length / 2, x_center + length / 2 + 1e-6, step)
    ys = np.arange(y_center - width / 2, y_center + width / 2 + 1e-6, step)

    points = []
    for x in xs:
        for y in ys:
            points.append([x, y, 0.0])
    return np.asarray(points, dtype=np.float64)


def grid_to_ascii(grid: np.ndarray, threshold: int = 1) -> str:
    chars = []
    for row in grid:
        line = "".join("#" if value >= threshold else "." for value in row)
        chars.append(line)
    return "\n".join(chars)


def main() -> None:
    np.set_printoptions(precision=2, suppress=True)

    lane_left = np.array([[x, 1.75, 0.0] for x in np.arange(0.0, 20.0, 0.5)])
    lane_right = np.array([[x, -1.75, 0.0] for x in np.arange(0.0, 20.0, 0.5)])
    car_a = make_box_points(x_center=12.0, y_center=0.0, length=4.0, width=2.0)
    car_b = make_box_points(x_center=18.0, y_center=-3.5, length=4.5, width=2.0)

    points = np.concatenate([lane_left, lane_right, car_a, car_b], axis=0)
    bev = points_to_bev(
        points=points,
        x_range=(0.0, 24.0),
        y_range=(-8.0, 8.0),
        resolution=0.5,
    )

    print("Input point cloud shape:", points.shape)
    print("Non-empty BEV cells:", int((bev > 0).sum()))
    print("\nASCII BEV map (# means occupied):")
    print(grid_to_ascii(bev))


if __name__ == "__main__":
    main()
