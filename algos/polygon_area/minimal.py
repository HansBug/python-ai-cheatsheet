from __future__ import annotations


def signed_polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        # 每一项都是相邻顶点组成的二维叉积。
        total += x1 * y2 - y1 * x2
    return 0.5 * total


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(signed_polygon_area(points))


def main() -> None:
    polygon = [(0.0, 0.0), (4.0, 0.0), (5.0, 2.0), (2.0, 4.0), (0.0, 3.0)]
    polygon_reversed = list(reversed(polygon))

    print("signed area (ccw):", signed_polygon_area(polygon))
    print("signed area (cw):", signed_polygon_area(polygon_reversed))
    print("geometric area:", polygon_area(polygon))


if __name__ == "__main__":
    main()
