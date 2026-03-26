from __future__ import annotations

from math import hypot

Point = tuple[float, float]


def subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def signed_polygon_area(points: list[Point]) -> float:
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - y1 * x2
    return 0.5 * total


def ensure_ccw(points: list[Point]) -> list[Point]:
    if len(points) < 3:
        return points[:]
    if signed_polygon_area(points) < 0:
        return list(reversed(points))
    return points[:]


def project_polygon(axis: Point, polygon: list[Point]) -> tuple[float, float]:
    projections = [dot(axis, point) for point in polygon]
    return min(projections), max(projections)


def has_separating_axis(axis: Point, poly_a: list[Point], poly_b: list[Point], eps: float = 1e-9) -> bool:
    min_a, max_a = project_polygon(axis, poly_a)
    min_b, max_b = project_polygon(axis, poly_b)
    return max_a < min_b - eps or max_b < min_a - eps


def convex_polygons_intersect(poly_a: list[Point], poly_b: list[Point], eps: float = 1e-9) -> bool:
    poly_a = ensure_ccw(poly_a)
    poly_b = ensure_ccw(poly_b)

    for polygon in (poly_a, poly_b):
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            edge = subtract(p2, p1)
            axis = (-edge[1], edge[0])
            if has_separating_axis(axis, poly_a, poly_b, eps):
                return False
    return True


def inside_half_plane(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    return cross(subtract(b, a), subtract(p, a)) >= -eps


def line_intersection(p1: Point, p2: Point, q1: Point, q2: Point, eps: float = 1e-9) -> Point:
    r = subtract(p2, p1)
    s = subtract(q2, q1)
    denom = cross(r, s)
    if abs(denom) < eps:
        # 最小实现主要处理一般位置；平行退化时直接回退到当前点。
        return p2

    t = cross(subtract(q1, p1), s) / denom
    return (p1[0] + t * r[0], p1[1] + t * r[1])


def cleanup_polygon(points: list[Point], eps: float = 1e-9) -> list[Point]:
    if not points:
        return []

    cleaned: list[Point] = []
    for point in points:
        if not cleaned:
            cleaned.append(point)
            continue

        if hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > eps:
            cleaned.append(point)

    if len(cleaned) > 1 and hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= eps:
        cleaned.pop()

    return cleaned


def clip_with_half_plane(subject: list[Point], a: Point, b: Point, eps: float = 1e-9) -> list[Point]:
    if not subject:
        return []

    output: list[Point] = []
    prev = subject[-1]
    prev_inside = inside_half_plane(a, b, prev, eps)

    for curr in subject:
        curr_inside = inside_half_plane(a, b, curr, eps)

        if prev_inside and curr_inside:
            output.append(curr)
        elif prev_inside and not curr_inside:
            output.append(line_intersection(prev, curr, a, b, eps))
        elif not prev_inside and curr_inside:
            output.append(line_intersection(prev, curr, a, b, eps))
            output.append(curr)

        prev = curr
        prev_inside = curr_inside

    return cleanup_polygon(output, eps)


def convex_polygon_intersection(subject: list[Point], clip: list[Point], eps: float = 1e-9) -> list[Point]:
    if not subject or not clip:
        return []

    output = ensure_ccw(subject)
    clip = ensure_ccw(clip)

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        output = clip_with_half_plane(output, a, b, eps)
        if not output:
            return []

    return cleanup_polygon(output, eps)


def main() -> None:
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    triangle = [(2.0, -1.0), (5.0, 2.0), (1.0, 5.0)]

    print("SAT intersect:", convex_polygons_intersect(square, triangle))
    intersection = convex_polygon_intersection(square, triangle)
    print("intersection polygon:", intersection)
    print("intersection area:", abs(signed_polygon_area(intersection)) if len(intersection) >= 3 else 0.0)


if __name__ == "__main__":
    main()
