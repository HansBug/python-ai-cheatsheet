from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-6 else inter / union


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    hits: int = 1
    missed: int = 0

    def predict(self) -> np.ndarray:
        predicted = self.box.copy()
        predicted[[0, 2]] += self.velocity[0]
        predicted[[1, 3]] += self.velocity[1]
        return predicted

    def update(self, new_box: np.ndarray) -> None:
        old_center = box_center(self.box)
        new_center = box_center(new_box)
        self.velocity = new_center - old_center
        self.box = new_box
        self.hits += 1
        self.missed = 0

    def mark_missed(self) -> None:
        self.box = self.predict()
        self.missed += 1


def greedy_match(
    tracks: list[Track],
    detections: list[np.ndarray],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    candidates: list[tuple[float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        predicted_box = track.predict()
        for det_idx, det in enumerate(detections):
            score = iou(predicted_box, det)
            if score >= iou_threshold:
                candidates.append((score, track_idx, det_idx))

    candidates.sort(reverse=True)

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for score, track_idx, det_idx in candidates:
        if track_idx in matched_tracks or det_idx in matched_dets:
            continue
        matches.append((track_idx, det_idx))
        matched_tracks.add(track_idx)
        matched_dets.add(det_idx)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 1) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: list[Track] = []

    def step(self, detections: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
        matches, unmatched_tracks, unmatched_dets = greedy_match(
            self.tracks,
            detections,
            self.iou_threshold,
        )

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_dets:
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    box=detections[det_idx].copy(),
                    velocity=np.zeros(2, dtype=np.float64),
                )
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return [(track.track_id, track.box.copy()) for track in self.tracks]


def main() -> None:
    np.set_printoptions(precision=2, suppress=True)

    frames = [
        [
            np.array([10, 10, 30, 30], dtype=np.float64),
            np.array([60, 12, 82, 32], dtype=np.float64),
        ],
        [
            np.array([14, 10, 34, 30], dtype=np.float64),
            np.array([56, 12, 78, 32], dtype=np.float64),
        ],
        [
            np.array([18, 10, 38, 30], dtype=np.float64),
        ],
        [
            np.array([22, 10, 42, 30], dtype=np.float64),
            np.array([48, 12, 70, 32], dtype=np.float64),
        ],
    ]

    tracker = SimpleTracker(iou_threshold=0.2, max_missed=1)
    for frame_idx, detections in enumerate(frames, start=1):
        outputs = tracker.step(detections)
        print(f"Frame {frame_idx}:")
        print("  detections:")
        for det in detections:
            print("   ", det)
        print("  active tracks:")
        for track_id, box in outputs:
            print(f"    id={track_id}, box={box}")


if __name__ == "__main__":
    main()
