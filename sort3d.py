from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from evaluate import evaluate_3d_mot


def track_name(index: int) -> str:
    letters = []
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def read_detections(path: Path) -> dict[int, list[np.ndarray]]:
    frames: dict[int, list[np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = int(float(row["Timestamp"]))
                xyz = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])], dtype=float)
            except (KeyError, ValueError):
                continue
            frames.setdefault(timestamp, []).append(xyz)
    return frames


def write_tracks(path: Path, rows: Iterable[tuple[str, int, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "Timestamp", "X", "Y", "Z", "group"])
        for name, timestamp, xyz in sorted(rows, key=lambda item: (item[1], item[0])):
            writer.writerow([name, timestamp, xyz[0], xyz[1], xyz[2], "{}"])


def linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat = [(cost[r, c], r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
        for _, row, col in sorted(flat, key=lambda item: item[0]):
            if row not in used_rows and col not in used_cols:
                pairs.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        return pairs


@dataclass
class KalmanTrack:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_timestamp: int
    missed: int = 0
    hits: int = 1
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @classmethod
    def start(cls, track_id: int, timestamp: int, xyz: np.ndarray) -> "KalmanTrack":
        state = np.zeros(6, dtype=float)
        state[:3] = xyz
        covariance = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
        return cls(track_id, state, covariance, timestamp, history=[(timestamp, xyz.copy())])

    def predict(self, timestamp: int, process_noise: float) -> np.ndarray:
        dt = max(1, int(timestamp) - int(self.last_timestamp))
        transition = np.eye(6)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        q = np.eye(6) * float(process_noise)
        q[3:, 3:] *= 4.0
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + q
        self.last_timestamp = timestamp
        self.missed += 1
        return self.state[:3].copy()

    def keep_prediction(self, timestamp: int) -> None:
        self.history.append((timestamp, self.state[:3].copy()))

    def update(self, timestamp: int, xyz: np.ndarray, measurement_noise: float) -> None:
        measurement = np.asarray(xyz, dtype=float)
        observe = np.zeros((3, 6), dtype=float)
        observe[:, :3] = np.eye(3)
        residual = measurement - observe @ self.state
        residual_cov = observe @ self.covariance @ observe.T + np.eye(3) * float(measurement_noise)
        gain = self.covariance @ observe.T @ np.linalg.inv(residual_cov)
        self.state = self.state + gain @ residual
        self.covariance = (np.eye(6) - gain @ observe) @ self.covariance
        self.last_timestamp = timestamp
        self.missed = 0
        self.hits += 1
        self.history.append((timestamp, measurement.copy()))


def run_sort(
    input_path: Path,
    output_path: Path,
    gate: float,
    max_age: int,
    min_hits: int,
    process_noise: float,
    measurement_noise: float,
    fill_misses: bool,
) -> None:
    detections = read_detections(input_path)
    if not detections:
        write_tracks(output_path, [])
        return

    tracks: list[KalmanTrack] = []
    finished: list[KalmanTrack] = []
    next_id = 0

    for timestamp in range(min(detections), max(detections) + 1):
        frame_dets = detections.get(timestamp, [])
        predictions = [track.predict(timestamp, process_noise) for track in tracks]

        if tracks and frame_dets:
            det_array = np.stack(frame_dets, axis=0)
            pred_array = np.stack(predictions, axis=0)
            cost = np.linalg.norm(pred_array[:, None, :] - det_array[None, :, :], axis=-1)
        else:
            cost = np.zeros((len(tracks), len(frame_dets)), dtype=float)

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for track_idx, det_idx in linear_assignment(cost):
            if cost[track_idx, det_idx] <= gate:
                matches.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)

        for track_idx, det_idx in matches:
            tracks[track_idx].update(timestamp, frame_dets[det_idx], measurement_noise)

        if fill_misses:
            for track_idx, track in enumerate(tracks):
                if track_idx not in used_tracks and 0 < track.missed <= max_age:
                    track.keep_prediction(timestamp)

        for det_idx, xyz in enumerate(frame_dets):
            if det_idx not in used_dets:
                tracks.append(KalmanTrack.start(next_id, timestamp, xyz))
                next_id += 1

        alive: list[KalmanTrack] = []
        for track in tracks:
            if track.missed <= max_age:
                alive.append(track)
            else:
                finished.append(track)
        tracks = alive

    finished.extend(tracks)

    rows: list[tuple[str, int, np.ndarray]] = []
    for track in finished:
        if track.hits < min_hits:
            continue
        name = track_name(track.track_id)
        rows.extend((name, timestamp, xyz) for timestamp, xyz in track.history)
    write_tracks(output_path, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple 3D-SORT baseline over an existing 3D detection stream.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/TXT detections with columns object, Timestamp, X, Y, Z or equivalent Timestamp/X/Y/Z fields.")
    parser.add_argument("--output", type=Path, required=True, help="Output track file path.")
    parser.add_argument("--gt", type=Path, help="Optional GT CSV for MOT evaluation.")
    parser.add_argument("--result", type=Path, help="Optional evaluation output file. Required with --gt.")
    parser.add_argument("--dist-thr", type=float, help="Optional MOT matching threshold used with --gt.")
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--process-noise", type=float, default=0.01)
    parser.add_argument("--measurement-noise", type=float, default=0.05)
    parser.add_argument("--no-fill-misses", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gt is not None and (args.result is None or args.dist_thr is None):
        raise SystemExit("--gt requires both --result and --dist-thr")
    if args.result is not None and args.gt is None:
        raise SystemExit("--result requires --gt")

    run_sort(
        input_path=args.input,
        output_path=args.output,
        gate=args.gate,
        max_age=args.max_age,
        min_hits=args.min_hits,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        fill_misses=not args.no_fill_misses,
    )

    if args.gt is not None:
        summary = evaluate_3d_mot(
            pred_path=args.output,
            gt_path=args.gt,
            dist_thr=args.dist_thr,
            match_pred_range=True,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with args.result.open("w", encoding="utf-8") as handle:
            handle.write(summary.summary_text)
            if not summary.summary_text.endswith("\n"):
                handle.write("\n")


if __name__ == "__main__":
    main()