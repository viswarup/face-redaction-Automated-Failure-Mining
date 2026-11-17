"""Signal detection logic for the 5 Automated Failure Mining Heuristics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from miner import config


@dataclass
class MinedFailureEvent:
    event_id: str
    clip_name: str
    frame_index: int
    timestamp_sec: float
    failure_category: str
    priority_score: int  # 1 to 10 (10 = highest retraining value)
    bbox: list[int]      # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int]
    signal_source: str
    details: str
    crop_path: Optional[str] = None
    frame_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def box_center(bbox: list[float] | list[int]) -> tuple[float, float]:
    """Calculate center (cx, cy) of bounding box."""
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def box_distance(box_a: list[float] | list[int], box_b: list[float] | list[int]) -> float:
    """Euclidean distance between two box centers."""
    c1 = box_center(box_a)
    c2 = box_center(box_b)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


class TemporalKinematicsMiner:
    """
    Mines temporal tracking memory breaks:
    1. Dropout Bursts: Tracking lost optical lock for >= 3 frames and had to interpolate.
    2. Ghost Detections: Detections appearing for exactly 1 frame (likely hallucinations).
    3. Velocity Shocks: Box centers jumping > 45px between consecutive frames.
    """

    def __init__(self, frames_data: list[dict[str, Any]]):
        self.frames = frames_data
        self.track_trajectories: dict[int, list[dict[str, Any]]] = {}
        self._build_trajectories()

    def _build_trajectories(self) -> None:
        """Organize frames into continuous track histories."""
        for f in self.frames:
            f_idx = f.get("frame_index", 0)
            for b in f.get("boxes", []):
                t_id = b.get("track_id")
                if t_id is not None:
                    if t_id not in self.track_trajectories:
                        self.track_trajectories[t_id] = []
                    self.track_trajectories[t_id].append(
                        {
                            "frame_index": f_idx,
                            "bbox": b.get("raw_bbox") or b.get("dilated_bbox"),
                            "source": b.get("source", "detection"),
                            "score": float(b.get("score", 0.0)),
                        }
                    )

    def mine_ghost_detections(self, clip_name: str, fps: float) -> list[MinedFailureEvent]:
        """Harvest tracks that existed for only 1 frame (transient false positives)."""
        events: list[MinedFailureEvent] = []
        for t_id, history in self.track_trajectories.items():
            if len(history) == config.GHOST_MAX_SPAN:
                item = history[0]
                box = [int(round(c)) for c in item["bbox"]]
                f_idx = item["frame_index"]
                events.append(
                    MinedFailureEvent(
                        event_id=f"GHOST_{clip_name[:8]}_f{f_idx}_t{t_id}",
                        clip_name=clip_name,
                        frame_index=f_idx,
                        timestamp_sec=round(f_idx / fps, 2),
                        failure_category="GHOST_DETECTION_FLICKER",
                        priority_score=6,
                        bbox=box,
                        confidence=item["score"],
                        track_id=t_id,
                        signal_source="TrackerKinematics",
                        details="Isolated single-frame detection without temporal continuity (suspected hallucination / false positive).",
                    )
                )
        return events

    def mine_dropout_bursts(self, clip_name: str, fps: float) -> list[MinedFailureEvent]:
        """Harvest sequences where the detector dropped out for >= 3 consecutive frames."""
        events: list[MinedFailureEvent] = []
        for t_id, history in self.track_trajectories.items():
            current_interpolated_streak: list[dict[str, Any]] = []
            for item in history:
                if item["source"] == "interpolated":
                    current_interpolated_streak.append(item)
                else:
                    if len(current_interpolated_streak) >= config.GAP_BURST_MIN_FRAMES:
                        # Mine the middle frame of the dropout burst as a hard false negative
                        mid_item = current_interpolated_streak[len(current_interpolated_streak) // 2]
                        box = [int(round(c)) for c in mid_item["bbox"]]
                        f_idx = mid_item["frame_index"]
                        streak_len = len(current_interpolated_streak)
                        events.append(
                            MinedFailureEvent(
                                event_id=f"BURST_{clip_name[:8]}_f{f_idx}_t{t_id}",
                                clip_name=clip_name,
                                frame_index=f_idx,
                                timestamp_sec=round(f_idx / fps, 2),
                                failure_category="DROPOUT_INTERPOLATION_BURST",
                                priority_score=8,
                                bbox=box,
                                confidence=mid_item["score"],
                                track_id=t_id,
                                signal_source="TrackerKinematics",
                                details=f"Primary detector suffered a {streak_len}-frame dropout lock break; tracker coasted trajectory.",
                            )
                        )
                    current_interpolated_streak = []

            # Check trailing streak
            if len(current_interpolated_streak) >= config.GAP_BURST_MIN_FRAMES:
                mid_item = current_interpolated_streak[len(current_interpolated_streak) // 2]
                box = [int(round(c)) for c in mid_item["bbox"]]
                f_idx = mid_item["frame_index"]
                streak_len = len(current_interpolated_streak)
                events.append(
                    MinedFailureEvent(
                        event_id=f"BURST_{clip_name[:8]}_f{f_idx}_t{t_id}",
                        clip_name=clip_name,
                        frame_index=f_idx,
                        timestamp_sec=round(f_idx / fps, 2),
                        failure_category="DROPOUT_INTERPOLATION_BURST",
                        priority_score=8,
                        bbox=box,
                        confidence=mid_item["score"],
                        track_id=t_id,
                        signal_source="TrackerKinematics",
                        details=f"Primary detector suffered a {streak_len}-frame dropout lock break; tracker coasted trajectory.",
                    )
                )
        return events

    def mine_velocity_shocks(self, clip_name: str, fps: float) -> list[MinedFailureEvent]:
        """Harvest frames where a track center jumped abruptly (> 45px between frames)."""
        events: list[MinedFailureEvent] = []
        for t_id, history in self.track_trajectories.items():
            if len(history) < 2:
                continue
            for idx in range(1, len(history)):
                prev = history[idx - 1]
                curr = history[idx]
                if curr["frame_index"] - prev["frame_index"] == 1:
                    dist = box_distance(prev["bbox"], curr["bbox"])
                    if dist >= config.VELOCITY_SHOCK_PIXELS:
                        box = [int(round(c)) for c in curr["bbox"]]
                        f_idx = curr["frame_index"]
                        events.append(
                            MinedFailureEvent(
                                event_id=f"SHOCK_{clip_name[:8]}_f{f_idx}_t{t_id}",
                                clip_name=clip_name,
                                frame_index=f_idx,
                                timestamp_sec=round(f_idx / fps, 2),
                                failure_category="VELOCITY_SHOCK_ANOMALY",
                                priority_score=7,
                                bbox=box,
                                confidence=curr["score"],
                                track_id=t_id,
                                signal_source="TrackerKinematics",
                                details=f"Kinematic velocity jump of {dist:.1f}px in a single frame (possible tracker identity switch / bounding box jitter).",
                            )
                        )
        return events


class UncertaintyMiner:
    """Mines boundary detections where model confidence hovers near decision cutoff (0.20 <= conf <= 0.28)."""

    @staticmethod
    def mine_margin_uncertainty(
        frames_data: list[dict[str, Any]], clip_name: str, fps: float
    ) -> list[MinedFailureEvent]:
        events: list[MinedFailureEvent] = []
        for f in frames_data:
            f_idx = f.get("frame_index", 0)
            for b in f.get("boxes", []):
                score = float(b.get("score", 0.0))
                if config.MARGIN_CONF_MIN <= score <= config.MARGIN_CONF_MAX and b.get("source") == "detection":
                    raw_b = b.get("raw_bbox") or b.get("dilated_bbox")
                    box = [int(round(c)) for c in raw_b]
                    events.append(
                        MinedFailureEvent(
                            event_id=f"UNCERT_{clip_name[:8]}_f{f_idx}_s{int(score*100)}",
                            clip_name=clip_name,
                            frame_index=f_idx,
                            timestamp_sec=round(f_idx / fps, 2),
                            failure_category="MARGIN_UNCERTAINTY_SAMPLING",
                            priority_score=5,
                            bbox=box,
                            confidence=score,
                            track_id=b.get("track_id"),
                            signal_source="ConfidenceDistribution",
                            details=f"Detection confidence {score:.3f} hovers on the razor edge of decision boundary ({config.MARGIN_CONF_MIN} - {config.MARGIN_CONF_MAX}).",
                        )
                    )
        return events


class VerifierDiscrepancyMiner:
    """Mines QA auditor leak flags recorded in SQLite jobs database."""

    @staticmethod
    def mine_verifier_leaks(
        flagged_frames: list[dict[str, Any]], clip_name: str, fps: float
    ) -> list[MinedFailureEvent]:
        events: list[MinedFailureEvent] = []
        for fl in flagged_frames:
            f_idx = fl.get("frame_index", 0)
            time_sec = float(fl.get("timestamp_sec") or (f_idx / fps))
            raw_box = fl.get("leak_bbox") or [0, 0, 50, 50]
            box = [int(round(c)) for c in raw_box]
            leak_type = fl.get("leak_type", "RESIDUAL_FACE_LEAK")
            score = float(fl.get("verifier_confidence", 0.35))
            events.append(
                MinedFailureEvent(
                    event_id=f"LEAK_{clip_name[:8]}_f{f_idx}",
                    clip_name=clip_name,
                    frame_index=f_idx,
                    timestamp_sec=round(time_sec, 2),
                    failure_category="POST_RENDER_VERIFIER_LEAK",
                    priority_score=10,  # Critical priority
                    bbox=box,
                    confidence=score,
                    track_id=None,
                    signal_source="RetinaFaceVerifierQA",
                    details=f"Independent post-render QA verifier detected residual facial feature leak: {leak_type} (conf={score:.2f}).",
                )
            )
        return events
