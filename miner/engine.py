"""Core Failure Mining Engine orchestrator: extracts frames, crops patches, and curates active learning dataset."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from miner import config
from miner.signals import (
    MinedFailureEvent,
    TemporalKinematicsMiner,
    UncertaintyMiner,
    VerifierDiscrepancyMiner,
)

logger = logging.getLogger("failure_miner.engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FailureMiningEngine:
    """
    Automated Failure Mining & Active Learning Curation Engine (Phase 2).
    Scans videos and companion pipeline telemetry to extract hard failure cases without ground-truth labels.
    """

    def __init__(self):
        config.ensure_directories()
        self.mined_events: list[MinedFailureEvent] = []

    @staticmethod
    def _read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a single frame from video via OpenCV."""
        if not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    @staticmethod
    def _crop_patch(frame: np.ndarray, bbox: list[int], margin: float = 0.25) -> np.ndarray:
        """Crop bounding box with fractional margin for contextual viewing."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        nw = bw * (1.0 + margin)
        nh = bh * (1.0 + margin)

        cx1 = max(0, int(cx - nw / 2.0))
        cy1 = max(0, int(cy - nh / 2.0))
        cx2 = min(w, int(cx + nw / 2.0))
        cy2 = min(h, int(cy + nh / 2.0))

        if (cx2 - cx1) < 5 or (cy2 - cy1) < 5:
            return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return frame[cy1:cy2, cx1:cx2]

    @staticmethod
    def _annotate_frame(frame: np.ndarray, event: MinedFailureEvent) -> np.ndarray:
        """Draw bounding box and diagnostic HUD on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = event.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Color palette by priority
        if event.priority_score >= 9:
            color = (0, 0, 255)       # Red: Critical leak
        elif event.priority_score >= 7:
            color = (0, 165, 255)     # Amber: High-value kinematic break
        else:
            color = (255, 200, 0)     # Yellow: Uncertainty / Ghost

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Category pill header
        label = f"[{event.failure_category}] Score: {event.priority_score}/10 | Conf: {event.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(vis, label, (x1 + 5, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return vis

    def mine_clip(
        self,
        clip_path: Path,
        tracks_json_path: Optional[Path] = None,
        flagged_frames: Optional[list[dict[str, Any]]] = None,
    ) -> list[MinedFailureEvent]:
        """Run all mining heuristics on a single video clip."""
        logger.info("Scanning clip for automated failure signals: %s", clip_path.name)
        clip_name = clip_path.name

        cap = cv2.VideoCapture(str(clip_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frames_data: list[dict[str, Any]] = []
        if tracks_json_path and tracks_json_path.exists():
            try:
                with open(tracks_json_path, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                    frames_data = t_data.get("frames", [])
            except Exception as e:
                logger.warning("Could not read tracks json %s: %s", tracks_json_path.name, e)

        clip_events: list[MinedFailureEvent] = []

        # 1. Post-Render Verifier Discrepancy (RetinaFace QA Leaks)
        if flagged_frames:
            leaks = VerifierDiscrepancyMiner.mine_verifier_leaks(flagged_frames, clip_name, fps)
            clip_events.extend(leaks[:config.MAX_CROPS_PER_EVENT])

        # 2. Temporal Kinematic Breaks (Dropout Bursts, Ghost Detections, Velocity Shocks)
        if frames_data:
            kinematics = TemporalKinematicsMiner(frames_data)
            bursts = kinematics.mine_dropout_bursts(clip_name, fps)
            ghosts = kinematics.mine_ghost_detections(clip_name, fps)
            shocks = kinematics.mine_velocity_shocks(clip_name, fps)

            clip_events.extend(bursts[:config.MAX_CROPS_PER_EVENT])
            clip_events.extend(ghosts[:config.MAX_CROPS_PER_EVENT])
            clip_events.extend(shocks[:config.MAX_CROPS_PER_EVENT])

            # 3. Margin Uncertainty Sampling (Conf on decision boundary)
            uncertainty = UncertaintyMiner.mine_margin_uncertainty(frames_data, clip_name, fps)
            clip_events.extend(uncertainty[:config.MAX_CROPS_PER_EVENT])

        # Extract frames & crops for top prioritized mined events
        logger.info("Mining complete for %s: found %d raw failure events. Extracting crops...", clip_name, len(clip_events))

        for ev in clip_events:
            frame_bgr = self._read_video_frame(clip_path, ev.frame_index)
            if frame_bgr is not None:
                # Save crop patch
                crop = self._crop_patch(frame_bgr, ev.bbox)
                crop_fname = f"{ev.event_id}_crop.jpg"
                crop_out = config.CROPS_DIR / crop_fname
                cv2.imwrite(str(crop_out), crop)
                ev.crop_path = f"crops/{crop_fname}"

                # Save annotated full frame
                annotated = self._annotate_frame(frame_bgr, ev)
                frame_fname = f"{ev.event_id}_full.jpg"
                frame_out = config.FRAMES_DIR / frame_fname
                cv2.imwrite(str(frame_out), annotated)
                ev.frame_path = f"frames/{frame_fname}"

        self.mined_events.extend(clip_events)
        return clip_events

    def export_dataset(self) -> tuple[Path, Path]:
        """Export collected active learning dataset to JSON and CSV."""
        json_path = config.OUTPUT_DIR / "active_learning_dataset.json"
        csv_path = config.OUTPUT_DIR / "active_learning_dataset.csv"

        records = [e.to_dict() for e in self.mined_events]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_mined_events": len(records),
                    "schema_version": "2.0-active-learning",
                    "priority_distribution": {
                        "p10_critical_leaks": sum(1 for r in records if r["priority_score"] == 10),
                        "p8_dropout_bursts": sum(1 for r in records if r["priority_score"] == 8),
                        "p7_velocity_shocks": sum(1 for r in records if r["priority_score"] == 7),
                        "p6_ghost_hallucinations": sum(1 for r in records if r["priority_score"] == 6),
                        "p5_margin_uncertainty": sum(1 for r in records if r["priority_score"] == 5),
                    },
                    "events": records,
                },
                f,
                indent=2,
            )

        # Export CSV
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)

        logger.info("Exported active learning dataset with %d entries to %s and %s", len(records), json_path.name, csv_path.name)
        return json_path, csv_path

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return failure category breakdown as a pandas DataFrame."""
        if not self.mined_events:
            return pd.DataFrame()
        df = pd.DataFrame([e.to_dict() for e in self.mined_events])
        summary = (
            df.groupby("failure_category")
            .agg(
                Count=("event_id", "count"),
                Mean_Priority=("priority_score", "mean"),
                Avg_Confidence=("confidence", "mean"),
                Clips_Impacted=("clip_name", "nunique"),
            )
            .reset_index()
            .sort_values(by="Count", ascending=False)
        )
        return summary
