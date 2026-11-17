"""Configuration and heuristics thresholds for the Automated Failure Mining Engine."""

from __future__ import annotations

import os
from pathlib import Path

# Base directories
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent

# Input data sources
DEMO_CLIPS_DIR = WORKSPACE_ROOT / "data" / "video_snippets" / "demo_clips"
TRACKS_DIR = WORKSPACE_ROOT / "data" / "outputs" / "redacted_videos"
JOBS_DB_PATH = WORKSPACE_ROOT / "data" / "jobs.db"

# Output data directory
OUTPUT_DIR = REPO_ROOT / "mined_failures"
CROPS_DIR = OUTPUT_DIR / "crops"
FRAMES_DIR = OUTPUT_DIR / "frames"

# Phase 2 Heuristics Thresholds
GAP_BURST_MIN_FRAMES = 3          # Dropout burst: >= 3 consecutive interpolated frames
GHOST_MAX_SPAN = 1                # Ghost detection: appears for exactly 1 frame
VELOCITY_SHOCK_PIXELS = 45.0      # Velocity shock: center jump > 45px between consecutive frames
MARGIN_CONF_MIN = 0.20            # Margin uncertainty: lower confidence floor
MARGIN_CONF_MAX = 0.28            # Margin uncertainty: upper confidence floor
LOW_LUMINANCE_FLOOR = 35.0        # Environmental stress: mean luminance < 35.0
LOW_BLUR_VAR_FLOOR = 80.0         # Environmental stress: Laplacian variance < 80.0
MAX_CROPS_PER_EVENT = 15          # Quota cap per failure event to prevent saturation


def ensure_directories() -> None:
    """Ensure output directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
