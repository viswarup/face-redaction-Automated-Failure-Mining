"""Generate curated visual comparison strips for the GitHub README."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CROPS_DIR = ROOT / "mined_failures" / "crops"
FRAMES_DIR = ROOT / "mined_failures" / "frames"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def create_showcase_strip(
    full_frame_path: Path,
    crop_path: Path,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    frame = cv2.imread(str(full_frame_path))
    crop = cv2.imread(str(crop_path))
    if frame is None or crop is None:
        print(f"Skipping {output_path.name}: missing input image.")
        return

    # Resize full frame to standard width 800, keep aspect
    fh, fw = frame.shape[:2]
    target_w = 750
    target_h = int(fh * (target_w / fw))
    frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Resize crop to height target_h
    ch, cw = crop.shape[:2]
    crop_w = int(cw * (target_h / ch))
    if crop_w > 450:
        crop_w = 450
    crop_resized = cv2.resize(crop, (crop_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Combine side-by-side with divider
    divider = np.full((target_h, 8, 3), 40, dtype=np.uint8)
    combined_body = np.hstack([frame_resized, divider, crop_resized])

    # Add header banner
    total_w = combined_body.shape[1]
    banner_h = 70
    banner = np.full((banner_h, total_w, 3), (25, 20, 15), dtype=np.uint8)

    # Title text
    cv2.putText(banner, title, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1, cv2.LINE_AA)

    # Label on left & right panel
    cv2.putText(combined_body, "[Full Diagnostic Frame with HUD]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_body, "[Harvested Patch]", (target_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    final_image = np.vstack([banner, combined_body])
    cv2.imwrite(str(output_path), final_image)
    print(f"Created showcase image: {output_path.name} ({final_image.shape[1]}x{final_image.shape[0]})")


def main() -> None:
    # 1. Leak Showcase
    leak_frame = FRAMES_DIR / "LEAK_clip_low_f0_full.jpg"
    leak_crop = CROPS_DIR / "LEAK_clip_low_f0_crop.jpg"
    create_showcase_strip(
        leak_frame,
        leak_crop,
        "MINING SIGNAL 1: POST-RENDER VERIFIER DISCREPANCY (LEAK CAUGHT)",
        "RetinaFace detected residual facial features on rendered video | Priority: 10/10",
        ASSETS_DIR / "showcase_verifier_leak.jpg",
    )

    # 2. Ghost Flicker Showcase
    ghost_frame = FRAMES_DIR / "GHOST_Uni+ DVR_f257_t167_full.jpg"
    ghost_crop = CROPS_DIR / "GHOST_Uni+ DVR_f257_t167_crop.jpg"
    create_showcase_strip(
        ghost_frame,
        ghost_crop,
        "MINING SIGNAL 2: TEMPORAL KINEMATIC GHOST (1-FRAME HALLUCINATION)",
        "Detection appeared for exactly 1 frame with zero trajectory continuity | Priority: 6/10",
        ASSETS_DIR / "showcase_ghost_hallucination.jpg",
    )

    # 3. Margin Uncertainty Showcase
    uncert_frame = FRAMES_DIR / "UNCERT_Uni+ DVR_f100_s25_full.jpg"
    uncert_crop = CROPS_DIR / "UNCERT_Uni+ DVR_f100_s25_crop.jpg"
    create_showcase_strip(
        uncert_frame,
        uncert_crop,
        "MINING SIGNAL 3: MARGIN UNCERTAINTY SAMPLING (DECISION BOUNDARY)",
        "Detection confidence 0.25 sits on razor edge of decision boundary (0.20 - 0.28) | Priority: 5/10",
        ASSETS_DIR / "showcase_margin_uncertainty.jpg",
    )


if __name__ == "__main__":
    main()
