"""Generate professional, high-contrast visual comparison strips for the GitHub README."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
CROPS_DIR = ROOT / "mined_failures" / "crops"
FRAMES_DIR = ROOT / "mined_failures" / "frames"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def draw_text_lines(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int], float, int]],
    start_x: int,
    start_y: int,
    line_spacing: int = 26,
) -> None:
    """Helper to draw multiple lines of text with specified colors and fonts."""
    y = start_y
    for text, color, scale, thickness in lines:
        cv2.putText(
            img,
            text,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_spacing


def create_signal_strip(
    full_frame_path: Path,
    crop_path: Path,
    title: str,
    subtitle: str,
    signal_name: str,
    priority_label: str,
    priority_color: tuple[int, int, int],
    meta_lines: list[tuple[str, str]],
    action_text: str,
    output_path: Path,
) -> None:
    """Build a balanced, high-resolution 3-panel diagnostic card (Frame | Crop | Telemetry HUD)."""
    frame = cv2.imread(str(full_frame_path))
    crop = cv2.imread(str(crop_path))
    if frame is None or crop is None:
        print(f"Skipping {output_path.name}: missing input frame or crop.")
        return

    # Total card dimensions
    total_w = 1240
    body_h = 460
    banner_h = 70
    card_h = banner_h + body_h

    canvas = np.full((card_h, total_w, 3), (18, 22, 28), dtype=np.uint8)

    # 1. Top Header Banner
    banner = canvas[:banner_h, :]
    banner[:] = (12, 15, 20)
    cv2.putText(
        banner,
        title,
        (25, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        subtitle,
        (25, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (160, 185, 210),
        1,
        cv2.LINE_AA,
    )

    # Accent line beneath header
    cv2.line(canvas, (0, banner_h - 1), (total_w, banner_h - 1), priority_color, 2)

    # 2. Left Panel: Full Diagnostic Frame (w=520, h= body_h - 40)
    panel1_w = 510
    panel1_h = body_h - 30
    p1_x = 20
    p1_y = banner_h + 15

    fh, fw = frame.shape[:2]
    scale_f = min(panel1_w / fw, panel1_h / fh)
    resized_fw = int(fw * scale_f)
    resized_fh = int(fh * scale_f)
    frame_resized = cv2.resize(frame, (resized_fw, resized_fh), interpolation=cv2.INTER_AREA)

    # Center frame in panel 1 region
    off_x = p1_x + (panel1_w - resized_fw) // 2
    off_y = p1_y + (panel1_h - resized_fh) // 2
    canvas[off_y : off_y + resized_fh, off_x : off_x + resized_fw] = frame_resized
    cv2.rectangle(
        canvas,
        (off_x, off_y),
        (off_x + resized_fw, off_y + resized_fh),
        (50, 60, 75),
        1,
    )

    cv2.putText(
        canvas,
        "[Panel 1: Full Video Frame with Diagnostic HUD]",
        (p1_x, p1_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 160, 185),
        1,
        cv2.LINE_AA,
    )

    # 3. Center Panel: Harvested Face Crop with Context (w=300, h= body_h - 30)
    p2_x = p1_x + panel1_w + 20
    panel2_w = 280
    p2_y = p1_y

    cv2.putText(
        canvas,
        "[Panel 2: Harvested Hard-Example Patch]",
        (p2_x, p2_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 160, 185),
        1,
        cv2.LINE_AA,
    )

    # Crop container background
    crop_container = canvas[p2_y : p2_y + panel1_h, p2_x : p2_x + panel2_w]
    crop_container[:] = (26, 32, 40)
    cv2.rectangle(
        canvas,
        (p2_x, p2_y),
        (p2_x + panel2_w, p2_y + panel1_h),
        (60, 75, 95),
        1,
    )

    ch, cw = crop.shape[:2]
    max_cw = panel2_w - 30
    max_ch = panel1_h - 50
    scale_c = min(max_cw / cw, max_ch / ch)
    target_cw = max(10, int(cw * scale_c))
    target_ch = max(10, int(ch * scale_c))

    # Resize crop with Lanczos interpolation for crisp look
    crop_resized = cv2.resize(crop, (target_cw, target_ch), interpolation=cv2.INTER_LANCZOS4)
    c_off_x = p2_x + (panel2_w - target_cw) // 2
    c_off_y = p2_y + (panel1_h - target_ch) // 2

    canvas[c_off_y : c_off_y + target_ch, c_off_x : c_off_x + target_cw] = crop_resized
    cv2.rectangle(
        canvas,
        (c_off_x, c_off_y),
        (c_off_x + target_cw, c_off_y + target_ch),
        priority_color,
        2,
    )

    cv2.putText(
        canvas,
        f"{cw}x{ch} px native patch",
        (p2_x + 10, p2_y + panel1_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 200, 220),
        1,
        cv2.LINE_AA,
    )

    # 4. Right Panel: Active Learning Telemetry HUD (w=370)
    p3_x = p2_x + panel2_w + 20
    panel3_w = total_w - p3_x - 20
    p3_y = p1_y

    cv2.putText(
        canvas,
        "[Panel 3: Mining Telemetry & Metadata]",
        (p3_x, p3_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 160, 185),
        1,
        cv2.LINE_AA,
    )

    # Telemetry card background
    card_bg = canvas[p3_y : p3_y + panel1_h, p3_x : p3_x + panel3_w]
    card_bg[:] = (24, 29, 38)
    cv2.rectangle(
        canvas,
        (p3_x, p3_y),
        (p3_x + panel3_w, p3_y + panel1_h),
        (60, 75, 95),
        1,
    )

    # Priority Pill Badge
    (pw, ph), _ = cv2.getTextSize(priority_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    badge_x = p3_x + 15
    badge_y = p3_y + 15
    cv2.rectangle(
        canvas,
        (badge_x, badge_y),
        (badge_x + pw + 16, badge_y + 24),
        priority_color,
        -1,
    )
    cv2.putText(
        canvas,
        priority_label,
        (badge_x + 8, badge_y + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Signal title
    cv2.putText(
        canvas,
        signal_name,
        (badge_x, badge_y + 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (230, 240, 255),
        1,
        cv2.LINE_AA,
    )

    # Metadata key-value rows
    cur_y = badge_y + 80
    for label, val in meta_lines:
        cv2.putText(
            canvas,
            f"{label}:",
            (badge_x, cur_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (140, 160, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{val}",
            (badge_x + 120, cur_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cur_y += 24

    # Separator line
    cv2.line(
        canvas,
        (badge_x, cur_y + 5),
        (p3_x + panel3_w - 15, cur_y + 5),
        (50, 60, 75),
        1,
    )

    # Action box
    cur_y += 30
    cv2.putText(
        canvas,
        "ACTIVE LEARNING CURATION ACTION:",
        (badge_x, cur_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )
    cur_y += 20

    # Wrap action text
    words = action_text.split()
    line_buf = ""
    for w in words:
        test_line = f"{line_buf} {w}".strip()
        (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        if tw > (panel3_w - 35):
            cv2.putText(
                canvas,
                line_buf,
                (badge_x, cur_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 215, 230),
                1,
                cv2.LINE_AA,
            )
            cur_y += 18
            line_buf = w
        else:
            line_buf = test_line
    if line_buf:
        cv2.putText(
            canvas,
            line_buf,
            (badge_x, cur_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 215, 230),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)
    print(f"Generated clean showcase card: {output_path.name} ({total_w}x{card_h})")


def copy_golden_evaluation_strips() -> None:
    """Copy high-resolution multi-model & tracking proof strips into assets."""
    gallery_dir = WORKSPACE_ROOT / "data" / "outputs" / "failure_gallery"

    # 1. Temporal Tracking vs No-Tracking proof strip
    src_track = gallery_dir / "tracking_vs_notracking_01.jpg"
    dst_track = ASSETS_DIR / "showcase_temporal_flicker.jpg"
    if src_track.exists():
        shutil.copy2(src_track, dst_track)
        print(f"Copied golden asset: {dst_track.name}")

    # 2. Multi-Model Blindspot comparison (Low Light)
    src_comp1 = gallery_dir / "comparison_01.jpg"
    dst_comp1 = ASSETS_DIR / "showcase_low_light_blindspot.jpg"
    if src_comp1.exists():
        shutil.copy2(src_comp1, dst_comp1)
        print(f"Copied golden asset: {dst_comp1.name}")

    # 3. Multi-Model Steep Angle comparison
    src_comp3 = gallery_dir / "comparison_03.jpg"
    dst_comp3 = ASSETS_DIR / "showcase_steep_angle_blindspot.jpg"
    if src_comp3.exists():
        shutil.copy2(src_comp3, dst_comp3)
        print(f"Copied golden asset: {dst_comp3.name}")


def main() -> None:
    print("Building high-resolution showcase visual strips...")

    # Copy golden multi-panel diagnostic strips
    copy_golden_evaluation_strips()

    # 1. Post-Render Verifier Leak Showcase (Priority 10/10)
    # Using real face leak from clip_low_light_01.mp4, frame 60
    f_leak = FRAMES_DIR / "LEAK_clip_low_lig_f60_2_full.jpg"
    c_leak = CROPS_DIR / "LEAK_clip_low_lig_f60_2_crop.jpg"
    create_signal_strip(
        full_frame_path=f_leak,
        crop_path=c_leak,
        title="SIGNAL 1: POST-RENDER VERIFIER DISCREPANCY (LEAK DETECTED)",
        subtitle="Secondary RetinaFace QA Auditor caught visible face escaping blur on final render",
        signal_name="Post-Render Verifier Discrepancy",
        priority_label="PRIORITY 10 / 10 - CRITICAL LEAK",
        priority_color=(0, 0, 230),  # Red
        meta_lines=[
            ("Source Clip", "clip_low_light_01.mp4"),
            ("Video Frame", "#60 (Timestamp: 4.00s)"),
            ("Auditor Score", "0.486 (High Face Conf)"),
            ("Blur Overlap IoU", "0.00 (Zero Protective Blur)"),
            ("Classification", "UNBLURRED_MISSED_FACE"),
            ("Detection Model", "SCRFD-34G (Missed face)"),
        ],
        action_text="Emergency Hard-Negative. Auto-flagged for human supervisor review; patch coordinates staged directly into active learning retraining pool.",
        output_path=ASSETS_DIR / "showcase_verifier_leak.jpg",
    )

    # 2. Ghost Detection Flicker Showcase (Priority 6/10)
    # Using real ghost detection from Uni+ DVR_ch4, frame 86
    f_ghost = FRAMES_DIR / "GHOST_Uni_DVR_ch4_f86_t298_full.jpg"
    c_ghost = CROPS_DIR / "GHOST_Uni_DVR_ch4_f86_t298_crop.jpg"
    create_signal_strip(
        full_frame_path=f_ghost,
        crop_path=c_ghost,
        title="SIGNAL 2: TEMPORAL KINEMATIC GHOST (1-FRAME HALLUCINATION)",
        subtitle="Isolated detection appeared for exactly 1 frame with zero trajectory continuity",
        signal_name="Kinematic Ghost Hallucination",
        priority_label="PRIORITY 6 / 10 - FALSE POSITIVE",
        priority_color=(200, 120, 30),  # Amber/Blue
        meta_lines=[
            ("Source Clip", "Uni+ DVR_ch4_main_...seg2"),
            ("Video Frame", "#86 (Timestamp: 5.73s)"),
            ("Detector Score", "0.289 (Low Confidence)"),
            ("Trajectory Span", "1 Frame (Lost optical lock)"),
            ("Failure Mode", "Background texture misclassified"),
            ("Mining Heuristic", "Kalman kinematic lifespan == 1"),
        ],
        action_text="Negative Hard Sample. Added to negative mining buffer to suppress false-positive background clutter in bodycam scenes.",
        output_path=ASSETS_DIR / "showcase_ghost_hallucination.jpg",
    )

    # 3. Margin Uncertainty Sampling (Priority 5/10)
    # Using margin uncertainty from clip_low_light_01.mp4, frame 27
    f_uncert = FRAMES_DIR / "UNCERT_clip_low_lig_f27_s26_full.jpg"
    c_uncert = CROPS_DIR / "UNCERT_clip_low_lig_f27_s26_crop.jpg"
    create_signal_strip(
        full_frame_path=f_uncert,
        crop_path=c_uncert,
        title="SIGNAL 3: MARGIN UNCERTAINTY SAMPLING (DECISION BOUNDARY)",
        subtitle="Detection confidence hovers precisely on the decision threshold (0.20 - 0.28)",
        signal_name="Margin Uncertainty Sampling",
        priority_label="PRIORITY 5 / 10 - BOUNDARY CASE",
        priority_color=(30, 160, 230),  # Light Blue
        meta_lines=[
            ("Source Clip", "clip_low_light_01.mp4"),
            ("Video Frame", "#27 (Timestamp: 1.80s)"),
            ("Detector Score", "0.263 (Decision Boundary)"),
            ("Lighting Condition", "Extreme Low-Light / Noise"),
            ("Primary Detector", "SCRFD-34G Borderline"),
            ("Mining Heuristic", "0.20 <= Confidence <= 0.28"),
        ],
        action_text="Boundary Informative Sample. Maximizes information gain for model fine-tuning without redundant easy examples.",
        output_path=ASSETS_DIR / "showcase_margin_uncertainty.jpg",
    )

    print("All showcase assets generated successfully!")


if __name__ == "__main__":
    main()
