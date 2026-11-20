"""CLI Runner for Automated Failure Mining Engine across all 6 demo video clips."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from miner import config
from miner.engine import FailureMiningEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_miner")


def load_jobs_from_db() -> dict[str, dict[str, Any]]:
    """Load jobs mapped by input snippet name."""
    db_path = config.JOBS_DB_PATH
    if not db_path.exists():
        logger.warning("jobs.db not found at %s", db_path)
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM jobs").fetchall()
    conn.close()

    mapping = {}
    for r in rows:
        inp = r["input_path"]
        if inp:
            base_name = Path(inp).name
            mapping[base_name] = dict(r)
    return mapping


def build_html_report(engine: FailureMiningEngine, output_html_path: Path) -> None:
    """Generate an interactive visual HTML report showcasing mined failure crops and statistics."""
    events = engine.mined_events
    total_events = len(events)
    p10_count = sum(1 for e in events if e.priority_score == 10)
    p8_count = sum(1 for e in events if e.priority_score == 8)
    p7_count = sum(1 for e in events if e.priority_score == 7)
    p6_count = sum(1 for e in events if e.priority_score == 6)
    p5_count = sum(1 for e in events if e.priority_score == 5)

    cards_html = []
    # Showcase top 30 representative failure cases
    sample_events = events[:36]

    for ev in sample_events:
        badge_color = "#ef4444" if ev.priority_score == 10 else "#f59e0b" if ev.priority_score >= 7 else "#3b82f6"
        crop_rel = ev.crop_path or ""
        frame_rel = ev.frame_path or ""

        card = f"""
        <div class="card">
            <div class="card-header">
                <span class="badge" style="background-color: {badge_color};">{ev.failure_category}</span>
                <span class="score">Priority: <b>{ev.priority_score}/10</b></span>
            </div>
            <div class="image-box">
                <img src="{crop_rel}" alt="{ev.event_id}" loading="lazy" />
            </div>
            <div class="card-body">
                <p><b>Clip:</b> {ev.clip_name[:32]}...</p>
                <p><b>Frame:</b> #{ev.frame_index} ({ev.timestamp_sec}s)</p>
                <p><b>Confidence:</b> {ev.confidence:.2f}</p>
                <p class="details">{ev.details}</p>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Active Learning Failure Mining Report (Phase 2)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0e1117;
            color: #f1f5f9;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            margin-bottom: 25px;
            border-bottom: 1px solid #334155;
            padding-bottom: 15px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            color: #ffffff;
            font-size: 26px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-num {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .kpi-label {{
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
        }}
        .badge {{
            font-size: 10px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            color: white;
            text-transform: uppercase;
        }}
        .score {{
            font-size: 11px;
            color: #cbd5e1;
        }}
        .image-box {{
            background: #000;
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .image-box img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .card-body {{
            padding: 12px;
            font-size: 12px;
            flex-grow: 1;
        }}
        .card-body p {{
            margin: 4px 0;
        }}
        .details {{
            color: #94a3b8;
            font-style: italic;
            margin-top: 8px !important;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Phase 2: Automated Failure Mining & Active Learning Curation</h1>
        <div class="subtitle">Heuristic failure detection across 6 body-camera video clips without live ground-truth labels.</div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-num" style="color: #38bdf8;">{total_events}</div>
            <div class="kpi-label">Total Mined Failure Points</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-num" style="color: #ef4444;">{p10_count}</div>
            <div class="kpi-label">P10: Verifier Leaks</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-num" style="color: #f59e0b;">{p8_count}</div>
            <div class="kpi-label">P8: Dropout Gap Bursts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-num" style="color: #fbbf24;">{p7_count}</div>
            <div class="kpi-label">P7: Velocity Shocks</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-num" style="color: #a855f7;">{p6_count}</div>
            <div class="kpi-label">P6: Ghost Hallucinations</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-num" style="color: #3b82f6;">{p5_count}</div>
            <div class="kpi-label">P5: Margin Uncertainty</div>
        </div>
    </div>

    <h2>🖼️ Curated Hard-Example Crop Gallery (Top Priority Exemplars)</h2>
    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    logger.info("Wrote interactive failure visual report to %s", output_html_path)


def main() -> None:
    logger.info("Starting Automated Failure Mining on 6 Demo Video Clips...")
    engine = FailureMiningEngine()
    db_mapping = load_jobs_from_db()

    demo_clips = sorted(list(config.DEMO_CLIPS_DIR.glob("*.mp4")))
    if not demo_clips:
        logger.error("No demo clips found in %s", config.DEMO_CLIPS_DIR)
        return

    logger.info("Found %d demo clips to mine for failure exemplars.", len(demo_clips))

    for clip in demo_clips:
        # Match companion tracks file
        tracks_candidates = list(config.TRACKS_DIR.glob(f"*{clip.stem}*.tracks.json"))
        # Match by job_id from db mapping if possible
        job_info = db_mapping.get(clip.name)
        tracks_file = None
        flagged_frames = []

        if job_info:
            j_id = job_info.get("job_id")
            candidate = config.TRACKS_DIR / f"{j_id}_redacted.mp4.tracks.json"
            if candidate.exists():
                tracks_file = candidate
            f_raw = job_info.get("flagged_frames_json", "[]")
            try:
                flagged_frames = json.loads(f_raw) if isinstance(f_raw, str) else f_raw
            except Exception:
                flagged_frames = []

        if not tracks_file and tracks_candidates:
            tracks_file = tracks_candidates[0]

        engine.mine_clip(clip_path=clip, tracks_json_path=tracks_file, flagged_frames=flagged_frames)

    # Export dataset
    json_path, csv_path = engine.export_dataset()

    # Build visual report
    html_report_path = config.OUTPUT_DIR / "mining_report.html"
    build_html_report(engine, html_report_path)

    # Display console summary table
    summary_df = engine.get_summary_dataframe()
    print("\n" + "=" * 90)
    print(">>> AUTOMATED FAILURE MINING COMPLETED (PHASE 2 SUMMARY) <<<")
    print("=" * 90)
    if not summary_df.empty:
        print(tabulate(summary_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print(f"\nActive Learning Dataset saved to:")
    print(f" • JSON: {json_path}")
    print(f" • CSV:  {csv_path}")
    print(f" • Interactive Visual Report: {html_report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
