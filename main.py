#!/usr/bin/env python3
"""
Run the real-time edge road-perception system.

Examples
--------
  # Run on the bundled demo clip, saving annotated video + JSON + report
  python main.py --source test_videos/solidWhiteRight.mp4

  # Quick smoke test on the first 100 frames only
  python main.py --source test_videos/solidWhiteRight.mp4 --max-frames 100

  # Use a webcam (device 0) and show a live window (needs a display)
  python main.py --source 0 --display
"""
import argparse
import json
import sys

from src.pipeline import PerceptionPipeline
import config


def parse_args():
    p = argparse.ArgumentParser(description="Edge road-perception system")
    p.add_argument("--source", default=config.DEFAULT_SOURCE,
                   help="Path to a video file, or an integer camera index (e.g. 0)")
    p.add_argument("--output-dir", default=config.OUTPUT_DIR,
                   help="Where to write annotated video / JSON / performance report")
    p.add_argument("--max-frames", type=int, default=config.MAX_FRAMES,
                   help="Stop after N frames (useful for quick tests)")
    p.add_argument("--no-video", action="store_true", help="Don't write an annotated video")
    p.add_argument("--no-json", action="store_true", help="Don't write structured JSON output")
    p.add_argument("--display", action="store_true", help="Show a live OpenCV window (needs a display)")
    p.add_argument("--draw-roi", action="store_true", help="Draw the lane-detection ROI polygon (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    pipeline = PerceptionPipeline(
        source=source,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        save_video=not args.no_video,
        save_json=not args.no_json,
        display=args.display,
        draw_roi=args.draw_roi,
    )

    print(f"[info] running on source: {source}")
    print(f"[info] detector provider: {pipeline.detector.active_provider}")

    report = pipeline.run()

    print("\n=== Performance summary ===")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
