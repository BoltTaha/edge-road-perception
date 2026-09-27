"""
Perception pipeline orchestrator.

Architecture:
  Video Input -> Preprocessing (in detector) -> Detection -> Tracking
  -> Lane/Drivable-area (parallel, independent of detection) ->
  Visualization + Structured Output

Each stage is a separate, independently testable/replaceable module
under src/. This file just wires them together and measures time
spent in each stage per frame.
"""
import os
import time

import cv2

import config
from src.detector import YOLOv8OnnxDetector
from src.tracker import ByteTracker
from src.lane_detection import LaneDetector
from src.performance import PerformanceMonitor
from src.output_writer import StructuredOutputWriter
from src import visualizer


class PerceptionPipeline:
    def __init__(self, source=None, output_dir=None, max_frames=None,
                 save_video=None, save_json=None, display=None, draw_roi=False):
        self.source = source or config.DEFAULT_SOURCE
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.max_frames = max_frames if max_frames is not None else config.MAX_FRAMES
        self.save_video = config.SAVE_VIDEO if save_video is None else save_video
        self.save_json = config.SAVE_JSON if save_json is None else save_json
        self.display = config.DISPLAY_WINDOW if display is None else display
        self.draw_roi = draw_roi

        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = YOLOv8OnnxDetector(
            onnx_path=config.DETECTOR_ONNX_PATH,
            input_size=config.DETECTOR_INPUT_SIZE,
            conf_threshold=config.CONF_THRESHOLD,
            nms_iou_threshold=config.NMS_IOU_THRESHOLD,
            relevant_class_ids=config.RELEVANT_CLASS_IDS,
            providers=config.EXECUTION_PROVIDERS,
        )
        self.tracker = ByteTracker(
            high_thresh=config.TRACK_HIGH_THRESH,
            low_thresh=config.TRACK_LOW_THRESH,
            match_thresh=config.TRACK_MATCH_THRESH,
            track_buffer=config.TRACK_BUFFER_FRAMES,
            min_box_area=config.MIN_BOX_AREA,
        )
        self.lane_detector = LaneDetector(
            canny_low=config.LANE_CANNY_LOW,
            canny_high=config.LANE_CANNY_HIGH,
            hough_threshold=config.LANE_HOUGH_THRESHOLD,
            hough_min_line_len=config.LANE_HOUGH_MIN_LINE_LEN,
            hough_max_line_gap=config.LANE_HOUGH_MAX_LINE_GAP,
            slope_min=config.LANE_SLOPE_MIN,
            roi_top_width_frac=config.LANE_ROI_TOP_WIDTH_FRAC,
            roi_bottom_width_frac=config.LANE_ROI_BOTTOM_WIDTH_FRAC,
            roi_top_y_frac=config.LANE_ROI_TOP_Y_FRAC,
            roi_bottom_y_frac=config.LANE_ROI_BOTTOM_Y_FRAC,
            smoothing_alpha=config.LANE_SMOOTHING_ALPHA,
        )
        self.perf = PerformanceMonitor()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_name = os.path.splitext(os.path.basename(str(self.source)))[0]
        video_writer = None
        if self.save_video:
            out_path = os.path.join(self.output_dir, f"{base_name}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_path, fourcc, src_fps, (width, height))

        json_writer = None
        if self.save_json:
            json_path = os.path.join(self.output_dir, f"{base_name}_detections.jsonl")
            json_writer = StructuredOutputWriter(json_path)

        frame_idx = 0
        try:
            while True:
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                t_frame_start = time.perf_counter()

                detections, inference_ms = self.detector.infer(frame)

                t_track_start = time.perf_counter()
                tracks = self.tracker.update(detections)
                tracking_ms = (time.perf_counter() - t_track_start) * 1000.0

                t_lane_start = time.perf_counter()
                lane_lines, roi_polygon = self.lane_detector.detect(frame)
                lane_ms = (time.perf_counter() - t_lane_start) * 1000.0

                self.perf.tick()
                total_ms = (time.perf_counter() - t_frame_start) * 1000.0
                self.perf.record_stage_times(inference_ms, lane_ms, tracking_ms, total_ms)

                annotated = frame
                annotated = visualizer.draw_lanes(annotated, lane_lines, roi_polygon, draw_roi=self.draw_roi)
                annotated = visualizer.draw_tracks(annotated, tracks)
                annotated = visualizer.draw_hud(
                    annotated, self.perf.rolling_fps(), inference_ms, frame_idx, len(tracks)
                )

                if video_writer is not None:
                    video_writer.write(annotated)
                if json_writer is not None:
                    json_writer.write_frame(frame_idx, frame_idx * (1000.0 / src_fps), tracks)
                if self.display:
                    cv2.imshow("Perception Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if json_writer is not None:
                json_writer.close()
            if self.display:
                cv2.destroyAllWindows()

        report_path = os.path.join(self.output_dir, f"{base_name}_performance_report.json")
        report = self.perf.save_report(
            report_path,
            test_hardware_note=f"CPU inference via onnxruntime ({self.detector.active_provider})",
        )
        return report
