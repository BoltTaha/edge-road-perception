"""
Central configuration for the perception pipeline.
Edit this file (or override via CLI flags in main.py) to change
model path, video source, thresholds, and device selection.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
DETECTOR_ONNX_PATH = os.path.join(BASE_DIR, "models", "yolov8n.onnx")
DETECTOR_INPUT_SIZE = 640          # square input side, pixels
CONF_THRESHOLD = 0.1  # retain candidates for the tracker's low-confidence pass
NMS_IOU_THRESHOLD = 0.45

# COCO class ids we care about for a driving scene.
# (full COCO index -> name is in src/coco_classes.py)
RELEVANT_CLASS_IDS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ---------------------------------------------------------------------------
# Tracking (ByteTrack)
# ---------------------------------------------------------------------------
TRACK_HIGH_THRESH = 0.5     # score threshold for "high confidence" detections
TRACK_LOW_THRESH = 0.1      # score threshold for the second-pass association
TRACK_MATCH_THRESH = 0.3    # minimum IoU accepted by Hungarian matching
TRACK_BUFFER_FRAMES = 30    # how many frames to keep a lost track alive
MIN_BOX_AREA = 10           # discard degenerate boxes

# ---------------------------------------------------------------------------
# Lane / drivable-area detection (classical CV, no extra model needed)
# ---------------------------------------------------------------------------
LANE_CANNY_LOW = 50
LANE_CANNY_HIGH = 150
LANE_HOUGH_THRESHOLD = 20
LANE_HOUGH_MIN_LINE_LEN = 20
LANE_HOUGH_MAX_LINE_GAP = 300
LANE_SLOPE_MIN = 0.4        # filter near-horizontal noise lines
# Region of interest as fractions of (width, height); a trapezoid over the
# lower half of the frame, roughly where a forward-facing dash-cam road is.
LANE_ROI_TOP_WIDTH_FRAC = 0.12
LANE_ROI_BOTTOM_WIDTH_FRAC = 1.0
LANE_ROI_TOP_Y_FRAC = 0.60
LANE_ROI_BOTTOM_Y_FRAC = 1.0
LANE_SMOOTHING_ALPHA = 0.2  # EMA smoothing of lane lines across frames

# ---------------------------------------------------------------------------
# Runtime / device
# ---------------------------------------------------------------------------
# onnxruntime execution providers, in priority order. CPUExecutionProvider
# always works; CUDAExecutionProvider is used automatically if available.
EXECUTION_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = os.path.join(BASE_DIR, "test_videos", "solidWhiteRight.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVE_VIDEO = True
SAVE_JSON = True
DISPLAY_WINDOW = False   # set True only on a machine with a display
MAX_FRAMES = None        # None = process whole video; set an int to cap for quick tests
