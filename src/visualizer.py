"""Real-time visualization: detections, track IDs, lane overlay, perf HUD."""
import colorsys
import cv2
import numpy as np

from src.coco_classes import COCO_CLASSES


def _color_for_id(track_id):
    """Deterministic, visually distinct color per track id."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        color = _color_for_id(t["track_id"])
        class_name = COCO_CLASSES[t["class_id"]] if t["class_id"] < len(COCO_CLASSES) else str(t["class_id"])
        label = f'ID {t["track_id"]} {class_name} {t["score"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_lanes(frame, lane_lines, roi_polygon=None, draw_roi=False):
    overlay = frame.copy()
    for (x1, y1, x2, y2) in lane_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    if len(lane_lines) == 2:
        (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = lane_lines
        poly = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        drivable_overlay = frame.copy()
        cv2.fillPoly(drivable_overlay, [poly], (0, 200, 0))
        frame = cv2.addWeighted(drivable_overlay, 0.15, frame, 0.85, 0)

    if draw_roi and roi_polygon is not None:
        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=1)
    return frame


def draw_hud(frame, fps, inference_ms, frame_idx, num_tracks):
    lines = [
        f"FPS: {fps:5.1f}",
        f"Inference: {inference_ms:5.1f} ms",
        f"Frame: {frame_idx}",
        f"Active tracks: {num_tracks}",
    ]
    x, y = 10, 22
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return frame
