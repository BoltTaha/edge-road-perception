"""
Lane / drivable-area perception stage.

Deliberately implemented with classical computer vision (grayscale ->
blur -> Canny edges -> region-of-interest mask -> probabilistic Hough
transform -> slope-based left/right line grouping) rather than a
learned segmentation model. This keeps the stage extremely cheap
(sub-millisecond on CPU) which matters for the project's edge-latency
budget, and it needs no additional weights to ship or quantize.

A short exponential moving average smooths the fitted line across
frames so the overlay doesn't flicker on noisy frames.
"""
import numpy as np
import cv2


class LaneDetector:
    def __init__(self, canny_low=50, canny_high=150, hough_threshold=20,
                 hough_min_line_len=20, hough_max_line_gap=300,
                 slope_min=0.4, roi_top_width_frac=0.12,
                 roi_bottom_width_frac=1.0, roi_top_y_frac=0.60,
                 roi_bottom_y_frac=1.0, smoothing_alpha=0.2):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.slope_min = slope_min
        self.roi_top_width_frac = roi_top_width_frac
        self.roi_bottom_width_frac = roi_bottom_width_frac
        self.roi_top_y_frac = roi_top_y_frac
        self.roi_bottom_y_frac = roi_bottom_y_frac
        self.alpha = smoothing_alpha

        self._smoothed_left = None   # (x1, y1, x2, y2)
        self._smoothed_right = None

    def _region_of_interest(self, edges):
        h, w = edges.shape[:2]
        top_y = int(h * self.roi_top_y_frac)
        bottom_y = int(h * self.roi_bottom_y_frac)
        top_half_w = (w * self.roi_top_width_frac) / 2
        bottom_half_w = (w * self.roi_bottom_width_frac) / 2
        cx = w / 2
        polygon = np.array([[
            (cx - bottom_half_w, bottom_y),
            (cx - top_half_w, top_y),
            (cx + top_half_w, top_y),
            (cx + bottom_half_w, bottom_y),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask), polygon

    def _fit_side(self, lines_xy, side):
        """Average a set of (x1,y1,x2,y2) segments into one representative line via linear fit."""
        if not lines_xy:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in lines_xy:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if len(xs) < 2:
            return None
        # fit x = m*y + b (more stable than y=mx+b for near-vertical lane lines)
        m, b = np.polyfit(ys, xs, 1)
        return m, b

    def detect(self, frame):
        """
        Returns:
            overlay_lines: list of (x1, y1, x2, y2) int tuples ready to draw
            roi_polygon: the ROI polygon used (for optional debug drawing)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        masked_edges, roi_polygon = self._region_of_interest(edges)

        raw_lines = cv2.HoughLinesP(
            masked_edges, rho=2, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

        left_segments, right_segments = [], []
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.slope_min:
                    continue  # near-horizontal, not a lane boundary
                if slope < 0:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        y_bottom = int(h * self.roi_bottom_y_frac)
        y_top = int(h * self.roi_top_y_frac)

        overlay_lines = []
        for side_segments, attr_name in ((left_segments, "_smoothed_left"),
                                          (right_segments, "_smoothed_right")):
            fit = self._fit_side(side_segments, attr_name)
            prev = getattr(self, attr_name)
            if fit is not None:
                m, b = fit
                x_bottom = m * y_bottom + b
                x_top = m * y_top + b
                candidate = (x_bottom, y_bottom, x_top, y_top)
                if prev is None:
                    smoothed = candidate
                else:
                    smoothed = tuple(
                        self.alpha * c + (1 - self.alpha) * p
                        for c, p in zip(candidate, prev)
                    )
                setattr(self, attr_name, smoothed)
            else:
                smoothed = prev  # hold last known line if nothing detected this frame

            if smoothed is not None:
                overlay_lines.append(tuple(int(round(v)) for v in smoothed))

        return overlay_lines, roi_polygon
