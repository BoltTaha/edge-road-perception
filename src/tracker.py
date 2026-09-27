"""
Multi-object tracking stage.

Implements the core ideas of ByteTrack (Zhang et al., 2022):
  - Kalman-filter constant-velocity motion model per track
  - Two-stage association: match high-confidence detections first,
    then try to recover unmatched tracks with low-confidence detections
  - IoU-based Hungarian matching

This is a compact NumPy re-implementation rather than a wrapper around
a heavy external tracking library,
so it stays easy to read, audit, and port to an edge device later.
"""
import numpy as np


def linear_sum_assignment(cost_matrix):
    """Solve a rectangular minimum-cost assignment with the Hungarian method."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if 0 in cost.shape:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True
    rows, cols = cost.shape
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        minv = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        col0 = 0
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                current = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if current < minv[col]:
                    minv[col] = current
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                elif col > 0:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            previous = way[col0]
            p[col0] = p[previous]
            col0 = previous
            if col0 == 0:
                break
    pairs = []
    for col in range(1, cols + 1):
        if p[col]:
            pair = (p[col] - 1, col - 1)
            pairs.append((pair[1], pair[0]) if transposed else pair)
    pairs.sort()
    return (
        np.array([pair[0] for pair in pairs], dtype=np.int64),
        np.array([pair[1] for pair in pairs], dtype=np.int64),
    )


def iou_batch(boxes_a, boxes_b):
    """Vectorized IoU between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.expand_dims(boxes_a, 1)
    b = np.expand_dims(boxes_b, 0)
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over an 8-D state: [cx, cy, w, h, vx, vy, vw, vh]."""

    count = 0

    def __init__(self, bbox, score, class_id):
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self._build_matrices()

        KalmanBoxTracker.count += 1
        self.track_id = KalmanBoxTracker.count
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def _build_matrices(self):
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0

    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    @staticmethod
    def _cxcywh_to_xyxy(cx, cy, w, h):
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state_bbox()

    def update(self, bbox, score):
        z = np.array(self._xyxy_to_cxcywh(bbox), dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def get_state_bbox(self):
        cx, cy, w, h = self.state[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array(self._cxcywh_to_xyxy(cx, cy, w, h), dtype=np.float32)


class ByteTracker:
    """Two-stage IoU tracker following the ByteTrack association strategy."""

    def __init__(self, high_thresh=0.5, low_thresh=0.1, match_thresh=0.3,
                 track_buffer=30, min_box_area=10):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.tracks = []  # list of KalmanBoxTracker (all "confirmed/active" tracks)

    @staticmethod
    def _associate(tracks_bboxes, det_bboxes, iou_threshold,
                   track_class_ids=None, detection_class_ids=None):
        """Hungarian matching on IoU. Returns matches, unmatched_tracks, unmatched_dets (all as index lists)."""
        if len(tracks_bboxes) == 0 or len(det_bboxes) == 0:
            return [], list(range(len(tracks_bboxes))), list(range(len(det_bboxes)))

        iou_mat = iou_batch(tracks_bboxes, det_bboxes)
        if track_class_ids is not None and detection_class_ids is not None:
            same_class = np.equal.outer(track_class_ids, detection_class_ids)
            iou_mat = np.where(same_class, iou_mat, 0.0)
        cost = 1.0 - iou_mat
        row_idx, col_idx = linear_sum_assignment(cost)

        matches, unmatched_tracks, unmatched_dets = [], [], []
        matched_tracks_set, matched_dets_set = set(), set()
        for r, c in zip(row_idx, col_idx):
            if iou_mat[r, c] >= iou_threshold:
                matches.append((r, c))
                matched_tracks_set.add(r)
                matched_dets_set.add(c)
        unmatched_tracks = [i for i in range(len(tracks_bboxes)) if i not in matched_tracks_set]
        unmatched_dets = [i for i in range(len(det_bboxes)) if i not in matched_dets_set]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections):
        """
        detections: list of Detection objects (from src.detector) for the current frame.
        Returns: list of dicts describing each currently-active, confirmed track.
        """
        # 1) predict all existing tracks forward one step
        predicted_bboxes = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 4))

        # split detections by confidence (ByteTrack's core trick)
        high_dets = [d for d in detections if d.score >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        high_boxes = np.array([d.as_xyxy() for d in high_dets]) if high_dets else np.empty((0, 4))
        track_classes = np.array([t.class_id for t in self.tracks], dtype=np.int32)
        high_classes = np.array([d.class_id for d in high_dets], dtype=np.int32)
        matches, unmatched_tracks, unmatched_high = self._associate(
            predicted_bboxes, high_boxes, self.match_thresh,
            track_classes, high_classes,
        )
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(high_dets[d_idx].as_xyxy(), high_dets[d_idx].score)

        # second-stage: try to rescue still-unmatched tracks with low-score detections
        remaining_track_bboxes = np.array([predicted_bboxes[i] for i in unmatched_tracks]) if unmatched_tracks else np.empty((0, 4))
        remaining_track_classes = track_classes[unmatched_tracks]
        low_boxes = np.array([d.as_xyxy() for d in low_dets]) if low_dets else np.empty((0, 4))
        low_classes = np.array([d.class_id for d in low_dets], dtype=np.int32)
        matches2, unmatched_tracks2, _unmatched_low = self._associate(
            remaining_track_bboxes, low_boxes,
            iou_threshold=max(0.2, self.match_thresh - 0.1),
            track_class_ids=remaining_track_classes,
            detection_class_ids=low_classes,
        )
        still_unmatched_tracks = []
        matched2_track_positions = {m[0] for m in matches2}
        for local_idx, orig_idx in enumerate(unmatched_tracks):
            if local_idx in matched2_track_positions:
                d_local = [d for t, d in matches2 if t == local_idx][0]
                self.tracks[orig_idx].update(low_dets[d_local].as_xyxy(), low_dets[d_local].score)
            else:
                still_unmatched_tracks.append(orig_idx)

        # spawn new tracks from unmatched high-confidence detections only
        for d_idx in unmatched_high:
            d = high_dets[d_idx]
            self.tracks.append(KalmanBoxTracker(d.as_xyxy(), d.score, d.class_id))

        # drop tracks that have been lost for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]

        # emit confirmed, currently-visible tracks
        results = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue  # was not updated this frame -> currently occluded/lost, don't draw
            x1, y1, x2, y2 = t.get_state_bbox()
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            results.append({
                "track_id": t.track_id,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(t.score),
                "class_id": int(t.class_id),
            })
        return results
