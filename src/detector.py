"""
Detection stage: loads a YOLOv8 ONNX model and runs inference via
ONNX Runtime (no PyTorch dependency at inference time -- this is the
same runtime you would use on an edge device).
"""
import time
import numpy as np
import cv2
import onnxruntime as ort

from src.coco_classes import COCO_CLASSES


class Detection:
    """A single detection result for one frame."""

    __slots__ = ("x1", "y1", "x2", "y2", "score", "class_id", "class_name")

    def __init__(self, x1, y1, x2, y2, score, class_id):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.score = score
        self.class_id = class_id
        self.class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)

    def as_xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_dict(self):
        return {
            "bbox": [round(float(v), 1) for v in (self.x1, self.y1, self.x2, self.y2)],
            "score": round(float(self.score), 3),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


class YOLOv8OnnxDetector:
    """Thin, dependency-light wrapper around an exported YOLOv8 ONNX graph."""

    def __init__(self, onnx_path, input_size=640, conf_threshold=0.35,
                 nms_iou_threshold=0.45, relevant_class_ids=None,
                 providers=None):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.relevant_class_ids = relevant_class_ids  # dict or None = all classes

        available = ort.get_available_providers()
        chosen = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=chosen)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    # -- preprocessing --------------------------------------------------
    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - nh) // 2
        left = (self.input_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def _preprocess(self, frame):
        canvas, scale, left, top = self._letterbox(frame)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        blob = blob.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(blob), scale, left, top

    # -- postprocessing ---------------------------------------------------
    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(), scores=scores.tolist(),
            score_threshold=0.0, nms_threshold=iou_threshold,
        )
        if len(idxs) == 0:
            return []
        return np.array(idxs).flatten().tolist()

    def _postprocess(self, output, scale, left, top, orig_shape):
        # YOLOv8 export shape: (1, 4 + num_classes, num_anchors)
        preds = output[0]
        preds = np.squeeze(preds, axis=0).T  # -> (num_anchors, 4 + num_classes)

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        keep = scores >= self.conf_threshold
        if self.relevant_class_ids is not None:
            allowed = np.isin(class_ids, list(self.relevant_class_ids.keys()))
            keep = keep & allowed
        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2

        # undo letterbox -> original frame coordinates
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        w = w / scale
        h = h / scale

        boxes_xywh = np.stack([x1, y1, w, h], axis=1)
        keep_idx = self._nms(boxes_xywh, scores, self.nms_iou_threshold)

        H, W = orig_shape[:2]
        detections = []
        for i in keep_idx:
            bx1 = max(0.0, float(x1[i]))
            by1 = max(0.0, float(y1[i]))
            bx2 = min(float(W), float(x1[i] + w[i]))
            by2 = min(float(H), float(y1[i] + h[i]))
            if bx2 <= bx1 or by2 <= by1:
                continue
            detections.append(Detection(bx1, by1, bx2, by2, float(scores[i]), int(class_ids[i])))
        return detections

    # -- public API -------------------------------------------------------
    def infer(self, frame):
        """Run detection on a single BGR frame. Returns (detections, inference_ms)."""
        blob, scale, left, top = self._preprocess(frame)
        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})
        inference_ms = (time.perf_counter() - t0) * 1000.0
        detections = self._postprocess(output, scale, left, top, frame.shape)
        return detections, inference_ms
