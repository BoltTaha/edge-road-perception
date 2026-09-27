import unittest

import numpy as np

from src.tracker import ByteTracker, iou_batch, linear_sum_assignment


class Detection:
    def __init__(self, bbox, score=0.9, class_id=2):
        self._bbox = np.array(bbox, dtype=np.float32)
        self.score = score
        self.class_id = class_id

    def as_xyxy(self):
        return self._bbox


class TrackerTests(unittest.TestCase):
    def test_hungarian_assignment(self):
        rows, cols = linear_sum_assignment(
            np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=np.float64)
        )
        self.assertEqual(
            list(zip(rows.tolist(), cols.tolist())), [(0, 1), (1, 0), (2, 2)]
        )

    def test_iou(self):
        result = iou_batch(
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([[5, 5, 15, 15]], dtype=np.float32),
        )
        self.assertTrue(np.isclose(result[0, 0], 25 / 175))

    def test_identity_persists(self):
        tracker = ByteTracker(match_thresh=0.3)
        first = tracker.update([Detection([100, 100, 180, 180])])
        second = tracker.update([Detection([108, 102, 188, 182])])
        self.assertEqual(second[0]["track_id"], first[0]["track_id"])

    def test_low_confidence_recovery(self):
        tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, match_thresh=0.3)
        first = tracker.update([Detection([100, 100, 180, 180], score=0.9)])
        second = tracker.update([Detection([106, 100, 186, 180], score=0.3)])
        self.assertEqual(second[0]["track_id"], first[0]["track_id"])

    def test_different_classes_do_not_match(self):
        matches, unmatched_tracks, unmatched_detections = ByteTracker._associate(
            np.array([[0, 0, 100, 100]], dtype=np.float32),
            np.array([[0, 0, 100, 100]], dtype=np.float32),
            0.3,
            np.array([2]),
            np.array([0]),
        )
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_detections, [0])


if __name__ == "__main__":
    unittest.main()

