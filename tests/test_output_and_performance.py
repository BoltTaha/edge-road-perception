import json
import tempfile
import unittest
from pathlib import Path

from src.output_writer import StructuredOutputWriter
from src.performance import PerformanceMonitor


class OutputAndPerformanceTests(unittest.TestCase):
    def test_jsonl_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "detections.jsonl"
            with StructuredOutputWriter(path) as writer:
                writer.write_frame(
                    7,
                    280.123,
                    [{"track_id": 4, "class_id": 2, "bbox": [1.23, 2.34, 30.45, 40.56], "score": 0.87654}],
                )
            record = json.loads(path.read_text())
        self.assertEqual(record["frame"], 7)
        self.assertEqual(record["timestamp_ms"], 280.12)
        self.assertEqual(record["tracks"][0]["score"], 0.877)

    def test_performance_summary(self):
        monitor = PerformanceMonitor()
        monitor.record_stage_times(10.0, 2.0, 1.0, 20.0)
        monitor.record_stage_times(14.0, 4.0, 3.0, 40.0)
        report = monitor.summary("test CPU")
        self.assertEqual(report["frames_processed"], 2)
        self.assertEqual(report["overall_fps"], 33.33)
        self.assertEqual(report["detector_inference_ms"]["mean"], 12.0)
        self.assertEqual(report["test_hardware"], "test CPU")


if __name__ == "__main__":
    unittest.main()

