"""Performance measurement: rolling FPS + a final summary report."""
import json
import platform
import time
from collections import deque

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class PerformanceMonitor:
    def __init__(self, window=30):
        self.window = window
        self._frame_times = deque(maxlen=window)
        self._inference_times = []      # full history, ms
        self._lane_times = []           # full history, ms
        self._tracking_times = []       # full history, ms
        self._total_frame_times = []    # full history, ms (whole loop, incl. I/O)
        self._last_tick = None

    def tick(self):
        now = time.perf_counter()
        if self._last_tick is not None:
            self._frame_times.append(now - self._last_tick)
        self._last_tick = now

    def rolling_fps(self):
        if not self._frame_times:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    def record_stage_times(self, inference_ms, lane_ms, tracking_ms, total_ms):
        self._inference_times.append(inference_ms)
        self._lane_times.append(lane_ms)
        self._tracking_times.append(tracking_ms)
        self._total_frame_times.append(total_ms)

    def _stats(self, values):
        if not values:
            return {"mean": 0, "p50": 0, "p95": 0, "max": 0}
        arr = np.array(values)
        return {
            "mean": round(float(arr.mean()), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "max": round(float(arr.max()), 2),
        }

    def summary(self, test_hardware_note=""):
        n = len(self._total_frame_times)
        overall_fps = 0.0
        if self._total_frame_times:
            overall_fps = 1000.0 / (sum(self._total_frame_times) / n)

        report = {
            "frames_processed": n,
            "overall_fps": round(overall_fps, 2),
            "detector_inference_ms": self._stats(self._inference_times),
            "lane_detection_ms": self._stats(self._lane_times),
            "tracking_ms": self._stats(self._tracking_times),
            "total_per_frame_ms": self._stats(self._total_frame_times),
            "test_hardware": test_hardware_note or platform.processor() or platform.machine(),
            "python_version": platform.python_version(),
        }
        if _HAS_PSUTIL:
            report["cpu_count"] = psutil.cpu_count(logical=True)
            report["peak_process_memory_mb"] = round(
                psutil.Process().memory_info().rss / (1024 * 1024), 1
            )
        return report

    def save_report(self, path, test_hardware_note=""):
        report = self.summary(test_hardware_note)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return report
