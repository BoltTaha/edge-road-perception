"""Writes structured per-frame detection/track output as JSON Lines."""
import json


class StructuredOutputWriter:
    """
    One JSON object per line (JSONL), so the file can be streamed/tailed
    during a long run instead of held entirely in memory:
        {"frame": 0, "timestamp_ms": 0.0, "tracks": [...]}
    """

    def __init__(self, path):
        self._file = open(path, "w")

    def write_frame(self, frame_idx, timestamp_ms, tracks):
        record = {
            "frame": frame_idx,
            "timestamp_ms": round(timestamp_ms, 2),
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "class_id": t["class_id"],
                    "bbox": [round(v, 1) for v in t["bbox"]],
                    "score": round(t["score"], 3),
                }
                for t in tracks
            ],
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
