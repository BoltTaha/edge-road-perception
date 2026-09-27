# Benchmarking

The bundled `detector_inference_ms` measures the ONNX Runtime session call. Tracking and lane processing are timed separately. `total_per_frame_ms` covers those measured compute stages but excludes capture, visualization, video encoding, and output writes.

For deployment measurements, record:

- Hardware, execution provider, model, precision, and input resolution
- Video resolution, codec, FPS, and evaluated frame count
- Whether capture, rendering, encoding, and structured output are included
- Mean, p50, p95, and maximum latency
- Sustained FPS and peak memory

Visual inspection verifies integration but is not an accuracy benchmark. Use labeled representative footage for detection metrics and HOTA, MOTA, IDF1, or another identity-aware metric for tracking.

