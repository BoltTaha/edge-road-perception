# Edge Deployment Recommendation

**Jetson Orin Nano is the most flexible first target** when future work may add learned lane segmentation, depth estimation, or more temporal models. TensorRT provides a direct path from the included ONNX detector.

**Raspberry Pi 5 with a Hailo accelerator is a strong lower-cost option** when the workload remains object detection plus CPU-side tracking and classical lane processing.

| Platform | Runtime | Fit |
|---|---|---|
| NVIDIA Jetson Orin Nano or NX | TensorRT | Best flexibility and model headroom |
| Raspberry Pi 5 plus Hailo-8 or Hailo-8L | HailoRT | Efficient detector offload |
| Intel NUC or edge box | OpenVINO | Practical on existing Intel hardware |
| Raspberry Pi 5 CPU only | ONNX Runtime | Development fallback at reduced input size |

## Optimization order

1. Establish an FP32 baseline on representative labeled footage.
2. Move detector inference to the target accelerator and test FP16.
3. Evaluate INT8 with representative calibration frames.
4. Compare input resolutions when more throughput is required.
5. Measure video decode, host-device transfers, rendering, and output after inference is accelerated.
6. Add bounded latest-frame processing only when target-device profiling justifies concurrency.

## Measurements to publish

- Detector and end-to-end mean, p50, p95, and maximum latency
- Sustained FPS, memory, accelerator use, and temperature
- Model, resolution, precision, runtime, and hardware model
- Detection precision and recall on representative labeled footage
- HOTA, MOTA, IDF1, or another justified tracking metric
- Cold start and long-run stability

Do not infer target-device performance from the bundled single-CPU reports.

