# Architecture

The repository uses replaceable stages rather than one monolithic script.

1. `main.py` parses CLI options and creates the pipeline.
2. `src/pipeline.py` owns the frame loop.
3. `src/detector.py` letterboxes frames, runs ONNX inference, applies NMS, and restores source coordinates.
4. `src/tracker.py` uses a Kalman motion model and class-aware Hungarian IoU association.
5. `src/lane_detection.py` combines Canny edges, a trapezoidal ROI, Hough lines, slope grouping, and temporal smoothing.
6. `src/visualizer.py` draws the lane region, boxes, IDs, and performance HUD.
7. `src/output_writer.py` writes one JSON record per frame.
8. `src/performance.py` reports latency distributions and process information.

The detector, tracker, lane stage, and output layer can be replaced independently. The default loop has no unbounded frame queue.

