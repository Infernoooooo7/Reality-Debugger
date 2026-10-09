"""Vision platform core: normalised results, model registry, model adapters,
sliced inference, segmentation, anomaly detection, reference comparison and
motion estimation.

Framework-free on purpose (numpy, OpenCV, ONNX Runtime; MediaPipe only for
evaluation) so the same code serves the backend API and the offline
evaluation engine (app.evaluation, tools/evaluate.py).
"""
