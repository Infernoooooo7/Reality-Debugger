# On-device model

`efficientdet_lite0.tflite` - EfficientDet-Lite0 (int8, 320×320, 80 COCO classes)
from the MediaPipe model zoo, licensed under the Apache License 2.0.

Source: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
SHA-256: `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb`

It is bundled so the app works offline on your LAN. To swap the model, drop
another MediaPipe-compatible object-detection `.tflite` file here and change
`MODEL_URL` in `src/vision/detector.ts` (or implement the `Detector`
interface in that file for a different runtime).
