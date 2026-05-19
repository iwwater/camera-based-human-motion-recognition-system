# MediaPipe Model

The browser demo in `index.html` uses MediaPipe Pose Landmarker (BlazePose).
It tries the local model first and falls back to the MediaPipe CDN if the file
is not found.

## Offline use

1. Download `pose_landmarker_lite.task` (~5 MB) from
   [MediaPipe Pose Landmarker models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models).
2. Place the file at `models/pose_landmarker_lite.task` (this directory).

The demo then loads the model locally with no network access after page load.

## Online fallback

When `models/pose_landmarker_lite.task` is absent, the demo fetches the model
from `storage.googleapis.com/mediapipe-models` automatically. No configuration
needed — the repository works out of the box after `git clone`.
