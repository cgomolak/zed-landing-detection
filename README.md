# zed-landing-detection
Object detection using a custom model designed to detect a predesigned landing pad with the ZED-SDK.Object detection using a custom model designed to detect a predesigned landing pad. This program is based off of the tensorrt_yolov5-v8 custom object detection sample from the zed-sdk.

Be sure to delete the included YOLOv5 model (.pt) that was trained on the NVIDIA 2070 Max-Q.

detector.py contains the main object detection and flight program while drone_flight_controls_v2.py contains supporting drone movement commands.
