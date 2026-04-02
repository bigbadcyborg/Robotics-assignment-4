#!/usr/bin/env python3
"""Live visualization node for YOLO JSON detections (Part 3 helper).

This node intentionally focuses only on display concerns (Single Responsibility):
- Subscribes to `/yolo/detections_json`
- Reads CSI camera frames directly on Jetson
- Overlays detection boxes/labels
- Shows a live OpenCV window similar to CUDA demo scripts
"""

import json
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

WINDOW_NAME = "Part3 YOLO Stream"
DETECTION_STALE_SEC = 0.6


@dataclass
class Detection:
    class_name: str
    confidence: float
    cx: float
    cy: float
    w: float
    h: float

    @property
    def x1(self) -> int:
        return int(self.cx - self.w / 2.0)

    @property
    def y1(self) -> int:
        return int(self.cy - self.h / 2.0)

    @property
    def x2(self) -> int:
        return int(self.cx + self.w / 2.0)

    @property
    def y2(self) -> int:
        return int(self.cy + self.h / 2.0)


class DetectionCache:
    """Holds the latest parsed detections from YOLO JSON topic."""

    def __init__(self) -> None:
        self._detections: List[Detection] = []
        self._last_update: float = 0.0

    def update_from_json(self, payload: str, now_sec: float) -> None:
        data = json.loads(payload)
        detections = data.get("detections", [])
        parsed: List[Detection] = []

        for det in detections:
            bbox = det.get("bbox", {})
            parsed.append(
                Detection(
                    class_name=str(det.get("class_name", "unknown")),
                    confidence=float(det.get("confidence", 0.0)),
                    cx=float(bbox.get("cx", 0.0)),
                    cy=float(bbox.get("cy", 0.0)),
                    w=float(bbox.get("w", 0.0)),
                    h=float(bbox.get("h", 0.0)),
                )
            )

        self._detections = parsed
        self._last_update = now_sec

    def get(self, now_sec: float) -> List[Detection]:
        if now_sec - self._last_update > DETECTION_STALE_SEC:
            return []
        return self._detections


class Part3DetectionVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("part3_detection_visualizer")
        self._cache = DetectionCache()
        self._sub = self.create_subscription(
            String, "/yolo/detections_json", self._on_detections, 10
        )

        # Same camera pipeline style used in CUDA demo / publisher examples.
        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
        self._cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError("Failed to open CSI camera for visualization")

        self.get_logger().info("Visualizer started. Press 'q' to quit window.")

    def _on_detections(self, msg: String) -> None:
        try:
            self._cache.update_from_json(msg.data, time.time())
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"JSON parse error: {exc}")

    def render_once(self) -> bool:
        ok, frame = self._cap.read()
        if not ok:
            self.get_logger().warn("Camera frame read failed")
            return True

        detections = self._cache.get(time.time())
        for det in detections:
            color = (40, 220, 40) if det.class_name.lower() == "bottle" else (0, 180, 255)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (det.x1, max(24, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        status = f"detections: {len(detections)}"
        cv2.putText(
            frame,
            status,
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def shutdown(self) -> None:
        self._cap.release()
        cv2.destroyAllWindows()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = Part3DetectionVisualizer()

    try:
        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.01)
            running = node.render_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
