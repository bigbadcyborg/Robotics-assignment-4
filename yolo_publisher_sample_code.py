# Sample Code for Robotics_Assignment_4
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# YOLO Publisher code for Jetson NX.

# This code will give you an introduction to coding in ROS2. You are required to modify it for the assignment requirement and submit the source code on Canvas.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You have to run this code on Turtlebot3 Nvidia Jetson.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming, along with ROS2 concepts, to understand this code.

# This will be a harder coding assignment compared to Milestone Assignment 3.
# We recommend doing a revision on the Milestone Assignment 3 code to get a hang of coding in ROS2.

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


@dataclass(frozen=True)
class PublisherConfig:
    """Runtime configuration exposed as ROS parameters."""

    model_path: str
    backend: str
    topic: str
    timer_period_sec: float
    frame_id: str


class DetectionSerializer:
    """Converts model output into assignment-compatible JSON payloads."""

    def __init__(self, frame_id: str) -> None:
        self._frame_id = frame_id

    def to_json(self, names: Dict[int, str], boxes: Any, timestamp_sec: float) -> str:
        detection_data: Dict[str, Any] = {
            "timestamp": timestamp_sec,
            "frame_id": self._frame_id,
            "detections": [],
        }

        for box in boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = names.get(class_id, str(class_id))
            detection_data["detections"].append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "cx": x_center,
                        "cy": y_center,
                        "w": width,
                        "h": height,
                    },
                }
            )

        return json.dumps(detection_data)


class YoloDetector:
    """Encapsulates model loading and frame inference."""

    def __init__(self, model_path: str, backend: str, logger: Any) -> None:
        self._logger = logger
        self._backend = backend.lower().strip()
        self._model = YOLO(model_path)
        self._configure_backend()

    @property
    def model_names(self) -> Dict[int, str]:
        return self._model.names

    def infer(self, frame: Any) -> Any:
        return self._model(frame, verbose=False)[0]

    def _configure_backend(self) -> None:
        if self._backend == "cuda":
            self._logger.info("Using CUDA backend for YOLO inference")
            self._model.to("cuda:0")
        elif self._backend == "cpu":
            self._logger.info("Using CPU backend for YOLO inference")
            self._model.to("cpu")
        elif self._backend == "tensorrt":
            self._logger.info(
                "Using TensorRT backend (engine model expected, no .to() call required)"
            )
        elif self._backend == "auto":
            self._logger.info(
                "Using AUTO backend (model default runtime selection by Ultralytics)"
            )
        else:
            self._logger.warn(
                f"Unsupported backend '{self._backend}'. Falling back to CUDA."
            )
            self._backend = "cuda"
            self._model.to("cuda:0")


class CsiCameraSource:
    """Owns camera lifecycle and frame acquisition from Jetson CSI input."""

    GSTREAMER_PIPELINE = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._cap = cv2.VideoCapture(self.GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError("Failed to open camera.")

    def read(self) -> Any:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()


class YoloJsonPublisher(Node):
    def __init__(self):
        super().__init__("yolo_json_publisher")

        self._config = self._load_config()
        self.publisher_ = self.create_publisher(String, self._config.topic, 10)
        self._serializer = DetectionSerializer(frame_id=self._config.frame_id)

        self.get_logger().info(
            "Loading YOLO model with parameters: "
            f"model_path={self._config.model_path}, backend={self._config.backend}, "
            f"topic={self._config.topic}, timer_period={self._config.timer_period_sec:.3f}s"
        )

        self._detector = YoloDetector(
            model_path=self._config.model_path,
            backend=self._config.backend,
            logger=self.get_logger(),
        )
        self._camera = CsiCameraSource(logger=self.get_logger())
        self.timer = self.create_timer(self._config.timer_period_sec, self.timer_callback)

    def _load_config(self) -> PublisherConfig:
        self.declare_parameter("model_path", "yolo11s.pt")
        self.declare_parameter("backend", "cuda")
        self.declare_parameter("topic", "/yolo/detections_json")
        self.declare_parameter("timer_period_sec", 0.05)
        self.declare_parameter("frame_id", "camera_link")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        backend = self.get_parameter("backend").get_parameter_value().string_value
        topic = self.get_parameter("topic").get_parameter_value().string_value
        timer_period_sec = (
            self.get_parameter("timer_period_sec")
            .get_parameter_value()
            .double_value
        )
        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        if timer_period_sec <= 0.0:
            self.get_logger().warn(
                f"timer_period_sec={timer_period_sec} is invalid. Using default 0.05."
            )
            timer_period_sec = 0.05

        if not topic:
            self.get_logger().warn("Empty topic parameter received. Using /yolo/detections_json.")
            topic = "/yolo/detections_json"

        if not model_path:
            self.get_logger().warn("Empty model_path parameter received. Using yolo11s.pt.")
            model_path = "yolo11s.pt"

        return PublisherConfig(
            model_path=model_path,
            backend=backend,
            topic=topic,
            timer_period_sec=timer_period_sec,
            frame_id=frame_id,
        )

    def timer_callback(self):
        ret, frame = self._camera.read()
        if not ret:
            self.get_logger().warn("Camera frame read failed; skipping this cycle.")
            return

        try:
            results = self._detector.infer(frame)
            json_str = self._serializer.to_json(
                names=self._detector.model_names,
                boxes=results.boxes,
                timestamp_sec=self.get_clock().now().nanoseconds / 1e9,
            )
        except Exception as exc:
            self.get_logger().error(f"Inference/publish preparation failed: {exc}")
            return

        msg = String()
        msg.data = json_str
        self.publisher_.publish(msg)

    def destroy_node(self):
        self._camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = YoloJsonPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Node startup/runtime failure: {exc}")
        else:
            print(f"Node startup failure: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
