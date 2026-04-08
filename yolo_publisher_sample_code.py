# Sample Code for Robotics_Assignment_4
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# YOLO Publisher code for Jetson NX.

# This code will give you an introduction to coding in ROS2. You are required to modify it for the assignment requirement and submit the source code on Canvas.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You have to run this code on Turtlebot3 Nvidia Jetson.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming, along with ROS2 concepts, to understand this code.

# This will be a harder coding assignment compared to Milestone Assignment 3.
# We recommend doing a revision on the Milestone Assignment 3 code to get a hang of coding in ROS2.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import cv2
import json  
from ultralytics import YOLO

class YoloJsonPublisher(Node):
    def __init__(self):
        ###############################################################
        # TODO 1: Initialize the Node with the name 'yolo_json_publisher'
        # Hint: super().__init__('your_node_name_here')
        super().__init__('yolo_json_publisher')
        ###############################################################


      
        ###############################################################
        # TODO 2: Create a publisher that sends standard String messages
        # Hint: self.create_publisher(MessageType, 'topic_name', queue_size)
        # We want to publish a String, to '/yolo/detections_json', with a queue of 10
        
        self.publisher_ = self.create_publisher(String, '/yolo/detections_json', 10)
        ###############################################################
        
        # Load YOLOv11 base model (Not Pose, Not Seg) onto the Jetson's GPU
        self.get_logger().info("Loading YOLOv11 model on CUDA...")
        
        ###############################################################
        # TODO 3: Select a model name and size to run. yolo11n (nano) would serve you well, but you can use a different model.
        # Also, select the optimization level you want for YOLO v11
        # CUDA - Use Nvidia GPU without optimizations
        # CUDA+TensorRT - Use Nvidia GPU with performance optimization

        # Note: It would take a long time to load the model when it is your first time. 
        
        # Option 1: CUDA:               self.model = YOLO('yolo11s.pt')
        #                               self.model.to('cuda:0')
        
        # Works only if you have created the engine file using the provided demo code and copied it to the same folder as this code.
        # Option 2: CUDA+TensorRT:      self.model = YOLO('yolo11s.engine') 
        
        self.model = YOLO('yolo11s.pt')
        self.model.to('cuda:0')
        ###############################################################
        
        
        # GStreamer pipeline for Raspberry Pi V2 Camera on Jetson CSI port
        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
        
        self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            return
        ###############################################################
        # TODO 4: Create a timer that triggers your callback function to capture frames
        # Hint: self.create_timer(timer_period_in_seconds, callback_function)
        # Set it to run every 0.05 seconds (20 Hz), and call `self.timer_callback`
        self.timer = self.create_timer(0.05, self.timer_callback)
  
        ###############################################################

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]

        # Initialize the dictionary to hold our data
        detection_data = {
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "frame_id": "camera_link",
            "detections": []
        }

        # Populate the dictionary with YOLO results
        for box in results.boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            detection_data["detections"].append({
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "bbox": {"cx": x_center, "cy": y_center, "w": width, "h": height}
            })
            
        ###############################################################
        # TODO 5: Convert the `detection_data` Python dictionary into a JSON formatted string
        # Hint: Use the json.dumps() function
        json_str = json.dumps(detection_data)
      
        ###############################################################
      
        ###############################################################
        # TODO 6: Create your ROS 2 String message and publish it
        # Hint:
        # msg = String()
        # msg.data = your_json_string
        # self.publisher_.publish(msg_variable)
      
        msg = String()
        msg.data = json_str
        self.publisher_.publish(msg)
        ###############################################################

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    
        ###############################################################
        # TODO 7: Initialize the ROS 2 Python client library
        # Hint: rclpy.init with necessary arguments (args)
        rclpy.init(args=args)
        ###############################################################
    
        node = YoloJsonPublisher()
        
        try:
            
            ###############################################################
            # TODO 8: Spin the node so it stays alive and continues to trigger the timer
            # Hint: rclpy.spin with the necessary arguments that define what to spin
            rclpy.spin(node)
            ###############################################################
            
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
