# Sample Code for Robotics_Assignment_4
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# YOLO Subscriber code for Jetson NX or Remote PC.

# This code will give you an introduction to coding in ROS2. You are required to modify it for the assignment requirement and submit the source code on Canvas.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You can run this code on Jeston NX or the Remote PC on the same network.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming, along with ROS2 concepts, to understand this code.

# This will be a harder coding assignment compared to Milestone Assignment 3.
# We recommend doing a revision on the Milestone Assignment 3 code to get a hang of coding in ROS2.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class YoloJsonSubscriber(Node):
    def __init__(self):
        ###############################################################
        # TODO 1: Initialize the Node with the name 'yolo_json_subscriber'
        # Hint: super().__init__('your_node_name_here')
        super().__init__('yolo_json_subscriber')
        ###############################################################

        ###############################################################
        # TODO 2: Create a subscriber to listen to the JSON string messages
        # Hint: self.create_subscription(MessageType, 'topic_name', callback_function, queue_size)
        # We are listening to a String, on '/yolo/detections_json', triggering `self.detection_callback`, queue of 10
        self.subscription = self.create_subscription(String, '/yolo/detections_json', self.detection_callback, 10)
        ###############################################################
      
        self.get_logger().info("Listening for YOLO JSON detections...")

    def detection_callback(self, msg):
        try:
            ###############################################################
            # TODO 3: Parse the incoming JSON string back into a usable Python dictionary
            # Hint: The string is stored in msg.data. Use json.loads() on msg.data
            data = json.loads(msg.data)
            ###############################################################
            
            # Extract the metadata
            # We have done this part for you.
            
            timestamp = data.get("timestamp", 0.0)
            frame_id = data.get("frame_id", "unknown")
            detections = data.get("detections", [])
            
            self.get_logger().info(f"Received {len(detections)} detections:")
            
            # Iterate through the detection list
            for i, det in enumerate(detections):

                ###############################################################
                # TODO 4: Extract the specific values from the `det` dictionary
                # Hint: The dictionary keys are "class_name", "confidence", and "bbox"
                class_name = det['class_name']
                score = det['confidence']
                bbox = det['bbox']
                ###############################################################
                
                self.get_logger().info(
                    f"  [{i}] {class_name} ({score:.2f}) | "
                    f"Center: ({bbox['cx']:.1f}, {bbox['cy']:.1f}), "
                    f"Size: {bbox['w']:.1f}x{bbox['h']:.1f}"
                )
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON string: {e}")

def main(args=None):
    ###############################################################
    # TODO 5: Initialize the ROS 2 Python client library
    # Hint: rclpy.init with necessary arguments (args)
    rclpy.init(args=args)
    ###############################################################
    node = YoloJsonSubscriber()
    
    try:
        ###############################################################
        # TODO 6: Spin the node so it stays alive and continues to trigger the timer
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
