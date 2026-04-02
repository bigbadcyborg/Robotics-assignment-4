# 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

## Programming Assignment: Milestone 4 (V1.05)

**Minhyuk Park and Tsz-Chiu Au**

### Introduction

In this fourth milestone, you will learn how to use YOLO, a real-time object detection algorithm based on a fully Convolutional Neural Network. YOLO’s input is a camera image fed in real time from a Raspberry Pi v2 camera, and the output would be a bounding box showing which part of the image belongs to which object, with a text label. You will find the tracking accuracy of YOLO to be acceptable, but you will find that labeling may not always be correct. This is because the generic YOLO model is trained on the Microsoft COCO dataset.

To do this, you will deploy our pre-configured Docker container that sets up all the software that is required for the assignment.
Please refer to the following video for an explanation of what a Docker container environment is. 

[https://www.youtube.com/watch?v=Gjnup-PuquQ](https://www.youtube.com/watch?v=Gjnup-PuquQ)

Robot Operating System version is associated with Ubuntu Long-Term Support Versions (e.g. Ubuntu 22.04 with Humble). We are using **ROS 2 Humble in a Docker environment** for Remote-PC. You might find the official tutorial on ROS 2 Humble useful in this course:

[https://docs.ros.org/en/humble/Tutorials.html](https://docs.ros.org/en/humble/Tutorials.html)

For all questions regarding milestone assignments and the robot, **you should contact the Doctoral Instructor Assistant via direct message on Slack**. Please do not contact the Instructor with questions regarding the milestone assignments. This is the URL for Slack for this course. 

<https://spring2026txstrobot.slack.com/>

Please head here for an introduction to the YOLO Computer Vision Object Detection Algorithm.
[https://docs.ultralytics.com/](https://docs.ultralytics.com/)

### Assignment requirement 

**Source Code Submission** is required for Milestone Assignment 4 on Canvas.

In addition, as usual, a **hardware video demonstration submission** is required for Milestone Assignment 4. 

You need to demonstrate that you have a working setup and can operate the turtlebot by making videos. This will also demonstrate that you have a working setup for working with a physical turtlebot. Refer to the demo requirement section at the end of the milestone assignment on what to include in the video.

**[SUBMISSION RULES]**

* **Individual Submission:** **Every team member must submit the video link(s) separately to Canvas.** If the video is duplicated within a team, that is acceptable; however, this ensures that only active participants who have access to the team’s recordings can receive credit. 

* **Standardized Hosting:** **To manage file sizes, do not upload raw video files (e.g., MP4) directly to Canvas.** Instead, **upload your videos to YouTube (set as "Unlisted")** and submit the links via a document.

### Video Demo Requirements

Your group will **record** one or more video clips. The estimated total length of the video clips is approximately three and a half minutes. **While you do not need to perform complex editing, please keep the total duration to a few minutes to ensure it remains concise.** One group member should narrate the video, explaining each step as it's performed. At the beginning of the first video clip, please show every group member's face and state the names of all group members.

Your recording setup should be organized to show all relevant windows at once: the terminal(s) used for launching nodes, the Gazebo simulation window, and the RViz visualization window.

You do not need to edit the videos, and uploading raw **footage** will suffice. You can split the demonstration into multiple videos **if necessary to show different parts of the requirement.** 

Refer to the demo requirement section at the end of the milestone assignment on what to include in the demo. Rules for robot usage will apply for working with the physical Turtlebot3. Please refer to the inventory list given to you separately.

> **Major Changes**
> * v 1.05  A lot of requested changes
> * v 1.0  Initial public release

---

### Objects Provided

In order to complete the demonstration requirements outlined below, we are providing you with a few objects. The objects provided are shown below.

* Bottle
* Bear Doll
* Computer Mouse

These object was chosen after testing with generic MS COCO trained yolo v11 on various angles and distances for the most stable inference results. You will find that under certain conditions, YOLO can detect but mislabel objects. You might have to consider multiple labels from YOLO’s COCO dataset to be a particular object of interest (e.g., a Teddy bear can be detected as a donut if the camera view is not just right. You can have some logic, such as if teddy bear and donut, this is a bear doll).

### Part 1: Object Detection on Jetson Demo with CUDA and TensorRT 

Part 1 will show you how to run YOLO, a popular object detection and localization algorithm on the Jetson NX. Jetson allows developers to start developing Artificial Intelligence applications on Edge devices with familiar tools present on a normal desktop environment for Nvidia GPUs.

To quote Nvidia, Jetson Xavier NX delivers up to 21 TOPS, making it ideal for high-performance compute and AI in embedded and edge systems. You get the performance of 384 NVIDIA CUDA® Cores, 48 Tensor Cores, 6 Carmel ARM CPUs, and two NVIDIA Deep Learning Accelerators (NVDLA) engines. Combined with over 59.7GB/s of memory bandwidth, video encode, and decode, these features make Jetson Xavier NX the platform of choice to run multiple modern neural networks in parallel and process high-resolution data from multiple sensors simultaneously.

This assumes that you have a working setup from Milestone Assignment 1 Part 1. Please execute all instructions with **[Remote PC]** on Docker shell. Note that you have to enable GUI and start the Docker container by following instruction from Milestone Assignment 1. Please execute all instructions with **[Turtlebot Nvidia Jetson]** on Turtlebot Jetson's native bash shell without Docker.

The following instructions are from our own and not from online sources. Please follow carefully and ask the IA for assistance should you face any problems.

1. **[Turtlebot Nvidia Jeston]** On Jetson, you can look at CPU and GPU usage in real-time by using the following command on a separate terminal window. CPU will show real-time CPU load, and GR3D_FREQ will give you an idea for real-time GPU load. You can run this on another terminal window.
```bash
tegrastats

```


2. **[Turtlebot Nvidia Jeston]** While connected to the internet, git clone and download the demo in Turtlebot's Nvidia Jetson. 

```bash
cd ~/
git clone https://github.com/dmz44/Robotics_Assignment_4.git

```


3. **[Turtlebot Nvidia Jeston]** Go to the folder in a new terminal window
```bash
cd ~/Robotics_Assignment_4/Assignment_4_demo/CUDA_Demo
```


4. **[Turtlebot Nvidia Jeston]** Run yolo v11 demo and point the camera in front of the robot at various objects. You should see real-time bounding boxes and class labels. **It is highly recommended for you to do this step every time before coding your custom scripts involving Jetson's camera to verify the operation of the hardware.**
```bash
python3 yolov11_demo.py

```


**[Optional][Turtlebot Nvidia Jeston]** YOLO has different-sized models with different computational requirements for real-time performance. We encourage you to experiment with different models for each script by replacing the names for engine and pt with different model names corresponding to size of the models. Listed from smallest to largest, these models trade inference speed and accuracy. After you replace the names, the Ultralytics package will automatically download and generate relevant files for you. Note that your frame per second would be bottlenecked by the camera’s sensor mode.

The following mentions the relevant code section. Please change the names of both the pt and engine files.

```python
ENGINE = "yolo11n.engine"
PT     = "yolo11n.pt"

```

**YOLO v11 base**

```text
yolo11n.pt    yolo11n.engine
yolo11s.pt    yolo11s.engine
yolo11m.pt    yolo11m.engine
yolo11l.pt    yolo11l.engine
yolo11x.pt    yolo11x.engine

```

---

### Part 2: Programming a publisher and a subscriber for YOLO

In order to complete part 3, you need to learn how to use ROS 2 to command the turtlebot using your own program. 

Take a look at the provided publisher and subscriber script for YOLO in the git repository. We are giving you sample codes, one for the subscriber and one for the publisher, to get you started. 

Onvr you have finished coding your publisher and subscriber by completing the skeleton code, execute the example code for the publisher on the Turtlebot Jetson, and execute the example code for the subscriber either on the Turtlebot Jetson or the Remote PC to test the functionality of the code.

Here is what you can expect from the completed skeleton code. **Note that launching the publisher for the first time would take a lot of time, since it needs to download the weights from the internet.** You can disconnect from the internet and connect back to small_blue_wifi once you are done downloading the weights from the initial run. 

**Publisher(Option1:CUDA)**
```
[INFO] [1774133694.824883296] [yolo_json_publisher]: Loading YOLOv11 model on CUDA...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...
100%|██████████████████████████████████████| 5.35M/5.35M [00:00<00:00, 9.65MB/s]
[INFO] [1774133708.662359424] [yolo_json_publisher]: Initializing CSI Camera...
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3280 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 3280 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: Running with following settings:
   Camera index = 0
   Camera mode  = 4
   Output Stream W = 1280 H = 720
   seconds to Run	= 0
   Frame Rate = 59.999999
GST_ARGUS: Setup Complete, Starting captures for 0 seconds
GST_ARGUS: Starting repeat capture requests.
CONSUMER: Producer has connected; continuing.
[ WARN:0@42.505] global /home/nvidia/workspace/opencv-4.6.0/modules/videoio/src/cap_gstreamer.cpp (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1

```
**Publisher(Option2:TensorRT_Optimized)**
```
[INFO] [1774134507.980644608] [yolo_json_publisher]: Loading YOLOv11 model on CUDA...
WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.
[INFO] [1774134507.984752448] [yolo_json_publisher]: Initializing CSI Camera...
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3280 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 3280 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: Running with following settings:
   Camera index = 0
   Camera mode  = 4
   Output Stream W = 1280 H = 720
   seconds to Run	= 0
   Frame Rate = 59.999999
GST_ARGUS: Setup Complete, Starting captures for 0 seconds
GST_ARGUS: Starting repeat capture requests.
CONSUMER: Producer has connected; continuing.
[ WARN:0@9.734] global /home/nvidia/workspace/opencv-4.6.0/modules/videoio/src/cap_gstreamer.cpp (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Loading yolo11n.engine for TensorRT inference...
[03/21/2026-18:08:39] [TRT] [I] Loaded engine size: 13 MiB
[03/21/2026-18:08:39] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[03/21/2026-18:08:45] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +342, GPU +237, now: CPU 717, GPU 4728 (MiB)
[03/21/2026-18:08:45] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +14, now: CPU 0, GPU 14 (MiB)
[03/21/2026-18:08:46] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 704, GPU 4730 (MiB)
[03/21/2026-18:08:46] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +142, now: CPU 0, GPU 156 (MiB)


```
**Subscriber**
```
nvidia@nvidia-desktop:~$ python3 sub.py
[INFO] [1774133827.058645760] [yolo_json_subscriber]: Listening for YOLO JSON detections...
[INFO] [1774133827.062601856] [yolo_json_subscriber]: Received 1 detections at t=1774133826.99 [camera_link]:
[INFO] [1774133827.064806720] [yolo_json_subscriber]:   [0] book (0.32) | Center: (689.2, 239.0), Size: 474.2x196.6


```
### Part 3: Programming a Custom Script for Physical Turtlebot3 Bottle Pick and Place

In part 3, you need to extend your teleoperation code in Assignment 3 and ROS2 subscriber code in Assignment 4 to make the turtlebot autonomously detect, approach, and pick up and place the bottle from and to the ground. We recommend having a single custom script for part 3 of the assignment.

In order to complete part 3, you need to learn how to use ROS 2 to command the turtlebot using your own program. 

To turn the turtlebot on its place, apply angular velocity in the Z axis so that the turtlebot can turn left or right.

To move the turtlebot forward and backward, apply linear velocity so that the turtlebot can move forward or backward.

For the turtlebot to react to the camera input, you would need to subscribe to the YOLO output from the publisher you coded in Part 2, and apply control inputs to the turtlebot so that the target object, the bottle, stays at the center of the camera's field of view. Not only do you have to adjust the rotation of the turtlebot, but you would also need to drive forward and backward so that the detection box in pixel coordinates is in a certain range that you determined from your own experimentation. 

To pick up the bottle with the robot arm, move the robot arm to home pose, open the gripper, extend the arm forward, close the gripper, and retract the arm to the home position. Do the reverse for placing the bottle on the ground.

---

### Demo Requirements (3.5-4 Minute Demonstration) - 100 points

Please refer to the video submission requirements in the introduction.

Your submission must include two items: **links to the video file and a single .zip file containing all of your source code**.

**Part A: YOLOv11 Demonstration on Jetson GPU -10 points**
This part demonstrates your ability to run various YOLOv11 models on the Jetson's GPU using a live camera feed.

* **Object Detection (YOLOv11):** Run the standard YOLOv11 detection model. Point the camera at several different objects (e.g., a person, a bottle, a laptop) and show the model drawing accurate bounding boxes and class labels in real-time with a given demo code.

* **Prove CUDA Usage:** While the YOLOv11 models are running, open a new terminal and run `tegrastats`. You must show the output of `tegrastats`, which should indicate that the CUDA cores are active and under load, confirming that inference is running on the CUDA cores.

**Part B: YOLOv11 Publisher and Subscriber Demonstration-20 points**

* **Publisher and Subscriber (YOLOv11):** Run the completed sample code for YOLOv11 publisher and subscriber. Point the camera at several different objects (e.g., a person, a bottle, a laptop) and show that two different programs can exchange necessary information over ROS2.

**Part C: ROS2 Control Demonstration With Physical Turtlebot - 70 points**

This part showcases your ability to integrate the given ROS2 code for a robotic task.

* a): Demonstrating Visual Servoing to a Bottle (Task 1) -20 points

For part C's a), you need to code a visual servoing code that turns the wheels of the Turtlebot 3 so that the front of the robot faces the target object, a bottle.

Demonstrate your code by moving the bottle slowly in front of the robot. The robot should "lock onto" the bottle and attempt to keep the subject at the center of the camera's field of view using the robot's wheels to turn left and right.

* b): Pick the bottle with Turtlebot (Task 1) -30 points

For part C's b), you need to extend a)’s code so that the Turtlebot 3 faces the bottle, drives towards the bottle, extends the arm, grabs onto the bottle, and lifts it off the ground. You can achieve this by publishing twist messages to control the wheels and using an action server and client to control the arm in the joint space. To test, place the bottle on the ground in the field of view of the Turtlebot's camera, but slightly off-center. 


* c): Pick and Place with Turtlebot (Task 2) -20 points

For part C's c), you need to extend b)’s code so that the Turtlebot 3 returns to the base you define after picking up the bottle, and release the bottle on the ground at the home base near the robot's starting location. To test, refer to part C's b). 

---

### Appendix

#### [Optional] Running YOLO v11 on Your Desktop Machine

You can run YOLO v11 with either CPU only or CUDA on your own desktop machine with any architecture, with the following setup and modifications. This capability might help you with offloading some of the development for this assignment to your own machine. However, while this instruction was tested on our machines, we would not offer official support for you running YOLO on your own machine. We would also not provide ways to hook up camera feed from Gazebo’s simulated turtlebot onto YOLO v11.

Recommended reading: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

**[Your PC]** It is not recommended to install YOLO in a Python virtual environment, such as Conda, if you want it to work well with ROS2 or other binary programs installed through apt-get. This is the primary reason why we did not install a Python virtual environment on Jetson.

You can deactivate your Python virtual environment temporarily and install Ultralytics packages if you need your virtual environment.

**[Your PC]** Install GPU drivers such as CUDA and ROCm, Pytorch, and TorchVision. When installing PyTorch and TorchVision, please follow the exact instructions on the official website. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) .

**[Your PC]** On your terminal, install the Ultralytics package and OpenCV

```bash
pip install ultralytics opencv-python

```

**[Your PC]** Here is an example YOLO v11 Python code adapted to a desktop environment. We have changed cap interface to webcam for you. Replace `PT_MODEL` with “yolov11-seg.pt” or “yolov11-pose.pt” to convert it to seg or pose models. Refer to above for full list of model names and associated .pt names.

**Desktop YOLO v11 demo code on CPU or CUDA**

```python
import cv2
import time
from ultralytics import YOLO
import numpy as np

def desktop_demo():

    PT_MODEL = "yolov11n.pt"  # Change this line for different models
    model = YOLO(PT_MODEL)
   
    # Capture through webcam
    cap = cv2.VideoCapture(0) # Replace with the name of a photo or video if needed
    if not cap.isOpened():
        print("Error: unable to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame)  
        end = time.time()
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        fps = 1 / (end - start)
        cv2.putText(annotated_frame,
                     f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Desktop Demo", annotated_frame)
        
        if cv2.waitKey(1) == 27: #ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desktop_demo()

```

#### [Optional Reading] Platform Specific Optimizations on Jetson for Real-time Performance

You might have noticed that the provided code is different from the normal code that you are used to running in other computer science classes. Jetson is an edge platform that requires optimization for real-time performance and requires some specialized ways to perform everyday functions.

**Camera connection for Jetson**
The first major difference is how we connected the camera to the Jetson. Instead of using a USB connection to a normal webcam, we use the MIPI CSI (Camera Serial Interface) port on the Jetson to connect to a camera at a lower level. The CSI port is a direct, high-bandwidth connection to the Jetson's dedicated Image Signal Processor (ISP).

By using the CSI port, the raw image data bypasses the slow, general-purpose USB controller and goes straight into the ISP. The ISP handles critical pre-processing tasks like de-Bayering (converting raw sensor data to a color image), color correction, and noise reduction using dedicated hardware. This frees the CPU from doing any of this initial heavy lifting and, most importantly, achieves lower photon-to-response latency.
In practice, this changes how the camera is accessed through code. The gst string in your code is actually a command that builds a hardware-accelerated video processing pipeline using GStreamer. More specifically, we use nvarguscamerasrc, which is a GStreamer plugin specifically designed for cameras connected to the MIPI CSI (Camera Serial Interface) port on the Jetson.

Let us break down the pipeline illustrated in gst string in more detail. As mentioned, nvarguscamerasrc captures video from the CSI/ISP hardware. video/x-raw(memory:NVMM) is arguably the most important performance trick in the pipeline. NVMM (NVIDIA Managed Memory) tells the system to keep the video frame entirely within the GPU/SoC's memory space. The frame is never copied to the CPU's main RAM. This concept, often called "zero-copy," is a massive performance win because moving large video frames between CPU and GPU memory is a slow, performance-killing bottleneck. nvvidconv is a hardware-accelerated video converter. Tasks like resizing the video (e.g., from 1080p to 480p) or changing the color format (e.g., from the camera's native format to the BGR format needed by OpenCV) are performed by the Jetson's dedicated video engine, not the CPU.

Therefore, by making Jetson-specific modifications to the Python code, your Python code can have all the capturing, pre-processing, and format conversion already been done efficiently in hardware. This means that by the frame is available from appsink, the frame would already be waiting in a GPU-accessible memory location for your use.

**Camera Sensor Modes for Raspberry Pi v2 camera**
The Raspberry Pi Camera Module v2 (IMX219) has several sensor modes available on the NVIDIA Jetson Xavier NX, but they are often grouped into a few commonly used configurations accessible through the driver. The specific modes you can use depend on the camera driver (like nvarguscamerasrc for GStreamer) and the software you're using. Using different sensor modes allows you to access outputs of the camera sensor with different frame rate, field of view and bit depth. 

On Jetson, the following modes are accessible for user’s applications.

* **Mode 0 (3264x2464 @ 21 FPS):** This is the full-resolution mode, capturing the entire 8-megapixel sensor area. It's best for high-quality still images but has a lower frame rate.
* **Mode 2 (1920x1080 @ 30 FPS):** This is a standard Full HD video mode. It uses the full sensor width and then crops or scales it down, providing a wide field of view. This is a very common mode for video applications.
* **Mode 4 (1280x720 @ 120 FPS):** This is the high frame rate mode, ideal for slow-motion capture. It achieves this speed by using 2x2 binning on the sensor, which groups pixels together. This results in a "partial" or cropped field of view compared to the other modes.

**OpenCV for Jetson**

While the OpenCV you are using is similar to the OpenCV you are using for other computer science classes, we needed to compile from source to exploit the hardware on the Jetson. This is because a binary install of OpenCV from pip is CPU-only and is not configured well.
In particular, you need to enable particular sets of plugins during source compilation. We leave some of them here. A good optional exercise for the reader is to find out what each flag does and how it fits within our robot assignment. Note: Jetson Xavier series has 7.2 for CUDA_ARCH_BIN.

```bash
cmake \
…
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN='7.2' \ 
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=1 \
-D WITH_GSTREAMER=ON \
-D WITH_LIBV4L=ON \
-D ENABLE_NEON=ON \
…

```

**Jetson AI Inference and TensorRT Exploitation**

Let's now look at part of the code dealing with .pt file and .engine file.

A .pt file is a standard model from the PyTorch framework. It's flexible and contains the model's architecture and weights. When you run it, the PyTorch framework interprets this file to perform the calculations. It's portable but not optimized for any specific hardware.
An .engine file is the result of taking the .pt blueprint and compiling it with NVIDIA TensorRT SDK. A wrapper from Ultralytics takes care of the specifics in utilizing TensorRT. TensorRT is an optimizer that aggressively modifies the model to run as fast as possible on a specific NVIDIA GPU—in this case, your Jetson's integrated GPU.

This "export" step performs several key optimizations using the TensorRT SDK:

* **Layer Fusion:** It combines multiple simple AI layers (e.g., a convolution, a bias, and an activation) into a single, highly efficient custom operation. By fusing multiple computational operations into a single operation, we can achieve higher inference speed. We also set up so that during the inference process, we exploit Nvidia’s Tensor Cores, a dedicated part of the silicon that supports operations such as Fused Multiply Add, a common operation during AI inference.
* **Tensor Cores:** Suggested Reading: [https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
* **Precision Calibration:** Also known as quantization, it can convert calculations from slow 32-bit numbers to much faster 16-bit (FP16) or 8-bit (INT8) numbers, often with little to no loss in accuracy.
* **Kernel Auto-Tuning:** It selects the fastest possible algorithms for your specific Jetson GPU.

In order to do this, you need some setup in addition to the Jetpack software that contains the TensorRT SDK. That part was done by the IA.

**Pytorch and TorchVision for Jetson**

The pre-compiled versions of PyTorch available on the standard Python Package Index (PyPI) are built for desktop computers, which almost always use an x86_64 CPU architecture (from Intel or AMD). The NVIDIA Jetson, however, uses an ARM64 CPU architecture, similar to what's in modern smartphones or Apple Silicon Macs. Therefore, NVIDIA provides PyTorch wheels (.whl files) that have been specifically compiled for the Jetson's ARM64 architecture so they can run correctly.

You also need to ensure compatibility with Jetpack. The Jetson's operating system, a customized Ubuntu operating system called JetPack, includes specific versions of CUDA, cuDNN, and TensorRT that are tailored for its mobile, power-efficient GPU. NVIDIA’s custom PyTorch builds are specifically compiled and linked against these exact libraries. This ensures a stable and high-performance bridge between the PyTorch framework and the GPU hardware. Also, during compilation, NVIDIA enables specific flags and optimizations that take advantage of the unique features of the Jetson's GPU, which are different from a desktop GPU like an RTX 4090.

Therefore, when installing PyTorch and TorchVision for Jeston, you need to match your Python version and your specific Jetpack version. This effectively means you would be installing an older version of Pytorch and TorchVision on Jetson, and problems with that, such as an older version of Numpy. You need to make sure which version works with which version when working with Jetson. That part was done for you by the IA.

---

## Practical Notes: Running Part 2 Publisher with Remote-PC + Jetson Connectivity

If your TurtleBot only has reliable connectivity to the Remote PC, run the Part 2 publisher on Jetson over SSH.

### Where to run each script

- **Part 2 YOLO publisher:** Run on **TurtleBot Jetson** (needs CSI camera + Jetson GPU).
- **Part 3 autonomous subscriber/controller:** Can run on Jetson or Remote PC (with ROS 2 network configured).
- **Part 3 visualizer:** Can run where a display + camera access is available.

### Copy and run Part 2 publisher over SSH

1. Copy your publisher script from Remote PC to Jetson:

```bash
scp your_part2_publisher.py nvidia@<jetson_ip>:~/
```

2. SSH into Jetson:

```bash
ssh nvidia@<jetson_ip>
```

3. Source ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

4. Run publisher script:

```bash
python3 ~/your_part2_publisher.py
```

### CUDA vs TensorRT for Part 2 publisher

- **CUDA (`.pt`)**: easiest for bring-up and debugging.
- **TensorRT (`.engine`)**: usually faster once engine file is built and compatible.

Recommended workflow: start with CUDA, then switch to TensorRT after your full pipeline is stable.

### Topic expectations

The Part 2 publisher should publish JSON `String` messages to:

```text
/yolo/detections_json
```

Then both Part 3 nodes can subscribe to the same topic.

### Note on `CUDA_Demo` scripts

The scripts under `Assignment_4_demo/CUDA_Demo/` are primarily camera/YOLO demos and visualization examples. They do not automatically replace your Part 2 ROS publisher unless you explicitly add ROS publishing logic.
