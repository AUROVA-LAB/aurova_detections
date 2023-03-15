# ROS YOLOv5 detector and tracker

This package uses the Deep Neural Network YOLOv5 to detect persons with the images obtained by a 3D LIDAR (Ouster OS1), with the channels signal, reflec, nearir and range. The tracker uses this neural network to detect the persons but also DaSiamRPN tracker of OpenCV, which is a Siamese Neural Network. We create a Docker to use these neural networks with CUDA support.

## Installation of the Docker image.
1. Download this package
2. The Docker is based on a image of [NVIDIA NGC](https://catalog.ngc.nvidia.com/). You need to create an account if you haven't one. When you are logged, click in your user (top right corner) and go to "Setup". Download and install the NGC CLI and then follow the instructions of "Generate API Key".
3. Build the Dockerfile
	```	
	cd /path/to/aurova_detections/yolo_ros_ouster/
	docker build -t yolo_ros_ouster
	```

## Use of the Docker Image
- This Docker image will execute the launch file `tracker.launch` directly. You can change the last line of the Dockerfile and rebuild it to change this behaviour.

- For using the Docker image:
	```
	sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --net host -it yolo_ros_ouster
	```

- To change parameters of the tracker or detector, yo can edit the launch files, remember to rebuild the docker to update the files. Some interesting parameters are: 
	- weights
	- data
	- confidence_threshold
	- iou_threshold
	- inference_size_h
	- inference_size_w
	- input_image_topic
	- output_topic
	- publish_image
	- output_image_topic

- The 4 channels are merged and publish in `/ouster_merged` topic.

- The detector (yolov5.launch) publish an image with the bounding boxes of the detected persons in the output_image_topic, by default `/yolov5/image_out`. The bounding boxes are also published in the output_topic, by default `/yolov5/detections`. This type of messages are defined in the package[detection_msgs](https://github.com/EPVelasco/detection_msgs), for using them outside the docker install the package.

- In the same way, the tracker (tracker.launch) publish the result image and the bounding box of the targe (it will be 2 if the target is on the limits of the image, as it is a 360 degrees image). The policy to select the target is that the person have to be in front of the robot for more than 5 seconds. The result image will show two lines delimiting the seach area when it is selecting a target otherwise the boundig box of the target. The blue box is the one obtained by DaSiamRPN, the red one is that YOLO detect the target and it have and intersection with the previous DaSiamRPN box and when it loses the targe and find it again the box is green.
