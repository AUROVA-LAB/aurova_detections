# ROS YOLOv5 detector and tracker

This package uses the Deep Neural Network YOLOv5 to detect persons with the images obtained by a 3D LIDAR (Ouster OS1), with the channels signal, reflec, nearir and range. The tracker uses this neural network to detect the persons but also DaSiamRPN tracker of OpenCV, which is a Siamese Neural Network. We create a Docker to use these neural networks with CUDA support.

## Installation of the Docker image.
1. Clone the repository [aurova_detections](https://github.com/AUROVA-LAB/aurova_detections).
2. The Docker is based on a image of [NVIDIA NGC](https://catalog.ngc.nvidia.com/). You need to create an account if you haven't one. When you are logged, click in your user (top right corner) and go to "Setup". Download and install the NGC CLI and then follow the instructions of "Generate API Key".
3. Build the Dockerfile
	```	
	cd /path/to/aurova_detections/yolo_ros_ouster/
	docker build -t aurova_image_tracker .
	```

## Use of the Docker Image
- This Docker image will execute the launch file `tracker.launch` directly. You can change the last line of the Dockerfile and rebuild it to change this behaviour.

- For using the Docker image:
	```
	sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --net host -it aurova_image_tracker
	```

- To change parameters of the tracker or detector, yo can edit the launch files, remember to rebuild the docker to update the files. Some interesting parameters are: 
	- weights
	- data
	- confidence_threshold
	- input_image_topic
	- output_topic_(yolo,dasiamrpn)
	- output_image_topic
	- threshold_tracker
	- select_time

- The output of the DaSiamRPN tracker is published in the topic `/tracker/dasiamrpn` and the one of YOLO tracker in `/tracker/yolo`.

- The policy to select the target is that the person have to be in front of the robot for more than `select_time` seconds. 

- The result image, published in the topic `/tracker/image_out` will show two lines delimiting the search area when it is selecting a target otherwise the boundig box of the target. The blue box is the one obtained by DaSiamRPN and the red one is that YOLO detect the target. Pressing the key **s**, the tracker is reset for selecting a new target. Pressing the key **t**, it prints the average computational time of YOLO tracker and the DaSiamRPN tracker.
