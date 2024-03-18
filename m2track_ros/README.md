# ROS Point Cloud Tracker

This package uses a Deep Neural Network to track a target in the point cloud. Especifically, it uses the M2-Track from [Open3DSOT](https://github.com/Ghostish/Open3DSOT), but it can work with others models. For using other models, adapt the [inference module](src/inference/__init__.py). Because of the different versions of software for Machine Learning, it's recommended to use the docker image available in this repository.

It uses the message received from `/target` to initialize the target object. Pressing the key **s**, the tracker is reset for selecting a new target. Pressing the key **t**, it prints the average computational time of the network prediction and the preprocessed time of reconstructing the point cloud from the depth image.

## Installation of the Docker image.
1. Clone the repository [aurova_detections](https://github.com/AUROVA-LAB/aurova_detections).
2. Build the Dockerfile
	```	
	cd /path/to/aurova_detections/m2track_ros/
	docker build -t aurova_pc_tracker .
	```

## Use of the Docker Image
- This Docker image will execute the launch file `m2track_ros.launch` directly. You can change the last line of the Dockerfile and rebuild it to change this behaviour.

- For using the Docker image:
	```
	sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --net host -it aurova_pc_tracker
	```

