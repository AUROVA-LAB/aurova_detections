# aurova_detections


## yolo_ros_ouster
This package implements *deep learning* methods to track pedestrians using images reconstructed from LiDAR point clouds. It implements two trackers, one based on YOLO and other in DaSiamRPN. 

## m2track_ros
This package implements the point cloud tracker $M^2$-Track in ROS.

## tracker_filter
This package fuses the predictions of the trackers implemented in *yolo_ros_ouster* and *m2track_ros* using an Extended Kalman Filter. It also publish a point cloud without the target, so the [local planner](https://github.com/AUROVA-LAB/aurova_planning) can follow the target without considering it an obstacle.

## ✨ License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0) and is intended for non-commercial academic use. If you are interested in using the software for commercial purposes, don't hesitate to get in touch with us via [email](mailto:miguelangel.munoz@ua.es).
