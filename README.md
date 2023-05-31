# aurova_detections


## yolo_ros_ouster
This package implements *deep learning* methods to track pedestrians using images reconstructed from LiDAR point clouds. It implements two trackers, one based on YOLO and other in DaSiamRPN. 

## m2track_ros
This package implements the point cloud tracker $M^2$-Track in ROS.

## tracker_filter
This package fuses the predictions of the trackers implemented in *yolo_ros_ouster* and *m2track_ros* using an Extended Kalman Filter. It also publish a point cloud without the target, so the [local planner](https://github.com/AUROVA-LAB/aurova_planning) can follow the target without considering it an obstacle.