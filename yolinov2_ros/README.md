# yolinov2 Docker

Ground boundary detector from LiDAR front-view images. 

**Input:** Using liDAR front-view image topics (depth and reflectance), the script [merge_channels_ouster.py](https://github.com/AUROVA-LAB/aurova_preprocessed/tree/master/merged_channels_ouster) can generate the input for the docker, i.e., a merged image.

**Output:** The output is a ROS topic that contains a road boundaries mask. 

Please see an [example](https://github.com/AUROVA-LAB/applications/tree/main/app_geo_localization) in the context of a localization application.

## How to run it?

Building the docker 
```
sudo docker build -t yolinov2 .
```

Run the docker (for example's user: mice85)
```
sudo docker run --shm-size=6gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus "all" --rm -it -u root --name yolinov2 --net host -v /home/mice85/docker_shared:/docker_shared yolinov2
```

Inside the docker, run one of these command to train, detect, or test:

```
python3 ${HOME}/ros_ws/src/yolinov2/train.py [experiment_path]* [num_epoch] [height] [width]  <!-- (if num_epoch == 0, random weights) -->
```

```
python3 ${HOME}/ros_ws/src/yolinov2/detect.py  [experiment_path]* [num_epoch]
```

```
python3 ${HOME}/ros_ws/src/yolinov2/test.py [experiment_path]* [num_epoch] [num_image]
```

An example for training from epoch 149:
```
python3 ${HOME}/ros_ws/src/yolinov2/train.py /docker_shared/yolinov2_shared/experiments/exp_2023-02-13/ 149 128 2048 
```

## An example for detections:

**Step 1:** Download the example [weights](https://drive.google.com/file/d/1iw4oEDDFjOqoGpUJpzZMV19_sxwy2lAI/view?usp=sharing) and put them in your home directory (/home).

**Step 2:** Enter the docker and run the following comand:
```
python3 ${HOME}/ros_ws/src/yolinov2/detect.py /docker_shared/yolinov2_shared/experiments/exp_2023-02-13/ 149 
```
**Step 3:** Run [merge_channels_ouster.py](https://github.com/AUROVA-LAB/aurova_preprocessed/tree/master/merged_channels_ouster). Alternatively, it is possible to provide context through the following [example](https://github.com/AUROVA-LAB/applications/tree/main/app_geo_localization).

**Step 4:** Download the example [rosbag](https://drive.google.com/file/d/1oW7MLIJhvlNtgJsetXNRY-BQxufgPUoJ/view?usp=sharing) and play it. Alternatively, it is possible to provide context through the following [example](https://github.com/AUROVA-LAB/applications/tree/main/app_geo_localization).

*The data structure for experiment_path:
### File tree
└── ~/experiment_path   
　└── exp_2023-02-13     
　　├── epochs   
　　├── out   
　　├── train  
　　├── train_masks   
　　├── val   
　　└── val_mask   

**Note**: This structure is an example for a experiment in the date 2023-02-13.