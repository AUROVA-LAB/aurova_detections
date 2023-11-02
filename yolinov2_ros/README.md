# yolinov2 Docker

## How to run it?

Building the docker 
```
sudo docker build -t yolinov2 .
```

Run the docker (for example's user: mice85)
```
sudo docker run --shm-size=6gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus "all" --rm -it -u root --name yolinov2_kitti --net host -v /home/mice85/docker_shared:/docker_shared yolinov2_kitti
```

Inside the docker, run one of these command to train, detect, or test:

```
python3 ${HOME}/ros_ws/src/yolinov2/train.py [path] [bool_load_epoch] [num_epoch] [height] [width]
```

```
python3 ${HOME}/ros_ws/src/yolinov2/detect.py  [path] [num_epoch]
```

```
python3 ${HOME}/ros_ws/src/yolinov2/test.py [path] [num_epoch] [num_test_image]
```

An example for training:
```
python3 ${HOME}/ros_ws/src/yolinov2/train.py /docker_shared/yolinov2_shared/experiments/exp_2023-10-17/ true 149 128 288 
```

The data structure for experiment_path:
### File tree
└── ~/experiment_path   
　└── exp_2023-10-17     
　　├── epochs   
　　├── out   
　　├── train  
　　├── train_masks   
　　├── val   
　　└── val_mask   

**Note**: This structure is an example for a experiment in the date 2023-10-17.