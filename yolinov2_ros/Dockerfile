# This is an auto generated Dockerfile for ros:desktop-full
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ros:jazzy

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop-full \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

RUN echo "ubuntu:ubuntu" | chpasswd

# Install pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

USER ubuntu
RUN mkdir -p /home/ubuntu/ros_ws/src

# Initialize the workspace Delete this before commit
RUN cd /home/ubuntu/ros_ws/src && mkdir src

# Install deepsort
RUN python3 -m pip config set global.break-system-packages true
RUN pip3 install gdown easydict 
RUN pip3 install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126

# Install segmentation models
RUN pip3 install segmentation-models-pytorch

# Create package
RUN cd /home/ubuntu/ros_ws/src && mkdir yolinov2
USER root
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*
COPY . /home/ubuntu/ros_ws/src/yolinov2/
RUN cd /home/ubuntu/ros_ws/src/yolinov2/yolinov2 && chmod +x detect.py && \
    chmod +x test.py && \
    chmod +x train.py && \
    chmod +x utils.py && \
    chmod +x models.py && \
    chmod +x dataset.py

USER ubuntu
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
## Compile the package
RUN cd /home/ubuntu/ros_ws/ && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
CMD /bin/bash -c "source /home/ubuntu/ros_ws/install/setup.bash && ros2 run yolinov2 detect /home/ubuntu/ros_ws/src/yolinov2/exp_2023-02-13/ 149"
# CMD python3 /home/ubuntu/ros_ws/src/yolinov2/yolinov2/train.py [path] [bool_load_epoch] [num_epoch] [height] [width]
# CMD python3 /home/ubuntu/ros_ws/src/yolinov2/yolinov2/test.py [path] [num_epoch] [num_test_image]

