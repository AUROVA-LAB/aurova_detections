# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx
RUN apt purge -y python3.8 && rm -rf /opt/conda
RUN apt install -y python3.8
RUN apt install -y python-is-python3
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx python3-pip git

# # Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install coremltools onnx gsutil notebook wandb>=0.12.2
# RUN pip install --no-cache -U torch torchvision numpy Pillow
# RUN pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Create working directory
# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app

# # Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

ENV DEBIAN_FRONTEND noninteractive

# ROS installlation
RUN apt-get update && apt-get install -y curl 
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base ros-noetic-cv-bridge\
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

# Now create the same user as the host itself
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ros
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} ros
RUN usermod -a -G dialout ros
RUN mkdir config && echo "ros ALL=(ALL) NOPASSWD: ALL" > config/99_aptget
RUN cp config/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

RUN mkdir -p /home/ros_ws/src
# Initialize the workspace
RUN pip install empy catkin_pkg
RUN cd /home/ros_ws/src && source /opt/ros/noetic/setup.bash && catkin_init_workspace
RUN cd /home/ros_ws && source /opt/ros/noetic/setup.bash && catkin_make
COPY ./update_bashrc /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# # Install yolov5
RUN cd /home/ros_ws/src/ && git clone https://github.com/EPVelasco/detection_msgs.git && git clone --recurse-submodules https://github.com/EPVelasco/yolov5_ros.git && cd yolov5_ros/src/yolov5 && pip install -r requirements.txt && cd /home/ros_ws/ 
# # Compile the package
RUN cd /home/ros_ws/ && source /opt/ros/noetic/setup.bash && catkin_make

# # Copy files from host to docker
COPY ./yolov5.launch /home/ros_ws/src/yolov5_ros/launch/
COPY ./upsampling.py /usr/local/lib/python3.8/dist-packages/torch/nn/modules/
COPY ./Arial.ttf /home/.config/Ultralytics/
COPY ./yolo.py /home/ros_ws/src/yolov5_ros/src/yolov5/models/
COPY ./detect.py /home/ros_ws/src/yolov5_ros/src/
COPY ./merge_channels_ouster.py /home/ros_ws/src/yolov5_ros/src/
COPY ./CMakeLists.txt /home/ros_ws/src/yolov5_ros/

# # Make program executable
RUN cd /home/ros_ws/src/yolov5_ros/src/ && chmod +x detect.py
RUN cd /home/ros_ws/src/yolov5_ros/src/ && chmod +x merge_channels_ouster.py

# Install OpenCV with CUDA support
RUN pip uninstall -y opencv-python
RUN apt update && apt-get -y install build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libblas-dev liblapack-dev gfortran \
    python3-dev

RUN cd ~ && wget -O opencv-4.5.5.zip https://github.com/opencv/opencv/archive/4.5.5.zip
RUN cd ~ && unzip -q opencv-4.5.5.zip && mv opencv-4.5.5 opencv && rm -f opencv-4.5.5.zip 

RUN cd ~ && wget -O opencv_contrib-4.5.5.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip && unzip -q opencv_contrib-4.5.5.zip
RUN cd ~ && mv opencv_contrib-4.5.5 opencv_contrib && rm -f opencv_contrib-4.5.5.zip
RUN cd ~/opencv && mkdir build
RUN cd ~/opencv/build/ && cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D PYTHON3_EXECUTABLE=/usr/bin/python \
      -D PYTHON_INCLUDE_DIR=/usr/include/python3.8/ \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.8/dist-packages/numpy/core/include/ \
      -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.8/dist-packages/ \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11 \
      -D OpenCL_LIBRARY=/usr/local/cuda-11/lib64/libOpenCL.so \
      -D OpenCL_INCLUDE_DIR=/usr/local/cuda-11/include/ \
      ..
RUN cd ~/opencv/build/ && make -j7 
RUN cd ~/opencv/build/ && make install 
RUN cd ~/opencv/build/ && ldconfig
RUN rm -rf ~/opencv*

#CMD export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && cd ./../home/ignacio/ros_ws/ && /bin/bash -c "source devel/setup.bash; catkin_make; source devel/setup.bash; roslaunch yolov5_ros yolov5.launch"