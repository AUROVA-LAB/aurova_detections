# check=skip=JSONArgsRecommended
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM ros:jazzy

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y python3.12
RUN apt install -y python-is-python3
RUN apt install -y zip htop screen libgl1 libglx-mesa0 python3-pip git

# # Install python dependencies
RUN echo "ubuntu:ubuntu" | chpasswd

USER ubuntu
RUN python3 -m pip config set global.break-system-packages true
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install coremltools onnx gsutil notebook wandb
# RUN pip install --no-cache -U torch torchvision numpy Pillow

# Create working directory
# RUN mkdir -p /usr/src/app
# WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app

# # Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

RUN mkdir -p /home/ubuntu/ros_ws/src/
RUN cd /home/ubuntu && git clone https://github.com/AUROVA-LAB/aurova_detections.git -b ros2
RUN mv /home/ubuntu/aurova_detections/detection_msgs /home/ubuntu/ros_ws/src/detection_msgs && rm -r /home/ubuntu/aurova_detections

RUN pip3 install torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126

# Install OpenCV with CUDA support
RUN pip uninstall -y opencv-python
USER root
RUN apt update && apt-get -y install build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libblas-dev liblapack-dev gfortran \
    python3-dev
RUN apt-get -y install wget 
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
RUN mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.0-560.28.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.0-560.28.03-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get -y install cuda-toolkit-12-6
RUN cd ~ && wget -O opencv-4.14.0.zip https://github.com/opencv/opencv/archive/4.14.0.zip
RUN cd ~ && unzip -q opencv-4.14.0.zip && mv opencv-4.14.0 opencv && rm -f opencv-4.14.0.zip 

RUN cd ~ && wget -O opencv_contrib-4.14.0.zip https://github.com/opencv/opencv_contrib/archive/4.14.0.zip && unzip -q opencv_contrib-4.14.0.zip
RUN cd ~ && mv opencv_contrib-4.14.0 opencv_contrib && rm -f opencv_contrib-4.14.0.zip
RUN cd ~/opencv && mkdir build
RUN cd ~/opencv/build/ && cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D PYTHON3_EXECUTABLE=/usr/bin/python \
      -D PYTHON_INCLUDE_DIR=/usr/include/python3.12/ \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.12/dist-packages/numpy/core/include/ \
      -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.12/dist-packages/ \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D CUDNN_LIBRARY=/home/ubuntu/.local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so \
      -D CUDNN_INCLUDE_DIR=/home/ubuntu/.local/lib/python3.12/site-packages/nvidia/cudnn/include \
      -D CUDA_CUDA_LIBRARY=/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so \
      -D WITH_NVCUVID=OFF -D WITH_NVCUVENC=OFF \
      ..
RUN ln -s /home/ubuntu/.local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /home/ubuntu/.local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so &&\
    ln -s /home/ubuntu/.local/lib/python3.12/site-packages/numpy/_core/include/numpy /usr/include/numpy

RUN cd ~/opencv/build/ && make -j7 
RUN cd ~/opencv/build/ && make install 
RUN cd ~/opencv/build/ && ldconfig
RUN rm -rf ~/opencv*

RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
RUN pip install cv-bridge pandas 
RUN pip install --no-deps nvidia-ml-py polars-runtime-32 kiwisolver fonttools cycler contourpy \
    polars matplotlib ultralytics-platform ultralytics-thop ultralytics scipy seaborn
USER root 
COPY . /home/ubuntu/ros_ws/src/image_trackers_ouster
# # Make program executable
RUN cd /home/ubuntu/ros_ws/src/image_trackers_ouster/image_trackers_ouster/ && chmod +x tracker.py \
    && chmod +x detect.py
USER ubuntu
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && cd /home/ubuntu/ros_ws/ && colcon build --symlink-install --cmake-args  -DCMAKE_BUILD_TYPE=Release"

#Tracker
CMD export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && cd /home/ubuntu/ros_ws/ && \
    /bin/bash -c "source install/setup.bash; ros2 launch image_trackers_ouster tracker.xml"