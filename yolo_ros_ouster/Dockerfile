# This is an auto generated Dockerfile for ros:desktop-full
# generated from docker_images/create_ros_image.Dockerfile.em
FROM osrf/ros:noetic-desktop-focal

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=noninteractive

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

# Change HOME environment variable
ENV HOME /home/ignacio
RUN mkdir -p ${HOME}/ros_ws/src

# Initialize the workspace
RUN cd ${HOME}/ros_ws/src && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_init_workspace"
RUN cd ${HOME}/ros_ws /bin/bash -c "source source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_make"

# set up environment
COPY ./update_bashrc /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# Install pip
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

# Install yolov5
RUN cd ${HOME}/ros_ws/src/ && git clone https://github.com/EPVelasco/detection_msgs.git && git clone --recurse-submodules https://github.com/EPVelasco/yolov5_ros.git && cd yolov5_ros/src/yolov5 && pip install -r requirements.txt && cd ${HOME}/ros_ws/ 

# Install deepsort
RUN pip install gdown easydict 
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Compile the package
RUN cd ${HOME}/ros_ws/src/ /bin/bash -c "source source /opt/ros/${ROS_DISTRO}/setup.bash; catkin build yolov5_ros"



# Copy files from host to docker
COPY ./yolov5.launch /home/ignacio/ros_ws/src/yolov5_ros/launch/
COPY ./upsampling.py /usr/local/lib/python3.8/dist-packages/torch/nn/modules/
COPY ./Arial.ttf /home/ignacio/.config/Ultralytics/
COPY ./yolo.py /home/ignacio/ros_ws/src/yolov5_ros/src/yolov5/models/
COPY ./detect.py /home/ignacio/ros_ws/src/yolov5_ros/src/
COPY ./merge_channels_ouster.py /home/ignacio/ros_ws/src/yolov5_ros/src/
COPY ./CMakeLists.txt /home/ignacio/ros_ws/src/yolov5_ros/

# Make program executable
RUN cd ${HOME}/ros_ws/src/yolov5_ros/src/ && chmod +x detect.py
RUN cd ${HOME}/ros_ws/src/yolov5_ros/src/ && chmod +x merge_channels_ouster.py

#RUN git clone https://github.com/N0rbert/deb-download.git
#RUN cd deb-download && chmod +x deb-download.sh && ./deb-download.sh ubuntu focal libmkl-full-dev
#RUN apt-get install ./storage/*mkl*.deb

# Launch everything when docker run
CMD export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && cd ./../home/ignacio/ros_ws/ && /bin/bash -c "source source devel/setup.bash; catkin_make; source devel/setup.bash; roslaunch yolov5_ros yolov5.launch"
