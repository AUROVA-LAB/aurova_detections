# Start FROM Nvidia PyTorch image
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

COPY requirement.txt .

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y python3.8
RUN apt install python-is-python3
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx python3-pip git
RUN pip install torch torchvision torchaudio
RUN pip install --no-cache -r requirement.txt
# Error using pomegranate, it was solved reinstalling it.
RUN pip install pomegranate==0.14.8
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install -y curl 
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt install -y ros-noetic-ros-base ros-noetic-vision-msgs ros-noetic-ros-numpy
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

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
RUN cd /home/ros_ws/src && bash -c 'source /opt/ros/noetic/setup.bash && catkin_init_workspace'
RUN cd /home/ros_ws && bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'
# COPY ./update_bashrc /sbin/update_bashrc
# RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# # Install Open3DSOT
RUN cd /home/ros_ws/src/ && git clone https://github.com/Ghostish/Open3DSOT.git 
# # Compile the package
RUN cd /home/ros_ws/ && bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

## Install this package
RUN apt update && apt install -y ros-noetic-geometry-msgs ros-noetic-visualization-msgs ros-noetic-sensor-msgs ros-noetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*
RUN cd /home/ros_ws/src/ && git clone https://github.com/EPVelasco/detection_msgs.git
RUN mkdir /home/ros_ws/src/pc_tracker
COPY ./ /home/ros_ws/src/pc_tracker
RUN chmod +x /home/ros_ws/src/pc_tracker/src/m2track_ros.py

CMD export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && cd /home/ros_ws/ && /bin/bash -c "source devel/setup.bash; catkin_make; source devel/setup.bash; roslaunch m2track_ros m2track_ros.launch"