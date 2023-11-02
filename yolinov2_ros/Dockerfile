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
ENV HOME /home/mice85
RUN mkdir -p ${HOME}/ros_ws/src

# Initialize the workspace
RUN cd ${HOME}/ros_ws/src && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_init_workspace"
RUN cd ${HOME}/ros_ws /bin/bash -c "source source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_make"

# set up environment
#COPY ./update_bashrc /sbin/update_bashrc
#RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# Install pip
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

# Install deepsort
RUN pip install gdown easydict 
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install segmentation models
RUN pip install albumentations
RUN pip install segmentation-models-pytorch

# Create package
#RUN cd ${HOME}/ros_ws/src
RUN cd ${HOME}/ros_ws/src && catkin_create_pkg yolinov2 --rosdistro noetic
COPY ./* ${HOME}/ros_ws/src/yolinov2/
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x detect.py
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x test.py
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x train.py
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x utils.py
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x models.py
RUN cd ${HOME}/ros_ws/src/yolinov2/ && chmod +x dataset.py

## Compile the package
RUN cd ${HOME}/ros_ws/src/ /bin/bash -c "source source /opt/ros/${ROS_DISTRO}/setup.bash; catkin build yolinov2"
#CMD python3 home/mice85/ros_ws/src/yolinov2/detect.py  [path] [num_epoch]
#CMD python3 home/mice85/ros_ws/src/yolinov2/train.py [path] [bool_load_epoch] [num_epoch] [height] [width]
#CMD python3 home/mice85/ros_ws/src/yolinov2/test.py [path] [num_epoch] [num_test_image]

#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

#ENV TZ=Europe/Madrid
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#ARG USER_ID
#ARG GROUP_ID
#ARG DEBIAN_FRONTEND=noninteractive

# Install segmentation models
#RUN pip install albumentations
#RUN pip install segmentation-models-pytorch

# Install dependences and tools
#RUN apt-get update
#RUN apt-get install nano
#RUN apt-get update && apt-get install -y gnupg
#RUN apt-get update && apt-get install -y lsb-release && apt-get clean all

# Install ros melodic
#RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#RUN apt install curl -y
#RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
#RUN apt update
#RUN apt install ros-melodic-desktop-full -y

# Initialize the workspace
#RUN mkdir -p /workspace/aurova_ws/src
#RUN cd /workspace/aurova_ws/src && /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_init_workspace"
#RUN cd /workspace/aurova_ws /bin/bash -c "source source /opt/ros/melodic/setup.bash; catkin_make"

# Create package
#RUN cd /workspace/aurova_ws/src
#RUN cd /workspace/aurova_ws/src && catkin_create_pkg yolinov2 --rosdistro melodic
#COPY ./* /workspace/aurova_ws/src/yolinov2/
#RUN cd /workspace/aurova_ws/src/yolinov2/ && chmod +x detect.py

#RUN pip install rospkg


# Compile the package
#RUN cd /workspace/aurova_ws/src/ /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make"

#RUN python3 -m venv /opt/venv
#RUN . /opt/venv/bin/activate && pip install PyYAML && pip install rospkg && pip install empy && pip install opencv-python && pip install opencv-contrib-python

#RUN pip install PyYAML && pip install rospkg && pip install empy && pip install opencv-python && pip install opencv-contrib-python

#SHELL [ "/bin/bash" , "-c" ]

#RUN source /opt/ros/melodic/setup.bash && mkdir -p catkin_ws/src && cd catkin_ws \
# && catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python3.7 && source devel/setup.bash \
# && echo "export ROS_MASTER_URI=http://192.168.100.6:11311/" >> /etc/bash.bashrc && cd src && \
# catkin_create_pkg tactile_segmentation std_msgs roscpp rospy && cd tactile_segmentation && mkdir scripts && cd ../.. && catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python3.7 \
# && source devel/setup.bash
 
