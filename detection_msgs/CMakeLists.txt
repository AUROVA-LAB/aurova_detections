cmake_minimum_required(VERSION 3.5)
project(detection_msgs)
set(CMAKE_CXX_STANDARD 14)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

## Generate added messages and services with any dependencies listed here
rosidl_generate_interfaces( ${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/BoundingBoxes.msg"
  "msg/PositionList.msg"
  DEPENDENCIES
  std_msgs 
  geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()