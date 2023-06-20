cmake_minimum_required(VERSION 3.0.2)
project(m2track_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  vision_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES iri_python_template
  CATKIN_DEPENDS rospy vision_msgs sensor_msgs visualization_msgs geometry_msgs cv_bridge
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()

catkin_install_python(PROGRAMS src/m2track_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

