cmake_minimum_required(VERSION 3.5)
project(tracker_filter)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
## Find catkin macros and libraries
# ******************************************************************** 
#                 Add catkin additional components here
# ******************************************************************** 
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(detection_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
# include_directories(${<dependency>_INCLUDE_DIRS})

## Declare a cpp library
# add_library(${PROJECT_NAME} <list of source files>)

## Declare a cpp executable
add_executable(${PROJECT_NAME} src/tracker_filter_alg_node.cpp src/ekf.cpp)

# ******************************************************************** 
#                   Add the libraries
# ******************************************************************** 
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME} rclcpp Eigen3 cv_bridge tf2 tf2_ros tf2_eigen tf2_geometry_msgs PCL pcl_conversions geometry_msgs detection_msgs image_transport)

install(TARGETS
  ${PROJECT_NAME} 
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rclcpp Eigen3)
ament_package()
