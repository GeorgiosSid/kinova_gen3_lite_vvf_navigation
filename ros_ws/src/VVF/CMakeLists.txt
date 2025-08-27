cmake_minimum_required(VERSION 3.0.2)
project(VVF)

## Find catkin + deps
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf2_ros
  xacro
  robot_state_publisher
  joint_state_publisher
  gazebo_ros
  controller_manager
  jackal_description
  jackal_control
  kortex_description
  velodyne_description
  velodyne_gazebo_plugins
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    rospy
    std_msgs
    tf2_ros
    xacro
    robot_state_publisher
    joint_state_publisher
    gazebo_ros
    controller_manager
    jackal_description
    jackal_control
    kortex_description
    velodyne_description
    velodyne_gazebo_plugins
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install non-code assets so roslaunch/rosparam can find them
install(DIRECTORY launch config urdf worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)
# (Optional) if you have scripts/
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
  install(DIRECTORY scripts/
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    USE_SOURCE_PERMISSIONS)
endif()
