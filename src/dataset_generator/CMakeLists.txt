cmake_minimum_required(VERSION 3.0.2)
project(dataset_generator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

## Install the Python script(s) in scripts/ into the usual ROS bin directory
catkin_install_python(PROGRAMS
  scripts/generate_labels.py
  scripts/generate_inputs.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
