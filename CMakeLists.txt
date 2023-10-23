cmake_minimum_required(VERSION 3.0.2)
project(husky_ML)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################


################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES husky_ML
CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



#############
## Install ##
#############

catkin_install_python(PROGRAMS
nodes/Husky_dqn
nodes/Husky_publisher
nodes/Husky_SAC_node
src/HUSKY_RL/husky.py
src/HUSKY_RL/respawnGoal.py
src/HUSKY_RL/Husky_ENV.py
src/HUSKY_RL/Husky_Memory.py
src/HUSKY_RL/Husky_Networks.py
src/HUSKY_RL/Husky_RNDUtils.py
src/HUSKY_RL/SAC_Husky.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch save_model
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

