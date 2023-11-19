to run:

export HUSKY_LMS1XX_ENABLED=1
export HUSKY_URDF_EXTRAS=$HOME/Desktop/realsense.urdf.xacro

roslaunch cpr_office_gazebo office_world.launch platform:=husky
roslaunch husky_gazebo empty_world.launch 

roslaunch husky_viz view_robot.launch

roslaunch husky_ML Leydon_husky.launch
roslaunch husky_ML Leydon_husky_SAC.launch 


roslaunch cpr_gleason_gazebo gleason_world.launch platform:=husky

