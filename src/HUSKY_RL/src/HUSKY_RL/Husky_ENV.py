#!/usr/bin/env python
import rospy
import numpy as np
import math
import torch
import torch.nn.functional as F

from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelStates
from src.HUSKY_RL.respawnGoal import Respawn

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from matplotlib import pyplot as plt

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.action_space = np.zeros(action_size)
        #self.observation_size = 480*640*3+2 #480x640 image * rgb + heading, distance
        self.observation_size = (3,86,86) 
        self.observation_space = np.random.random((480,640,3))
        self.observation_arr_size = (4+86)
        self.observation_arr_space = np.random.random(4+86)
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        #self.sub_odom = rospy.Subscriber('husky_velocity_controller/odom', Odometry, self.getOdometry)
        self.sub_gazebo = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

        self.vel = 0.6
        self.ang = 0.0
        self.vel_cmd = Twist()
        self.ang_decision = 0 # used in reward calculation
        self.previous_distance = 0
        self.current_distance = 0
        self.initial_distance = 0
        self.goal_distance = 0

        self.roll = 0
        self.pitch = 0

        self.crash_table = [0.5277445361,0.4807631344,0.448892577,0.4271859065,0.4133652076,0.406783548,0.406431967,0.4068361089,0.3983151644,0.3957613125,
                            0.3347469971,0.2897197455,0.2595413186,0.2395230107,0.2256503619,0.5,0.5,0.5,0.2076443799,0.2102404863,
                            0.2158739664,0.2255596749,0.2400515437,0.2593861885,0.2894249037,0.334886238,0.4090430573,0.4263875529,0.4363628432,0.4534364715,
                            0.4803785101,0.4995101631,0.5169088393,0.5445159167,0.5851512015,0.6444034994]

    def resize_n_reshape(self, img):
        resized=np.asarray(cv2.resize(img, (86,86)))
        reshaped=resized.transpose((2,0,1))
        reshaped = reshaped/255
        #transposed_array = reshaped.transpose((1, 2, 0))
        #plt.imshow(transposed_array, )
        #plt.show()
        
        return reshaped

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        position = (self.position.x, self.position.y)
        return goal_distance
    
    def model_states_callback(self, msg):
        # Find the index of 'husky' in the names list
        try:
            husky_index = msg.name.index('husky')
        except ValueError:
            rospy.logwarn("Could not find 'husky' in model_states")
            return

        # Extract the position information for 'husky'
        self.position = msg.pose[husky_index].position
        # give velocity in callback
        self.pub_cmd_vel.publish(self.vel_cmd)


    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.roll, self.pitch, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan, img, depth):
        scan_range = []
        scan_ds = []
        heading = self.heading
        min_range = 0.2
        done = False
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        #720 scan ranges, average by 20 scand (10 degrees)
        crash_flag = False
        for i in range(36):
            if i== 15 or i == 16 or i==17:
                scan_ds.append(1) #safe number
                continue # seeing camera
            start_index = i * 20
            end_index = start_index + 20
            sample = scan_range[start_index:end_index]
            val = sum(sample)/20
            scan_ds.append(val)
            if val <= self.crash_table[i]+0.05:
                crash_flag = True

        if crash_flag:
            done = True

        # update distances
        self.previous_distance = self.current_distance
        self.current_distance = self.getGoalDistace()

        if self.current_distance < 1:
            self.get_goalbox = True

        #image is 480x640, down sample by 8 to 60x80
        bridge = CvBridge()
        #cv_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        cv_image = bridge.imgmsg_to_cv2(img, 'bgr8')
        # Convert sensor_msgs/Image to an OpenCV image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        img = self.resize_n_reshape(cv_image)
        self.observation_space = img

        depth_image = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        # resize
        depth_image=np.asarray(cv2.resize(depth_image, (86,86)))
        depth_image[np.isnan(depth_image)] = 8
        depth_image = depth_image / 8

        self.observation_arr_space = np.hstack((np.array((self.position.x, self.position.y, self.goal_x, self.goal_y)), depth_image[45, :]))
        return done

    def setReward(self, done, action):
        #reward = -2*(self.current_distance - self.previous_distance) # small reward for moving towards goal.
        reward = -0.02*(self.current_distance - self.previous_distance) # max 0.374 * 2

        if done:
            rospy.loginfo("Collision!!")
            #reward = -30 #-200
            reward = -0.3
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            #reward = 200
            reward = 0.5
            self.pub_cmd_vel.publish(Twist())
            print("Twist")
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            print("get Position")
            self.goal_distance = self.getGoalDistace()
            print("get goal distance")
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        #self.ang = ((5- 1)/2 - action) * max_angular_vel * 0.5
        action_map = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
        self.ang = action_map[action]
        #self.ang_decision = action

        self.vel_cmd.linear.x = self.vel
        self.vel_cmd.angular.z = self.ang
        #self.pub_cmd_vel.publish(vel_cmd)

        Laser_data = None
        while Laser_data is None:
            try:
                Laser_data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass

        Image_data = None
        while Image_data is None:
            try:
                Image_data = rospy.wait_for_message('/realsense/color/image_raw', Image, timeout=5)
            except:
                pass

        Depth_data = None
        while Depth_data is None:
            try:
                Depth_data = rospy.wait_for_message('/realsense/depth/image_rect_raw', Image, timeout=5)
            except:
                pass
        done = self.getState(Laser_data, Image_data, Depth_data)
        reward = self.setReward(done, action)

        return self.observation_space, self.observation_arr_space, reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        Laser_data = None
        while Laser_data is None:
            try:
                Laser_data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass

        Image_data = None
        while Image_data is None:
            try:
                Image_data = rospy.wait_for_message('/realsense/color/image_raw', Image, timeout=5)
            except:
                pass

        Depth_data = None
        while Depth_data is None:
            try:
                Depth_data = rospy.wait_for_message('/realsense/depth/image_rect_raw', Image, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        #self.initial_distance = 0 #reset 
        self.goal_distance = self.getGoalDistace()
        done = self.getState(Laser_data, Image_data, Depth_data)
        return self.observation_space, self.observation_arr_space,
