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
#from respawnGoal import Respawn
from src.HUSKY_RL.respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.action_space = torch.zeros(action_size)
        #self.observation_size = 480*640*3+2 #480x640 image * rgb + heading, distance
        self.observation_size = 60*80*3+2
        self.observation_space = torch.zeros(self.observation_size)
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

        self.vel = 0.15
        self.ang = 0.0
        self.ang_decision = 0 # used in reward calculation

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.position.y, self.goal_x - self.position.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan, img):
        scan_range = []
        heading = self.heading
        min_range = 0.1
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y),2)
        if current_distance < 1:
            self.get_goalbox = True

        # aRather than taking 24 samples, the 360 are downsampled to 
        # 24 by taking the min of the nearest 15 points

        min_scan_range = np.zeros(24)
        for i in range(14):
            min_scan_range[i] = np.min(scan_range[i:i+15])

        #image is 480x640, down sample by 8 to 60x80
        img = img.data
        print(img)
        img  = np.array(img)
        #print(img)
        #print(img.)
        img_torch = torch.from_numpy(img)
        # Convert the uint8 tensor to a float tensor for interpolation
        image_float = img_torch.to(img, dtype=torch.float32) / 255.0

        # Downsample the image using nn.functional.interpolate
        downsampled_image = F.interpolate(image_float, size=(60,80), mode='bilinear', align_corners=False)

        # If needed, convert the downsampled image back to uint8
        downsampled_image = (downsampled_image * 255.0).to(torch.uint8)

        #self.observation_space = np.append(img.view(-1), [heading, current_distance])
        self.observation_space = torch.cat(downsampled_image.view(-1), [heading, current_distance]) #1d tensor
        return self.observation_space, done

    def setReward(self, state, done, action):
        yaw_reward = 0
        current_distance = state[-1]
        heading = state[-2]
        heading_reward = - 0.1 
        if type(action) is not int:
            action = int(action[0]) # don't actually need this

        distance_rate = 2 ** (current_distance / self.goal_distance)

        
        angle = -pi / 4 + heading + (pi / 8 * self.ang_decision) + pi / 2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        yaw_reward = tr

        
        #reward = ((round(yaw_reward * 5, 2)) * distance_rate)
        reward = 0.01 *  current_distance - self.goal_distance # small reward for movint towards goal.

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            print("Twist")
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            print("get Position")
            self.goal_distance = self.getGoalDistace()
            print("get goal distance")
            self.get_goalbox = False

        return reward

    def step(self, action):
        #action = int(action[0]) # may get upset by this
        max_angular_vel = 1.5
        if action < 5:
            self.ang = ((5- 1)/2 - action) * max_angular_vel * 0.5
            if type(action) is not int:
                self.ang_decision = int(action[0])
            else:
                self.ang_decision = action
        
        elif action < 10:
            self.vel = -(action - 5)*0.1 + 0.1 # range of 0.1 to 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = self.vel
        vel_cmd.angular.z = self.ang
        self.pub_cmd_vel.publish(vel_cmd)

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

        state, done = self.getState(Laser_data, Image_data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

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


        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(Laser_data, Image_data)

        return np.asarray(state)