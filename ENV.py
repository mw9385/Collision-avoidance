#!/usr/bin/env python
# coding: utf-8

###########################################################################
###########################################################################
from __future__ import print_function
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
from keras.utils import to_categorical
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import rospy
import std_msgs.msg 
import sys, select, termios, tty
import time
###########################################################################
###########################################################################

class ENV(object):
    def __init__(self):        
        # define gazebo connection and import gazebo msgs
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 100)
        self.pose_pub = rospy.Publisher('/command/pose', Pose, queue_size = 100)        
        self.g_set_state = rospy.ServiceProxy('gazebo/set_model_state',SetModelState)        
              
        self.twist = Twist()
        self.pose = Pose()                       
        self.action_table = [0.5, 0.25, 0.03 , 0.3, -0.3, 0, -0.15, 0.15]                  
        self.state = ModelState()
        self.state.model_name = 'quadrotor'
        self.state.reference_frame = 'world'        
                        
        
    def stop(self):
        self.twist.linear.x = 0; self.twist.linear.y = 0; self.twist.linear.z = 0
        self.twist.angular.x = 0; self.twist.angular.y = 0; self.twist.angular.z = 0                
        self.cmd_pub.publish(self.twist)               
        
    def hovering(self):
             
        self.state.pose.position.x = 0
        self.state.pose.position.y = 0
        self.state.pose.position.z = 1.2
        self.state.pose.orientation.x = 0
        self.state.pose.orientation.y = 0
        self.state.pose.orientation.z = 0
        self.state.twist.linear.x = 0
        self.state.twist.linear.y = 0
        self.state.twist.linear.z = 0
        self.state.twist.angular.x = 0
        self.state.twist.angular.y = 0
        self.state.twist.angular.z = 0

        ret = self.g_set_state(self.state)
        
        
    def Control(self,action):
        if action < 3:
            self.self_speed[0] = self.action_table[action]
            self.self_speed[1] = 0
        else:                    
            self.self_speed[1] = self.action_table[action]

        self.twist.linear.x = self.self_speed[0]
        self.twist.linear.y = 0.
        self.twist.linear.z = 0.
        self.twist.angular.x = 0.
        self.twist.angular.y = 0.
        self.twist.angular.z = self.self_speed[1]
        self.cmd_pub.publish(self.twist)                
    
    def euler_to_quaternion(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]
    
    def quaternion_to_euler(self, x, y, z, w):
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z
    
    def reset_sim(self, pose, orientation):                
        pose_x = pose.x
        pose_y = pose.y
        pose_z = pose.z
        
#         spawn_table= [
#                     [-38.71, 41.93],
#                     [-27.13, 41.97],
#                     [-18.66, 41.67],
#                     [-15.15, 33.80],
#                     [-36.60, 15.96],
#                     [-14.69, 13.68]
            
#                     [6.96, 33.80],
#                     [16.58, 30.71],
#                     [24.69, 36.19],
#                     [20.49, 19.59],
#                     [5.28, 4.63],
#                     [33.73, 8.40],
#                     [26.91, 15.03],
#                     [36.53, 35.03],
#                     [24.00, 2.85],
#                     [16.37,5.91]
            
#                     [-27, -10],
#                     [-34, -20],
#                     [-25, -25],
#                     [-14, -15],
#                     [-5, -18],
#                     [-12, -5],
#                     [-32, -35]
            
#                     [12.63, -19.98],
#                     [24.00, -22.83],
#                     [27.53, -8.13],
#                     [18.55, -32.01],
#                     [26.32, -30.26],
#                     [34.77, -28.82],
#                     [6.70, -34.52],
#                     [3.63, -15.97],
#                     [4.40, -8.01]
#                     ]
        
#         spawn_table = [[-26.48,6.65],[-18.62,-12.41],[-1.54,-22.16],[-19.89,-27]] # map b (indoor)
#        spawn_table = [[26.75,30.14],[-25.33,-19.98],[-1.181,30.48],[10.99,-22.30],[19.23,30.83]]  #Map test_3
#         spawn_table = [[-14.75, -13.34]] # Map test_2
        spawn_table = [[0, 0]] # Map test_
#         spawn_table = [[-9.10, 34.95]] # Map test_2
#         spawn_table = [[-36.20, -3.09], [-24.59, 2.25], [-21.00, -5.13], [-11.37, -0.64], [-2.44, -0.59], [5.04, -3.37],                               [-0.848, -7.81], [-37.99, 1.96]] # indoor environment_4_1
#        spawn_table = [[-26.48,6.65],[-18.62,-12.41],[-1.54,-22.16],[-19.89,-27],[19.49,5.97],[20.45, -10.62],[20.76,-23.48]] 
#         spawn_table = [[-24.5, -5.37],[-9.53, 9.62],[-5.53,26.08], [-13.54, 20.62], [-23.54, 23.62], [3.46, -1.37]] # map a
#         spawn_table = [[0.67, 0.27], [-13.10, 14.02], [-14.38, -14.96], [14.06, -10.86], [29.35, -4.52], [47.32, 7.84], [46.99, -8.37]]

        rand_indx = np.random.randint(1)
        
        ori_x = orientation.x
        ori_y = orientation.y
        ori_z = orientation.z        
        ori_w = orientation.w
        
#         pos_rand = np.random.randint(-150,150)
#         pos_rand = 0
        
        [yaw, pitch, roll] = self.quaternion_to_euler(ori_x, ori_y, ori_z, ori_w)              
        [ori_x, ori_y, ori_z, ori_w] = self.euler_to_quaternion(0, 0, 0)        
                
        self.state.pose.position.x = spawn_table[rand_indx][0] + np.random.randint(-1,1)
        self.state.pose.position.y = spawn_table[rand_indx][1] + np.random.randint(-1,1) 
        self.state.pose.position.z = 1.2
        self.state.pose.orientation.x = ori_x
        self.state.pose.orientation.y = ori_y
        self.state.pose.orientation.z = ori_z
        self.state.pose.orientation.w = ori_w
        self.state.twist.linear.x = 0
        self.state.twist.linear.y = 0
        self.state.twist.linear.z = 0
        self.state.twist.angular.x = 0
        self.state.twist.angular.y = 0
        self.state.twist.angular.z = 0        
        self.self_speed = [0.03, 0.0]
        ret = self.g_set_state(self.state)    
        rospy.sleep(0.5)
