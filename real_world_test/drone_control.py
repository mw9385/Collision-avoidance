#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rospy
import numpy as np
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class PID():
    def __init__(self):
        # init_node
        rospy.init_node('sample', anonymous=False)
        self._error = 0.0 # previous error term
        self._integral = 0.0 # previous integral term
        self.delta_t = 0.2 # command frequency
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._velocity_z = 0.0
        self._angular_x = 0.0
        self._angular_y = 0.0
        self._angular_z = 0.0
        
        # publisher
        self.cmd_vel = rospy.Publisher('/bebop/cmd_vel',Twist, queue_size=10)
        self.twist = Twist()
        # subscribe velocity topics
        self.velocity = rospy.Subscriber('/bebop/odom', Odometry, self.velocity_callback)
        
        #action table
        self.speed = [0.1, 0.0]
        self.action_table = [0.4, 0.1, 0.0, 0.3, -0.3, 0.0, -0.15, 0.15]       
    def get_velocity(self):
        return [self.velocity_x, self.velocity_y]
    
    def velocity_control(self, action):
        desired_control = self.action_decision(action)
        
        self.desired_x = desired_control[0]
        self.desired_y = 0.0
        self.desired_z = 0.0
        self.desired_ax = 0.0
        self.desired_ay = 0.0
        self.desired_az = desired_control[1]

        # define PID gain
        self.Kp = 1.2
        self.Kd = 0.05
        
        # error define
        error_x =  self.desired_x - self.velocity_x 
        error_y =  self.desired_y - self.velocity_y 
        error_z =  self.desired_z - self.velocity_z
        error_ax =  self.desired_ax - self.angular_x 
        error_ay =  self.desired_ay - self.angular_y 
        error_az =  self.desired_az - self.angular_z
        
        # PID control part
        u_x = self.Kp * error_x + self.Kd * (self.velocity_x - self._velocity_x) / self.delta_t
        u_y = self.Kp * error_y + self.Kd * (self.velocity_y - self._velocity_y) / self.delta_t
        u_z = self.Kp * error_z + self.Kd * (self.velocity_z - self._velocity_z) / self.delta_t
        
        a_x = self.Kp * error_ax + self.Kd * (self.angular_x - self._angular_x) / self.delta_t
        a_y = self.Kp * error_ay + self.Kd * (self.angular_y - self._angular_y) / self.delta_t
        a_z = self.Kp * error_az + self.Kd * (self.angular_z - self._angular_z) / self.delta_t
        
        
        self_velocity_x = self.velocity_x
        self_velocity_y = self.velocity_y
        self_velocity_z = self.velocity_z
        self_angular_x = self.angular_x
        self_angular_y = self.angular_y
        self_angular_z = self.angular_z
        
        self.twist.linear.x = u_x
        self.twist.linear.y = u_y
        self.twist.linear.z = u_z
        self.twist.angular.x = a_x
        self.twist.angular.y = a_y
        self.twist.angular.z = a_z
        self.cmd_vel.publish(self.twist)
        return u_x, a_z
    
    def action_decision(self, action):
        if action < 3:
            self.speed[0] = self.action_table[action]
            self.speed[1] = 0.0
        else:
            self.speed[1] = self.action_table[action]
            
        return self.speed
    
    
    def velocity_callback(self, msg):
        self.velocity_x = msg.twist.twist.linear.x
        self.velocity_y = msg.twist.twist.linear.y
        self.velocity_z = msg.twist.twist.linear.z
        self.angular_x = msg.twist.twist.angular.x
        self.angular_y = msg.twist.twist.angular.y
        self.angular_z = msg.twist.twist.angular.z
        
