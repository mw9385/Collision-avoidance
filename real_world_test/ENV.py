#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import print_function
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from drone_control import PID

import numpy as np
import rospy


class environment():
    def __init__(self): 
        self.vel_pid = PID()
        
        # set the action table
        self.cmd_vel = rospy.Publisher('/bebop/cmd_vel',Twist, queue_size=10)
        self.twist = Twist()
        self.speed = [0.3, 0.0]
        self.action_table = [0.1, 0.1, 0.0, 0.2, -0.2, 0.0, -0.10, 0.10]        
        
    def control(self, action):
        if action < 3:
            self.speed[0] = self.action_table[action]
            self.speed[1] = 0.0
            self.twist = Twist()
        else:
            self.speed[1] = self.action_table[action]
        vel_command = self.vel_pid.velocity_control(self.speed)
        
        self.twist.linear.x = vel_command[0]
        self.twist.linear.y = vel_command[1]
        self.twist.linear.z = vel_command[2]
        self.twist.angular.x = vel_command[3]
        self.twist.angular.y = vel_command[4]
        self.twist.angular.z = vel_command[5]
        self.cmd_vel.publish(self.twist)
        return self.speed
