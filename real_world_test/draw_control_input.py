#!/usr/bin/env python
# coding: utf-8

# In[13]:

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation

def speed_plot(a,b):
#     linear_speed = velocity_command.linear.x
#     angular_speed = velocity_command.angualr.z
    linear_speed = a
    angular_speed = b
    
    angular_list = (['angular_velocity'])
    linear_list = (['linear velocity'])
    
    x_pos = np.arange(len(angular_list))
    y_pos = np.arange(len(linear_list))
    
    fig = plt.figure(1)
    #plot figure with subplots of different sizes
    gridspec.GridSpec(6,6)
    #set up subplot grid    
    plt.subplot2grid((6,6), (1,0), colspan=4, rowspan=2)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.title('Angular (rad/s)')
    plt.xlabel('value')
    plt.barh(y_pos, [-0.5, 0.5], color = [1.0,1.0, 1.0])
    plt.barh(y_pos, angular_speed, color = 'r')
    plt.yticks([])
    
    plt.subplot2grid((6,6), (0,5), colspan=1, rowspan= 5)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.title('Linear (m/s)')
    plt.xlabel('value')
    plt.bar(x_pos, [0, 0.5], color = [1.0, 1.0, 1.0])
    plt.bar(x_pos, linear_speed, color = 'b')
    plt.xticks([])
    fig.savefig('sample.png', dpi=fig.dpi, bbox_inches = 'tight')

    
def realtime_plot_cv2(a,b):
    speed_plot(a,b)
    img = cv2.imread('sample.png', cv2.IMREAD_COLOR)
    cv2.imshow('control_pannel',img)
    cv2.waitKey(1)

k = 0.01
while(True):
    realtime_plot_cv2(1 + k,-k)
    k = k - 0.01
    
