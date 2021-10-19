#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


from keras.layers.convolutional import Conv2D
from keras.layers import Input, Dense, Flatten, Lambda, add
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential ,load_model, Model
from keras.backend.tensorflow_backend import set_session
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Vector3Stamped, Twist
from skimage.transform import resize
from PIL import Image as iimage
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
from ENV import environment
from drone_control import PID
from std_msgs.msg import Empty
from matplotlib import gridspec
from matplotlib.figure import Figure


import matplotlib.pyplot as plt
import rospy
import tensorflow as tf
import scipy.misc
import numpy as np
import random
import time
import random
import pickle
import models
import cv2
import copy


image = []

def callback_camera(msg):
    global image
    img_height = msg.height
    img_width = msg.width
        
    image = np.frombuffer(msg.data, dtype=np.uint8)       
    image = np.reshape(image, [img_height,img_width,3]) 
    image = np.array(image)     

def speed_plot(a,b):
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
    plt.barh(y_pos, [-0.6, 0.6], color = [1.0,1.0, 1.0])
    plt.barh(y_pos, angular_speed, color = 'r')
    plt.yticks([])
    
    plt.subplot2grid((6,6), (0,5), colspan=1, rowspan= 5)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.title('Linear (m/s)')
    plt.xlabel('value')
    plt.bar(x_pos, [0, 0.2], color = [1.0, 1.0, 1.0])
    plt.bar(x_pos, linear_speed, color = 'b')
    plt.xticks([])
    fig.savefig('sample.png', dpi=fig.dpi, bbox_inches = 'tight')

    
def realtime_plot_cv2(a,b):
    speed_plot(a,b)
    img = cv2.imread('sample.png', cv2.IMREAD_COLOR)
    cv2.imshow('control_pannel',img)
    cv2.waitKey(1)


# load model
g1 = tf.Graph()
g2 = tf.Graph()


with g1.as_default():
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    model_data_path = './NYU_FCRN-checkpoint/NYU_FCRN.ckpt'

    # Construct the network
    print('start create the session and model')
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    cnn_init = tf.global_variables_initializer()
    cnn_config = tf.ConfigProto(allow_soft_placement = True)
    cnn_sess = tf.Session(config = cnn_config)
    cnn_sess.run(cnn_init)
    cnn_saver = tf.train.Saver()     
    cnn_saver.restore(cnn_sess, model_data_path)

    print('Finishied')
    
def get_depth(image):
    width = 304
    height = 228
    
    test_image = iimage.fromarray(image,'RGB')
    test_image = test_image.resize([width, height], iimage.ANTIALIAS)    
    test_image = np.array(test_image).astype('float32')
    test_image = np.expand_dims(np.asarray(test_image), axis = 0)
    pred = cnn_sess.run(net.get_output(), feed_dict={input_node: test_image})
    pred = np.reshape(pred, [128,160])    
    pred = np.array(pred, dtype=np.float32)
    
    pred[np.isnan(pred)] = 5. 
    pred = pred / 3.5
    pred[pred>1.0] = 1.0
    
    return pred

class TestAgent:
    def __init__(self, action_size):
        self.state_size = (128, 160 , 8)
        self.action_size = action_size
        self.model = self.build_model()
        self.config = tf.ConfigProto()                
        self.sess = tf.InteractiveSession(config=self.config)         
        self.sess.run(tf.global_variables_initializer())
        K.set_session(self.sess)           
    
    def build_model(self):
        input = Input(shape=self.state_size)
        h1 = Conv2D(32, (10, 14), strides = 8, activation = "relu", name = "conv1")(input)
        h2 = Conv2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2")(h1)
        h3 = Conv2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3")(h2)
        context = Flatten(name = "flatten")(h3)
        
        value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(context)
        value = Dense(1, name = "value")(value_hidden)
        action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(context)
        action = Dense(self.action_size, name = "action")(action_hidden)
        action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
        output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'output')([action, value, action_mean])
        model = Model(inputs = input, outputs = output)
        model.summary()
        
        return model

    def get_action(self, history):
        flag = False
        if np.random.random() < 0.001:
            flag = True
            return random.randrange(8), flag 
        history = np.float32(history)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0]), flag

    def load_model(self, filename):
        self.model.load_weights(filename)        
        
# Intial hovering for 2 seconds and collecting the data from the laser and camera
# Receivng the data from laser and camera
# checking the crash using crash_check function and if crash occurs, the simulation is reset
with g2.as_default():
    
    if __name__ == '__main__':
        # Check the gazebo connection
        "Publisher"
        takeoff = rospy.Publisher('/bebop/takeoff',Empty, queue_size= 10)
        land = rospy.Publisher('/bebop/land', Empty, queue_size= 10)
        
        "Subscribe"
#         rospy.init_node('D3QN_TEST', anonymous=True)  
        rospy.Subscriber('/bebop/image_raw', Image, callback_camera, queue_size = 10)
               
        # Parameter setting for the simulation
        agent = TestAgent(action_size = 8)  ## class name should be different from the original one
        agent.load_model("./Saved_models/D3QN_V_17_single.h5")
        EPISODE = 100000    
        global_step = 0 

        env = environment()
        # Observe    
        rospy.sleep(2.)                
        e = 0
        rate = rospy.Rate(5)
        vel_pid = PID()
        
        while e < EPISODE and not rospy.is_shutdown():
            takeoff.publish()
            rospy.sleep(5)
            
            e = e + 1            
            # get the initial state
            state = get_depth(image)
            history = np.stack((state, state, state, state,state, state, state, state), axis = 2)                
            history = np.reshape([history], (1,128,160,8))        

            step, score  = 0. ,0.
            done = False                    

            while not done and not rospy.is_shutdown():  

                global_step = global_step + 1             
                step = step + 1   
                # Receive the action command from the Q-network and do the action           
                [action, flag] = agent.get_action(history)                                                      
                # give control_input
                [linear, angular] = vel_pid.velocity_control(action)
    
                # image preprocessing            
                next_state = get_depth(image)                   
                
                # plot real time depth image
                aa = cv2.resize(next_state, (128*3, 160*3), interpolation = cv2.INTER_CUBIC)
                cv2.imshow('input image', aa)
                cv2.waitKey(1)
                
                #plot real time control input
                realtime_plot_cv2(linear, -angular)
                
                # image for collision check            
                next_state = np.reshape([next_state],(1,128,160,1))                        
                next_history = np.append(next_state, history[:,:,:,:7],axis = 3)

                history = next_history

                if step >= 2000:
                    done = True
                # Update the score            
                rate.sleep()

            if done:    
                print(score)
