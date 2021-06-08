#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.layers.convolutional import Conv2D
from keras.layers import Input, Dense, Flatten, Lambda, add
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential ,load_model, Model, model_from_json
from keras.backend.tensorflow_backend import set_session
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
from ENV import ENV
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Vector3Stamped
from skimage.transform import resize
from PIL import Image as iimage
from keras.utils.training_utils import multi_gpu_model
from models.cnn_model_LSTM_many_to_one import cnn_lstm
from matplotlib import style, gridspec                              
from tensorflow import Session, Graph
from keras.models import model_from_json
from keras.utils import to_categorical

import matplotlib
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


# load depth estimation model
with tf.name_scope("predict"):    
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    model_data_path = './NYU_FCRN-checkpoint/NYU_FCRN.ckpt'

    print('start create the session and model')
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    config = tf.ConfigProto()
    cnn_sess = tf.Session(config=config)
    cnn_saver = tf.train.Saver()     
    cnn_saver.restore(cnn_sess, model_data_path)
    print('Finishied')

# define empty parameters
laser = None
velocity = None
vel = None
theta = None
pose = None
orientation = None
image = None
depth_img = None

def callback_laser(msg):    
    global laser
    laser = msg    
    laser = laser.ranges 
    
def DepthCallBack(img):
    global depth_img
    depth_img = img.data
    
def callback_camera(msg):
    global image
    image = np.frombuffer(msg.data, dtype=np.uint8)    
    image = np.reshape(image, [480,640,3]) 
    image = np.array(image)
    
def state_callback(msg):
    global velocity, pose, orientation, vel, theta
    idx = msg.name.index("quadrotor")      
    
    pose = msg.pose[idx].position
    orientation = msg.pose[idx].orientation    
    vel = msg.twist[idx]
    
    velocity_x = vel.linear.x
    velocity_y = vel.linear.y    
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)    
    theta = vel.angular.z
    
def GetDepthObservation(image):
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
    pred[pred > 1.0] = 1.0     
        
    return pred

def GetKineticDepthObservation(depth_img):
    
    noise = np.random.random([128,160]) * 0.5
    a = copy.deepcopy(depth_img)
    a = np.frombuffer(a, dtype = np.float32)
    a = np.reshape(a,[480,640]) 
    a = resize(a, [128,160])
    a = np.array(a)
    a[np.isnan(a)] = 5. ## YOU SHHOULD CHANGE THIS!!!!!!!!!
        
    dim =[128, 160]
    gauss = np.random.normal(0., 1.0, dim)
    gauss = gauss.reshape(dim[0], dim[1])
    a = np.array(a, dtype=np.float32)
    a = a + gauss
    a[a<0.00001] = 0.
    a[a > 5.0] = 5.0
    a = a/5
    a = cv2.GaussianBlur(a, (25,25),0)   
    
    return a


def crash_check(laser_data, velocity, theta, delta_depth):    
    laser_sensor = np.array(laser_data)
    laser_index = np.isinf(laser_sensor)
    laser_sensor[laser_index] = 30
    laser_sensor = np.array(laser_sensor[300:800])    
    done = False
    vel_flag = False
    zone_1_flag = False    
    
    crash_reward = 0
    depth_reward = 0
    vel_reward = 0
    depth_value = (np.min(laser_sensor) - 0.5) / 2.0       
    
    # reward for zone 1
    if depth_value >= 0.4:         
        depth_reward = 0.4
        vel_flag = True        
        
    # reward for zone 2
    else:                                
        vel_factor = np.absolute(np.cos(velocity))
        _depth_reward = depth_value * vel_factor + delta_depth
        depth_reward = np.min([0.4, _depth_reward])           
        vel_flag = False        
        
    # reward for crash
    if np.min(laser_sensor) <= 0.6:
        done = True                
        vel_flag = False
    
    # reward for velocity
    else:
        if vel_flag:
            vel_reward = velocity * np.cos(theta)* 0.2                
            
        else:
            vel_reward = 0            
    # select reward
    if done:
        reward = -1.0
    else:
        reward = depth_reward + vel_reward  
         
    return done, reward, np.min(laser_sensor), depth_value
  
def depth_change(depth,_depth):
    laser = depth   # current depth
    _laser = _depth # previous depth  
    eta = 0.2
    
    delta_depth = eta * np.sign(laser - _laser)
    return delta_depth
    
def show_figure(image):
    #show image using cv2    
    image = cv2.resize(image, (256*2, 320*2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Prediction image', image)    
    cv2.waitKey(1)
         
    
class TestAgent:
    def __init__(self, action_size):
        self.state_size = (128, 160 , 6)
        self.action_size = action_size
        self.model = multi_gpu_model(self.build_model(), gpus = 2)
        " Erase the config and tf.initializer when you load another model by keras!!!"

    def build_model(self):
        input = Input(shape=self.state_size)
        h1 = Conv2D(32, (8, 8), strides = (8,8), activation = "relu", name = "conv1")(input)
        h2 = Conv2D(64, (4, 4), strides = (2,2), activation = "relu", name = "conv2")(h1)
        h3 = Conv2D(64, (3, 3), strides = (1,1), activation = "relu", name = "conv3")(h2)
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


if __name__ == '__main__':
    rospy.init_node('env', anonymous=True)
    env = ENV()   

    # Parameter setting for the simulation
    # Class name should be different from the original one
    agent = TestAgent(action_size = 8)  
    EPISODE = 1000000
    global_step = 0 

    # Observe
    rospy.Subscriber('/camera/depth/image_raw', Image, DepthCallBack,queue_size = 10)
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback_camera,queue_size = 10)    
    rospy.Subscriber("/scan", LaserScan, callback_laser,queue_size = 10)
    rospy.Subscriber('gazebo/model_states', ModelStates, state_callback, queue_size= 10)        
    rospy.sleep(2.)       
         
    e = 0
    rate = rospy.Rate(5)
    agent.load_model("./save_model/D3QN_V3.h5")

    while e < EPISODE and not rospy.is_shutdown():
        e = e + 1
        env.reset_sim(pose, orientation) 
        # get the initial state
        state = GetDepthObservation(image)
        history = np.stack((state, state, state, state, state, state), axis = 2)                
        history = np.reshape([history], (1,128,160,6))        
              
        laser_distance = np.stack((0, 0))        
        delta_depth = 0
        
        step, score  = 0. ,0.
        done = False            

        while not done and not rospy.is_shutdown():  

            global_step = global_step + 1             
            step = step + 1   
            
	    # get action through D3QN policy                    
            [action, flag] = agent.get_action(history)
            env.Control(action)
            
            # Observe: get_reward
            [done, reward, _depth, depth_value] = crash_check(laser, velocity, theta, delta_depth)   
            delta_depth = depth_change(laser_distance[0], laser_distance[1])                                            

            # image preprocessing
            next_state = GetDepthObservation(image)
            next_distance = _depth                      
            show_figure(next_state)        

            # image for collision check            
            next_state = np.reshape([next_state],(1,128,160,1))                    
            next_history = np.append(next_state, history[:,:,:,:5],axis = 3)
            
            # action hisotory
            _action = to_categorical(action, 8)
            next_action_history = np.append(_action, action_history[:5,:])
            next_action_history = np.reshape([next_action_history], [6,8])            

            # store next states
            history = next_history
            action_history = next_action_history

            if step >= 2000:
                done = True
            # Update the score
            score +=reward   
            rate.sleep()

        if done:    
            print("This is score " + str(score))
