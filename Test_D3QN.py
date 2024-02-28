# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import deque
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Flatten, Lambda, add, Conv2D
from keras.models import Sequential ,load_model, Model, model_from_json
from keras.optimizers import RMSprop, Adam
from matplotlib import style, gridspec                              
from skimage.color import rgb2gray
from sensor_msgs.msg import LaserScan, Image, Imu
from skimage.transform import resize
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from PIL import Image as iimage                                         
from tensorflow import Session, Graph

import cv2
import copy
import matplotlib
import matplotlib.pyplot as plt
import models
import numpy as np
import pickle
import rospy
import random
import random
import scipy.misc
import tensorflow as tf
import time

from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int8
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

# load model for collision probability estimation
from keras.models import model_from_json

# define empty image variable
depth = np.zeros([128,160]).astype('float32')
obs_flag = False
    
def callback_camera(msg):
    global image
    image = np.frombuffer(msg.data, dtype=np.uint8)    
    image = np.reshape(image, [480,640,3]) 
    image = np.array(image)  


def show_figure(image):
    #show image using cv2    
    image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Prediction image', image)    
    cv2.waitKey(1)


def get_depth(img_from_depth_est):
    global depth
    global obs_flag

    obs_threshold = 0.55
    
    #np_image = np.frombuffer(img_from_depth_est.data, np.float32)
    np_image = img_from_depth_est.data
    np_image = np.reshape(np_image, [480, 640])
    
    # obstacle detecting array
    obs_detector_array = copy.deepcopy(np_image[:-100,100:-100])
    obs_detector_array[obs_detector_array >= obs_threshold] = 1
    obs_detector_array[obs_detector_array < obs_threshold] = 0
    cv2.imshow("obs_detect",obs_detector_array)
    cv2.waitKey(1)

    detection_threshold = np.average(obs_detector_array)
    print(detection_threshold)
    if detection_threshold <= 0.94:
        obs_flag = True
        print("obstacle detected")
    else:
        obs_flag = False

    pil_image =  iimage.fromarray(np.float32(np_image))
    
    pil_image = pil_image.resize((160, 128), iimage.LANCZOS)
    depth = np.array(pil_image)
    # show_figure(depth)


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
sess.run(init)


class TestAgent:
    def __init__(self, action_size):
        self.state_size = (128, 160, 6)
        self.action_size = action_size
        self.model = self.build_model()        
        " Erase the config and tf.initializer when you load another model by keras"

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
    
        history = np.float32(history)
        
        with graph.as_default():
            q_value = self.model.predict(history)
        
        return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)
        global graph
        graph = tf.get_default_graph()


if __name__ == '__main__':
    rospy.init_node('Avoider', anonymous=True)
    
    # Parameter setting for the simulation
    agent = TestAgent(action_size = 8)  ## class name should be different from the original one
    
    # Observe
    # Change the rosSubscriber to another ros topic
    rospy.Subscriber('/Depth_est', Float32MultiArray, get_depth, queue_size = 10)
    OA_action_pubslisher = rospy.Publisher('/OA_action', Int8, queue_size=10)
    OA_flag_publisher = rospy.Publisher('/OA_flag', Bool, queue_size=10)            
    # rospy.sleep(2.)                
    
    rate = rospy.Rate(5)
       
    # model for collision avoidance
    agent.load_model("./save_model/D3QN_V_3_single.h5")
   
    # get the initial state
    state = depth
    history = np.stack((state, state, state, state, state, state), axis = 2)                
    history = np.reshape([history], (1,128,160,6))           

    while not rospy.is_shutdown():

        action = agent.get_action(history)  
        print(action)
        
        OA_action_pubslisher.publish(action)
        OA_flag_publisher.publish(obs_flag)
        
        # image preprocessing                        
        next_state = depth
        #show_figure(next_state)        

        # image for collision check            
        next_state = np.reshape([next_state],(1,128,160,1))
        next_history = np.append(next_state, history[:,:,:,:5],axis = 3)

        history = next_history
        rate.sleep()


