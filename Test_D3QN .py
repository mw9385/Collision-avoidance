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
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Flatten, Lambda, add, Conv2D
from keras.models import Sequential ,load_model, Model, model_from_json
from keras.optimizers import RMSprop, Adam
from matplotlib import style, gridspec                              
from skimage.color import rgb2gray
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from skimage.transform import resize
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Vector3Stamped
from skimage.transform import resize
from PIL import Image as iimage
from matplotlib import style, gridspec                              
from keras.models import model_from_json
from keras.utils import to_categorical
from nav_msgs.msg import Odometry
from PIL import Image as iimage                                         
from tensorflow import Session, Graph


import cv2
import copy
import matplotlib
import matplotlib.pyplot as plt
import rospy
import tensorflow as tf
import scipy.misc
import models
import numpy as np
import pickle
import rospy
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

    image = np.array(image)  


def show_figure(image):
    #show image using cv2    
    image = cv2.resize(image, (256*2, 320*2), interpolation=cv2.INTER_CUBIC)
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
        self.state_size = (128, 160 , 6)
        self.state_size = (128, 160, 6)
        self.action_size = action_size
        self.model = multi_gpu_model(self.build_model(), gpus = 2)
        " Erase the config and tf.initializer when you load another model by keras!!!"

        self.model = self.build_model()        
        " Erase the config and tf.initializer when you load another model by keras"

    def build_model(self):
        input = Input(shape=self.state_size)
        h1 = Conv2D(32, (8, 8), strides = (8,8), activation = "relu", name = "conv1")(input)
def build_model(self):


    def get_action(self, history):
        flag = False
        if np.random.random() < 0.001:
            flag = True
            return random.randrange(8), flag 

        history = np.float32(history)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0]), flag


        with graph.as_default():
            q_value = self.model.predict(history)

        return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)


        global graph
        graph = tf.get_default_graph()


if __name__ == '__main__':
    rospy.init_node('env', anonymous=True)
    env = ENV()   

    rospy.init_node('Avoider', anonymous=True)

    # Parameter setting for the simulation
    # Class name should be different from the original one
    agent = TestAgent(action_size = 8)  
    EPISODE = 1000000
    global_step = 0 

    agent = TestAgent(action_size = 8)  ## class name should be different from the original one

    # Observe
    rospy.Subscriber('/camera/depth/image_raw', Image, DepthCallBack,queue_size = 10)
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback_camera,queue_size = 10)    
    rospy.Subscriber("/scan", LaserScan, callback_laser,queue_size = 10)
    rospy.Subscriber('gazebo/model_states', ModelStates, state_callback, queue_size= 10)        
    rospy.sleep(2.)       

    e = 0
    # Change the rosSubscriber to another ros topic
    rospy.Subscriber('/Depth_est', Float32MultiArray, get_depth, queue_size = 10)
    OA_action_pubslisher = rospy.Publisher('/OA_action', Int8, queue_size=10)
    OA_flag_publisher = rospy.Publisher('/OA_flag', Bool, queue_size=10)            
    # rospy.sleep(2.)                

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
        # image preprocessing                        
        next_state = depth
        #show_figure(next_state)        

        # image for collision check            
        next_state = np.reshape([next_state],(1,128,160,1))
        next_history = np.append(next_state, history[:,:,:,:5],axis = 3)

        history = next_history
        rate.sleep()
