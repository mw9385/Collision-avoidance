#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ENV import ENV
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers.convolutional import Conv2D
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential ,load_model, Model
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Flatten, Lambda, add
from sensor_msgs.msg import LaserScan, Image, Imu
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Vector3Stamped, Twist
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from PIL import Image as iimage

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

laser = None
velocity = None
vel = None
theta = None
pose = None
orientation = None
image = None
depth_img = None

def DepthCallBack(img):
    global depth_img
    depth_img = img.data

def callback_laser(msg):    
    global laser
    laser = msg    
    laser = laser.ranges 
    
def callback_camera(msg):
    global image
    image = np.frombuffer(msg.data, dtype=np.uint8)    
    image = np.reshape(image, [480,640,3]) 
    image = np.array(image)

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

def callback_laser(msg):    
    global laser
    laser = msg    
    laser = laser.ranges 

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
    #image = cv2.resize(image, (256*2, 320*2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('input image', image)    
    cv2.waitKey(1)
    
    
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

class D3QNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # state and size of actiojn
        self.state_size = (128, 160 , 6)
        self.action_size = action_size
        # D3QN hyperparameters
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.0001
        self.exploration_steps = 100000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 16
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # Replay memory size, maximum: 50000
        # This could be changed depends on the RAM size of your computer
        self.memory = deque(maxlen=50000)
        self.no_op_steps = 5
        # Target network initialization
        # Build model using multi-GPU, if you want to train it with single-GPU changed the code
        # self.model = self.build_model()
        # self.target_model = self.build_model()

        self.model = multi_gpu_model(self.build_model(), gpus = 2)
        self.target_model = multi_gpu_model(self.build_model(), gpus = 2)
        self.update_target_model()        
        self.optimizer = self.optimizer()

        # Tensorboard settings
        self.config = tf.ConfigProto()                
        self.sess = tf.InteractiveSession(config=self.config)         
        self.sess.run(tf.global_variables_initializer())
        K.set_session(self.sess)    

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op =             self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/D3QN', self.sess.graph)        
        
        if self.load_model:
            self.model.load_weights("./save_model/D3QN.h5")

        
    # Defining optimizer to utilize Huber Loss function
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        
        optimizer = Adam(lr=0.0001, epsilon=0.01)  
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train


    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    # dueling network's Q Value is sum of advantages and state value
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
    # Update target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Action selection through epsilon-greedy search
    def get_action(self, history):
        flag = False
        history = np.float32(history)
        if np.random.rand() <= self.epsilon:
            flag = True
            return random.randrange(self.action_size), flag
        else:
            flag = False
            q_value = self.model.predict(history)
            return np.argmax(q_value[0]), flag

    # Store samples in the order of [s,a,r,s'] into the replay memory
    def append_sample(self, history, action, reward, next_history, deads):
        self.memory.append((history, action, reward, next_history, deads))

    # Train the model by using the ramdomly extracted samples in the replay-memory
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, deads = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0]) 
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            deads.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)
        
        # like Q Learning, get maximum Q value at s'
        # But from target model
        
        for i in range(self.batch_size):
            if deads[i]:
                target[i] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                
                target[i] = reward[i] + self.discount_factor * target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
        
    # store the reward and Q-value on every episode
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

if __name__ == '__main__':
    # Check the gazebo connection
    rospy.init_node('env', anonymous=True)
    
    # Class define
    env = ENV()   
    # Parameter setting for the simulation
    agent = D3QNAgent(action_size = 8)
    EPISODE = 100000    
    global_step = 0 
    # Observe
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback_camera,queue_size = 5)    
    rospy.Subscriber("/scan", LaserScan, callback_laser,queue_size = 5)
    rospy.Subscriber('gazebo/model_states', ModelStates, state_callback, queue_size= 5)      
    rospy.Subscriber('/camera/depth/image_raw', Image, DepthCallBack,queue_size = 5)
       
    # define command step
    rospy.sleep(2.)             
    rate = rospy.Rate(5)
    
    # define episode and image_steps
    e = 0
    image_steps = 0  
    collision_steps = 0
    env.reset_sim(pose, orientation) 

    while e < EPISODE and not rospy.is_shutdown():
        e = e + 1        
        env.reset_sim(pose, orientation)     
        # get the initial state     
        state = GetDepthObservation(image)
        history = np.stack((state, state, state, state, state, state), axis = 2)                
        history = np.reshape([history], (1,128,160,6))
        
        laser_distance = np.zeros([1,2])
        delta_depth = 0
        step, score  = 0. ,0.
        done = False                    
                        
        while not done and not rospy.is_shutdown():  
            # wait for service
            rospy.wait_for_message('/camera/rgb/image_raw', Image)
            rospy.wait_for_message('/camera/depth/image_raw', Image)
            rospy.wait_for_message('/gazebo/model_states', ModelStates)
            rospy.wait_for_message('/scan', LaserScan)           
                
            global_step = global_step + 1             
            step = step + 1                       
            
            # Receive the action command from the Q-network and do the action                        
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
            laser_distance = np.append(next_distance, laser_distance[0])

            # Calculate the average_q_max
            agent.avg_q_max += np.amax(agent.model.predict(history)[0])                                                                                                
            agent.append_sample(history,action,reward,next_history, done)
            history = next_history

            if e >= agent.no_op_steps:
                agent.train_model()
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()      
            
            if step >= 2000:
                done = True           
                
            # Update the score
            score +=reward   
            rate.sleep()            
                        
        if done:    
            if e >= agent.no_op_steps:                    
                stats =[score, agent.avg_q_max/float(step),step, agent.avg_loss/ float(step)]    
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={agent.summary_placeholders[i]: float(stats[i])})

                summary_str = agent.sess.run(agent.summary_op)
                agent.summary_writer.add_summary(summary_str, e - agent.no_op_steps)
               
            print("episode: " + str(e), 
                  "score: " + str(score), 
                  " memory length: " + str(len(agent.memory)),
                  "epsilon: " + str(agent.epsilon), 
                  " global_step:"+ str(global_step),
                  "average_q:" + str(agent.avg_q_max/float(step)),
                  "average loss:" + str(agent.avg_loss / float(step)), 
                  'step:' + str(step))

            agent.avg_q_max, agent.avg_loss = 0,0

        if e % 50 == 0:
            agent.model.save_weights("./save_model/D3QN_V3.h5")            
