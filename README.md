# Collision-avoidance
Towards Monocular Vision Based Collision Avoidance Using Deep Reinforcement Learning

### Depth Estimation
- Any tensorflow version above 1.10.0 would be fine
- Depth Estimation model is based on ResNet 50 architecture
- python file that contains the model architecture is located in **models**
- Due to huge size of trained depth estimation model, you have to download the depth estimation model [here](https://github.com/iro-cp/FCRN-DepthPrediction).

To implement the code, you need
```
- fcrn.py
- __init__.py
- network.py
- NYU_FCRN-chekpoint
```
### Training Environment in Robot Operating System
- In our setup, **ubuntu 16.04** and **[ROS kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)** are used
- Training env file contains the figures of the training map in ROS
- You could use the training environments in **model_editor_models**
- Place **editor models** in your gazebo_model_repository

### Training Environment Setup
#### 1. Spawning the drone for training

Training agent for the drone is [hector_qaudrotor](http://wiki.ros.org/hector_quadrotor). Please take a look at the ROS description and install it.
To spawn the training agent for our setup, type the command below:
```
roslaunch spawn_quadrotor_with_asus_with_laser.launch
```
To enable motors, type the command below:
```
rosservice call /enable_motors true
```
#### 2. Setting the initial position and velocity of the agent
You could change the initial position and velocity in the ENV.py. 
- To change the spawining position of the drone, change the **spawn_table** in the ENV.py
- To change the velocity of the drone, change the action_table: (three linear speed, five angular rate)

### Training 
To train the model `python3 D3QN.py`. You could change the hyperparameters in the D3QN.py.
### Testing
To test the model, please change the trained model's directory and then type `python3 Test.py`.
