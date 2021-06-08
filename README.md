# Collision-avoidance
Towards Monocular Vision Based Collision Avoidance Using Deep Reinforcement Learning

### Depth Estimation
- Any version of tensorflow above 1.10.0 would be fine
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
- In our setup, **ubuntu 16.04** and **ROS kinetic** are used
- Training env file contrains the figures of the training map in ROS
- You could use the training environments in **model_editor_models**
