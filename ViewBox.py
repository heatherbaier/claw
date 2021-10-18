from torchvision import models, transforms
from collections import namedtuple, deque
import matplotlib.patches as patches
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym import Env, spaces
import PIL.Image as Image
from copy import deepcopy
import rasterio as rio
import torch.nn as nn
import numpy as np 
import random
import torch
import time
import math
import cv2 
import gym

from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


class ViewBox(object):

    """ 
    used to define any arbitrary point on our observation image
    (x,y): position of the point on the image
    (x_min, x_max, y_min, y_max): permissible coordinates for the point. if you try to set 
                                  posiiton outisde of the limits, the position values are clmaped.
    name: name of the point
    """
    
    def __init__(self, obs_shp):

        self.obs_shp = obs_shp
        self.x = int(self.obs_shp[1] / 2)
        self.y = int(self.obs_shp[0] / 2)
        self.color = (255, 0, 0) # Blue color in BGR
        self.thickness = 2 # Line thickness of 2 px
        self.radius = 64
        self.start_point = (self.x - self.radius, self.y + self.radius)
        self.end_point = (self.x + self.radius, self.y - self.radius)     
        self.x_min = int(self.radius)
        self.x_max = int(self.obs_shp[1] - self.radius)
        self.y_min = int(self.radius)
        self.y_max = int(self.obs_shp[0] - self.radius)

    def get_position(self):
        return (self.x, self.y)

    def move_box(self, a):

        """ moves the box in a given direction based on the action value """

        if a == 0: # MOVE BOX DOWN
            self.y -= self.radius
        elif a == 1: # MOVE BOX UP
            self.y += self.radius
        elif a == 2: # MOVE BOX LEFT
            self.x += self.radius
        elif a == 3: # MOVE BOX RIGHT
            self.x -= self.radius
        elif a == 4: # GET SCREEN CENTER
            print("Select!")
            # pass for now

        self.x = self.clamp(self.x, self.x_min, self.x_max)
        self.y = self.clamp(self.y, self.y_min, self.y_max)        

        print("moved: ", self.x, self.y)

        self.start_point = (self.x - self.radius, self.y + self.radius)
        self.end_point = (self.x + self.radius, self.y - self.radius)   

    def clamp(self, n, minn, maxn):
        """ clamp box to bounds of image """
        return max(min(maxn, n), minn)

    def clip_image(self, im):
        """ Function to send back just the view_box.radius*2 x view_box.radius*2 square around the current image coordinate """
        return im[self.x - self.radius:self.x + self.radius, self.y - self.radius:self.y + self.radius, :]