from torchvision import models, transforms
from collections import namedtuple, deque
from ViewBox import *
import torch
import json
import cv2
import os

from earth_env import *


class Dataset():

    def __init__(self, 
                 batch_size, 
                 imagery_dir, 
                 json_path, 
                 valid = False,
                 BATCH_SIZE = 4,
                 GAMMA = 0.999,
                 EPS_START = .9,
                 EPS_END = 0.05,
                 EPS_DECAY = 200,
                 TARGET_UPDATE = 10,
                 steps_done = 0,
                 total_moves = 0):

        self.batch_size = batch_size
        self.imagery_dir = imagery_dir
        self.valid = valid

        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE

        with open(json_path, "r") as f:
            self.ys = json.load(f)

        self.Batch = namedtuple('Batch', ('images', 'environment', 'ys'))

        self.data = []
        self.to_tens = transforms.ToTensor()
        self.load_data()


    def load_data(self):

        count = 0

        for impath in os.listdir(self.imagery_dir):

            # If have counted up to the amount of a batch, reset all of the lists
            if (count % self.batch_size) == 0:
                cur_data = []

            # Load in and prep all of the data
            im = cv2.imread(os.path.join(self.imagery_dir, impath))

            cur_data.append((os.path.join(self.imagery_dir, impath), 
                             self.ys[impath.replace(".png", "")], 
                             EarthObs(impath = os.path.join(self.imagery_dir, impath), 
                                      y_val = self.ys[impath.replace(".png", "")], 
                                      num_channels = 3, 
                                      num_actions = 5, 
                                      display = True, 
                                      valid = self.valid,
                                      BATCH_SIZE = self.BATCH_SIZE,
                                      GAMMA = self.GAMMA,
                                      EPS_START = self.EPS_START,
                                      EPS_END = self.EPS_END,
                                      EPS_DECAY = self.EPS_DECAY,
                                      TARGET_UPDATE = self.TARGET_UPDATE)))

            # If we are at the last element before we reset the lists, create a Batch 
            # namedtuple from our current lists and append them to our dataset
            check_val = count % self.batch_size
            desired_val = self.batch_size - 1
            if (count % self.batch_size) == (self.batch_size - 1):
            # if check_val == desired_val:
                self.data.append(cur_data)


            count += 1




"""
A TENSOR HAS SHAPE B, C, H, W
"""