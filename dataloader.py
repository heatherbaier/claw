from torchvision import models, transforms
from collections import namedtuple, deque
from ViewBox import *
import torch
import json
import cv2
import os

class Dataset():

    def __init__(self, batch_size, imagery_dir, json_path):

        self.batch_size = batch_size
        self.imagery_dir = imagery_dir

        with open(json_path, "r") as f:
            self.ys = json.load(f)

        self.Batch = namedtuple('Batch', ('images', 'vbs', 'ys'))

        self.data = []
        self.to_tens = transforms.ToTensor()
        self.load_data()

        # self.data = [self.data[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]

        # self.train_dl = torch.utils.data.DataLoader(self.data, batch_size = self.batch_size, shuffle = True)

        # print(self.data)


    def load_data(self):

        print("BATCH SIZE: ", self.batch_size)

        count = 0

        for impath in os.listdir(self.imagery_dir):

            print(impath)

            print("COUNT: ", count, "  |  COUNT % BATCH SIZE: ", count % self.batch_size)

            # If have counted up to the amount of a batch, reset all of the lists
            if (count % self.batch_size) == 0:
                ims, vbs, ys = [], [], []

            # Load in and prep all of the data
            im = cv2.imread(os.path.join(self.imagery_dir, impath))
            im = self.to_tens(im).unsqueeze(0)

            ims.append(im)
            vbs.append(ViewBox(image = im, name = impath))
            ys.append(self.ys[impath.replace(".png", "")])

            # If we are at the last element before we reset the lists, create a Batch 
            # namedtuple from our current lists and append them to our dataset
            if (count % self.batch_size) == self.batch_size - 1:
                # print("ADDING DATA")
                cur_batch = self.Batch(ims, vbs, ys)
                self.data.append(cur_batch)

            count += 1



"""
A TENSOR IS C, H, W
"""