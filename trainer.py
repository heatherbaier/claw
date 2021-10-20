from torchvision import models, transforms
from collections import namedtuple, deque
from gym import Env, spaces
import torch.nn as nn
import numpy as np 
import random
import torch
import time
import math
import cv2 

from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


class Trainer():

    def __init__(self, num_channels, 
                       num_actions, 
                       model,
                       criterion,
                       optimizer,
                       memory,
                       Transition,
                       train_dl,
                       device,
                       max_grabs = 5,
                       epochs = 1):

        super(Trainer, self).__init__()

        # Define an action space range from 0 to 4
        self.action_space = spaces.Discrete(num_actions,)

        # Maximum number of grabs the model can take
        self.max_grabs = max_grabs

        # Variable initlialization
        self.n_actions = self.action_space.n
        self.device = "cpu"
        self.eps_threshold = .9
        self.epoch = 0
        self.BATCH_SIZE = 4
        self.GAMMA = 0.999
        self.EPS_START = .9
        self.EPS_END = 0.05
        self.EPS_DECAY = 600
        self.TARGET_UPDATE = 10
        self.steps_done = 0
        self.epochs = epochs

        # Model setup
        self.policy_net = model
        self.target_net = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = memory
        self.Transition = Transition

        # Store data
        self.train_dl = train_dl



    def reset(self, epoch):

        # Reset all of the tracking variables
        self.first_grab = True
        self.mig_pred = 0
        self.error = 0
        self.epoch = epoch
        self.grabs_left = self.max_grabs
        self.grab_vectors = []

        """ TO-DO: WHEN THE DONE FLAG IS LIFTED FOR AN OBSERVATION 
            WITHIN A BATCH, RANDOMIZE ITS VIEWBOX LOCATION THEN """


    def select_action(self, state):

        """
        Function to select either a random action or computer it using the policy_net
        """

        global steps_done

        # Get a random number between 0 & 1
        sample = random.random()

        # Calculate the new epsilon threshold
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # If the random number is larger than eps_threshold, use the trained policy to pick an action
        if sample > self.eps_threshold:
            with torch.no_grad():
                print("Computed action!", self.eps_threshold)
                return self.policy_net(state)[0].max(1)[1].view(1, 1)

        # Otherwise, pick a random action
        else:
            print("Random action! Epsilon = ", self.eps_threshold)
            return torch.cat([torch.tensor([[random.randrange(self.n_actions)]]) for i in range(0, state.shape[0])])


    def step(self, action, initial_state, vb, y_val):

        """
        Function to handle what happens each time the agent makes a move
        """

        # Assert that the action is valid
        # assert self.action_space.contains(action), "Invalid Action"

        # Flag that marks the termination of an episode
        done = False        

        # If the action is a select...
        if action == 4:

            # pass

            # Update the number of selects left
            vb.update_grabs_left()

            print("GRABS LEFT: ", vb.grabs_left)

            # Get the new screen and extract the landsat from that area
            new_screen = vb.clip_image()
            
            if len(self.grab_vectors) == 0:
                _, mig_pred, fc_layer = self.policy_net(new_screen, seq = None, select = True)
            else:
                seq = torch.cat(self.grab_vectors, dim = 1)
                print("SEQUENCE SHAPE: ", seq.shape)
                _, mig_pred, fc_layer = self.policy_net(new_screen, seq = seq, select = True)

        #     self.grab_vectors.append(fc_layer.detach())

        #     # Calculate the loss and ~optimize~
        #     mig_loss = self.criterion(mig_pred, self.y_val)
        #     optimizer.zero_grad()
        #     mig_loss.backward()
        #     optimizer.step() 

        #     # Save the previous prediction of the LSTM so we can use it to calculate the reward
        #     prev_pred = self.mig_pred

        #     print("RNN MIG PRED: ", mig_pred, self.y_val)

        #     # Update the new error
        #     self.error = mig_pred - self.y_val

        #     # De-tensorize (lol words)
        #     self.mig_pred = mig_pred.item()

        #     # Update the canvas, but draw the box as green since it was a select
        #     self.draw_elements_on_canvas(red = False)
     
        #     # If there are no grabs left, update the canvas with this steps results, set the done flag to True & return 
        #     if self.grabs_left == 0:

        #         self.draw_elements_on_canvas()
        #         done = True
        #         return [1,2,done,4]

        #     # If there are still more grabs left, calculate the reward and return not done
        #     else:

        #         print("OVERALL MIG PRED: ", self.mig_pred)

        #         if abs(self.y_val - prev_pred) > abs(self.y_val - self.mig_pred):
        #             reward = 20
        #         else:
        #             reward = 0
                
        #         self.first_grab = False

        #         return [1,2,done,4]
                

        # If the action is just a simple move...
        elif action != 4:

            # Update total number of moves (not really important to acutal model training)
            # self.total_moves += 1

            # Get the screen & the prediction for the current state before you take an action
            current_screen = initial_state

            # print("current_screen shape: ", current_screen.shape)

            if len(self.grab_vectors) == 0:
                _, mig_pred_t1 = self.policy_net(current_screen, seq = None)
            else:
                seq = torch.cat(self.grab_vectors, dim = 1)
                _, mig_pred_t1 = self.policy_net(current_screen, seq = seq)

            # print("OLD PRED: ", mig_pred_t1)

            # self.update_mig_weights(mig_pred_t1)

            # print("OLD POSITION: ", vb.get_position())
            
            # Now take the action and update the view_boxes position (and therefore our state)
            vb.move_box(action)

            # print("NEW POSITION: ", vb.get_position())

            # # Draw pretty
            # self.draw_elements_on_canvas()

            # # Get the screen & the prediction for the current state before you take an action
            new_screen = vb.clip_image()

            # print("NEW SCREEN SHAPE: ", new_screen.shape)

            if len(self.grab_vectors) == 0:
                _, mig_pred_t2 = self.policy_net(current_screen, seq = None)
            else:
                seq = torch.cat(self.grab_vectors, dim = 1)
                _, mig_pred_t2 = self.policy_net(current_screen, seq = seq)

            """ FIX THIS LATER ON YO """
            # self.update_mig_weights(pred = mig_pred_t1, target = y_val)
            self.update_mig_weights(pred = mig_pred_t2, target = y_val)

            # If the screen after the action was taken is closer to the true value than before, give the model a reward
            if abs(y_val - mig_pred_t1) > abs(y_val - mig_pred_t2):
                reward = 10
            else:
                reward = 0


            return [1,reward,done,4,vb]


    def update_mig_weights(self, pred, target):

        target = torch.tensor([[target]])
        pred = pred.squeeze(0)

        # print("PRED: ", pred, "  |  TARGET: ", target)

        mig_loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        mig_loss.backward()
        self.optimizer.step() 


    def train(self):

        for epoch in range(0, self.epochs):
            self.reset(epoch = epoch)
            self.train_one_epoch(epoch = epoch)




    def train_one_epoch(self, epoch):

        for b, (batch) in enumerate(self.train_dl):

            print(batch.vbs)

            done = False

            print([i.obs_shp for i in batch.vbs])

            # Convert states and target outputs into a batch
            states = torch.cat([i.clip_image() for i in batch.vbs])
            targets = torch.tensor(batch.ys).view(-1, 1)

            while not done:

                # Select actions for each iamges in the batch
                actions = self.select_action(states)

                print(actions)

                # Get the current states prior to taking the above action
                current_state = states

                # Calculate the state-action pair's reward and done flag
                _, reward, done, _, batch.vbs[0] = self.step(actions, current_state, batch.vbs[0], batch.ys[0])
                

            print(actions)

        # Reset the 'game' paremeters after each batch of images is played
        self.reset(epoch = epoch)



