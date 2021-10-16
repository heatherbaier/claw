import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import rasterio as rio
import gym
import random

from gym import Env, spaces
import time

import matplotlib.patches as patches

from models import *

from torchvision import models, transforms

from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from copy import deepcopy

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

torch.autograd.set_detect_anomaly(True)




BATCH_SIZE = 1
GAMMA = 0.999
EPS_START = .9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))






class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




n_actions = 4
device = "cpu"

policy_net = DQN(64, 64, n_actions).to(device)
target_net = DQN(64, 64, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class EarthObs(Env):

    def __init__(self, num_channels, num_actions):
        super(EarthObs, self).__init__()

        """ define obervation and action spaces in here"""

        # define a 2-d observation space
        self.observation_shape = (953, 1240, 3)
        self.obseervation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                             high = np.ones(self.observation_shape),
                                             dtype = np.float16)

        # define an action space range from 0 to 4
        self.action_space = spaces.Discrete(num_actions,)

        # create a canvas to render the environemnt images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # maximum number of grabs the model can take
        self.max_grabs = 5

        # permissible area a helicopter can be in
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

        self.view_box = ViewBox(self.observation_shape)

        # init the canvas
        self.canvas = cv2.imread("./test_image.png")

        # self.actor = models.resnet18()
        # self.actor.fc = torch.nn.Linear(512, 1)
        # self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.01)
        # self.criterion = torch.nn.L1Loss()

        self.critic = mig_model(models.resnet18(pretrained = True))
        self.critic.fc = torch.nn.Linear(512, 1)
        self.mig_optim = torch.optim.Adam(self.critic.parameters(), lr = 0.01)
        self.mig_criterion = torch.nn.L1Loss()     

        self.rnn = lstm()
        self.rnn_optim = torch.optim.Adam(self.rnn.parameters(), lr = 0.01)
        self.rnn_criterion = torch.nn.L1Loss()   

        self.to_tens = transforms.ToTensor()

        self.y_val = torch.tensor([[420]])

        self.sequence_preds = []

        self.n_actions = self.action_space.n

        self.device = "cpu"

        self.error = 0

    def optimize_model(self):

        print("Optimizing model!")

        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.sample(BATCH_SIZE)
        # print("transitions: ", transitions)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        print("Batch: ", batch)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
                                                    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        print("action_batch: ", action_batch)
        print("reward_batch: ", reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        print("state_action_values: ", state_action_values)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()






    def select_action(self):
        global steps_done
        state = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print("Computed action!", eps_threshold)
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            print("Random action! Epsilon = ", eps_threshold)
            return torch.tensor([[random.randrange(self.n_actions)]], device = self.device, dtype=torch.long)
            


    def get_screen(self):
        return self.view_box.clip_image(cv2.imread("./test_image.png"))


    def draw_elements_on_canvas(self, red = False):

        self.canvas = cv2.imread("./test_image.png")

        if not red:
            self.canvas = cv2.rectangle(self.canvas, self.view_box.start_point, self.view_box.end_point, self.view_box.color, self.view_box.thickness)        
        else:
            self.canvas = cv2.rectangle(self.canvas, self.view_box.start_point, self.view_box.end_point, (0,255,0), self.view_box.thickness)        

        text = ['Grabs Left: {}'.format(self.grabs_left), "Current Prediction: {} migrants".format(int(self.mig_pred)), "Current Error: {} migrants".format(int(self.error)), "Land Cover %: {}".format(0)]
        locs = [(50,50), (50,100), (50,150)]

        # put the info on the canvas
        for i in zip(text, locs):
            self.canvas = cv2.putText(self.canvas, i[0], i[1], font,  
                    1.5, (255, 255, 255), 1, cv2.LINE_AA)

    def reset(self):

        # reset the fuel consumed
        self.grabs_left = self.max_grabs

        self.first_grab = True
        self.mig_pred = torch.tensor([0])

        # Reset the viewbox to its inital position
        self.view_box = ViewBox(self.observation_shape)

        # reset the canvas
        self.canvas = cv2.imread("./test_image.png")

        # draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas


    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.canvas

    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Select"}
    
    def close(self):
        cv2.destroyAllWindows()


    def step(self, action):

        # Assert that the action is valid
        assert self.action_space.contains(action), "Invalid Action"

        # Flag that marks the termination of an episode
        done = False        

        if action == 4:

            print("ACTION IS 4 SO SELECTING!")

            self.grabs_left -= 1

            # print("GRAB NUMBER: ", self.max_grabs - self.grabs_left)

            self.view_box.move_box(action)

            # Get the new screen and extract the landsat from that area
            new_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            
            # Runthe features through the feature extractor to get the FC layer of shape [1,512], then unsqueeze to [1,1,512]
            fc_layer, _ = self.critic(new_screen)
            print("CRITIC PREDICTION: ", _.item())
            fc_layer = fc_layer.unsqueeze(0)

            # print("FULLY CONNECTED LAYER SHAPE: ", fc_layer.shape)

            # Save the previous prediction so we can use it to calculate the reward
            prev_pred = self.mig_pred

            # If it's the first grab, don't use the hidden layers (but calculate them)
            if self.first_grab:
                mig_pred, self.hidden = self.rnn(fc_layer)

            # If it's not the first grab, then use the hidden layers
            else:
                mig_pred, self.hiddden = self.rnn(fc_layer, self.hidden)

            # self.hidden = tuple([each.data for each in self.hidden])

            print("RNN MIG PRED: ", mig_pred, self.y_val)

            self.error = self.y_val - mig_pred

            # Calculate the loss and ~optimize~
            rnn_loss = self.rnn_criterion(mig_pred.squeeze(0), self.y_val)
            rnn_loss.backward()
            self.rnn_optim.step()
            self.rnn_optim.zero_grad() 

            self.draw_elements_on_canvas(red = True)
     
            if self.grabs_left == 0:

                self.first_grab = True
                self.draw_elements_on_canvas()
                done = True

                self.mig_pred = 0
                self.error = 0

                return [1,2,done,4]

            else:

                # rnn_loss.backward(retain_graph=True)

                self.mig_pred = mig_pred.item()

                print("OVERALL MIG PRED: ", self.mig_pred)

                # reward = prev_pred

                if abs(self.y_val - prev_pred) > abs(self.y_val - self.mig_pred):
                    reward = 20
                else:
                    reward = 0
                
                self.first_grab = False

                return [1,2,done,4]


                


        if action != 4:


            # Get the screen & the prediction for the current state before you take an action
            current_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            _, mig_pred_t1 = self.critic(current_screen)

            # self.update_mig_weights(val = mig_pred_t1)
            
            # Now take the action and update the view_boxes position (and therefore our state)
            self.view_box.move_box(action)

            # Draw pretty
            self.draw_elements_on_canvas()
            # time.sleep(.5)

            # Get the screen & the prediction for the current state before you take an action
            new_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            _, mig_pred_t2 = self.critic(new_screen)
            
            # self.update_mig_weights(val = mig_pred_t2, send = True)
            # self.mig_preds.append(mig_pred_t2)

            # If the screen after the action was taken is closer to the true value than before,
            # give the model a reward
            if abs(self.y_val - mig_pred_t1) > abs(self.y_val - mig_pred_t2):
                reward = 10
            else:
                reward = 0


            # print("t1 pred: ", mig_pred_t1, "  |  t2 pred: ", mig_pred_t2, "  |  reward: ", reward)

            return [1,reward,done,4]


    def update_mig_weights(self, val = None, send = False):

        if not send:

            mig_loss = self.mig_criterion(val, self.y_val)
            # print("Migration Loss: ", mig_loss.item())
            mig_loss.backward()

        else:

            mig_loss = self.mig_criterion(val, self.y_val)
            # print("Migration Loss: ", mig_loss.item())
            mig_loss.backward()
            self.mig_optim.step()
            self.mig_optim.zero_grad()            



        # for i in self.mig_preds:

        #     print(i)

        #     mig_loss = self.mig_criterion(i, self.y_val)
        #     print("Migration Loss: ", mig_loss.item())

        #     self.mig_optim.zero_grad()
        #     mig_loss.backward()
        #     self.mig_optim.step()



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
        self.radius = 32
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
        # return im[:, self.y - self.radius:self.y + self.radius, self.x - self.radius:self.x + self.radius]
        return im[self.x - self.radius:self.x + self.radius, self.y - self.radius:self.y + self.radius, :]