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

from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


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

policy_net = DQN(128, 128, n_actions).to(device)
target_net = DQN(128, 128, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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

        # Set up the view box object that the agent controls
        self.view_box = ViewBox(self.observation_shape)

        # init the canvas
        self.canvas = cv2.imread("./test_image.png")

        # This is currently the model that predicts for a singular square of landsat (BUT EVENTUALLY MODIS), the number of migrants
        # NEED TO THE TEST JUST USING TH EPOLICY NET'S CONVOLUTIONAL LAYERS TO DO THIS
        self.critic = mig_model(models.resnet18(pretrained = True))
        self.critic.fc = torch.nn.Linear(512, 1)
        self.mig_optim = torch.optim.Adam(self.critic.parameters(), lr = 0.01)
        self.mig_criterion = torch.nn.L1Loss()     

        # This is the model that stacks the selected squares together into a final prediction
        # THIS NEEDS A LOT OF WORK
        self.rnn = lstm()
        self.rnn_optim = torch.optim.Adam(self.rnn.parameters(), lr = 0.01)
        self.rnn_criterion = torch.nn.L1Loss()   

        # OBVIOUSLY TAKE THIS OUT WHEN YOU START REAL TRAINING
        self.y_val = torch.tensor([[420]])
        self.error = 0
        self.mig_pred = 0

        self.to_tens = transforms.ToTensor()

        # Variable initlialization
        self.n_actions = self.action_space.n
        self.device = "cpu"
        self.eps_threshold = .9
        self.epoch = 0
        self.BATCH_SIZE = 1
        self.GAMMA = 0.999
        self.EPS_START = .9
        self.EPS_END = 0.05
        self.EPS_DECAY = 600
        self.TARGET_UPDATE = 10
        self.steps_done = 0
        self.total_moves = 0


    def optimize_model(self):

        """
        Function to optimize the policy_net based on saved ReplayMemory
        """

        if len(memory) < self.BATCH_SIZE:
            return

        transitions = memory.sample(self.BATCH_SIZE)
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
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * SELF.GAMMA) + reward_batch

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

        """
        Function to select either a random action or computer it using the policy_net
        """

        global steps_done

        # Read in the image and convert it to a tensor
        state = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
        
        # Get a random number between 0 & 1
        sample = random.random()

        # Calculate the new epsilon threshold
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # If the random number is larger than eps_threshold, use the trained policy to pick an action
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print("Computed action!", self.eps_threshold)
                return policy_net(state).max(1)[1].view(1, 1)

        # Otherwise, pick a random action
        else:
            print("Random action! Epsilon = ", self.eps_threshold)
            return torch.tensor([[random.randrange(self.n_actions)]], device = self.device, dtype=torch.long)
            

    def draw_elements_on_canvas(self, red = True):

        """ 
        Function to draw all the Landsat info and the current model status information on the screen
        """

        # Sert up the canvas with the initial Landsat image
        self.canvas = cv2.imread("./test_image.png")

        # If the action was a select, the box will draw as green, otherwise it'll draw as red
        if red:
            self.canvas = cv2.rectangle(self.canvas, self.view_box.start_point, self.view_box.end_point, self.view_box.color, self.view_box.thickness)        
        else:
            self.canvas = cv2.rectangle(self.canvas, self.view_box.start_point, self.view_box.end_point, (0,255,0), self.view_box.thickness)        

        # Set up a list of text variables and their screen locations
        text = ["Epoch: {}".format(self.epoch), 'Grabs Left: {}'.format(self.grabs_left), "Current Prediction: {} migrants".format(int(self.mig_pred)), "Current Error: {} migrants".format(int(self.error)), "Epsilon: {}".format(self.eps_threshold), "Land Cover %: {}".format(0)]
        locs = [(50,50), (50,90), (50,130), (50,170), (50,210)]

        # Then interate over the lists and put the information on the canvas
        for i in zip(text, locs):
            self.canvas = cv2.putText(self.canvas, i[0], i[1], font,  
                    1.2, (255, 255, 255), 1, cv2.LINE_AA)

    def reset(self, epoch):

        # Set up a blank (black) canvas so we can put summarized Epoch information on it
        self.canvas = np.zeros((self.observation_shape[0], self.observation_shape[1], 3))

        # Set up a list of text variables summarizing the Epoch and their screen locations
        text = ["Epoch: {}".format(self.epoch), "Epoch Predicted # Migrants: {}".format(int(self.mig_pred)), "Epoch Total Error: {} migrants".format(int(self.error)), "Epsilon: {}".format(self.eps_threshold), "Land Cover %: {}".format(0)]
        locs = [(50,50), (50,90), (50,130), (50,170), (50,210)]

        # Then interate over the lists and put the information on the canvas
        for i in zip(text, locs):
            self.canvas = cv2.putText(self.canvas, i[0], i[1], font,  
                    1.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Render the Epoch summary screen and make it wait a sec so user can read it
        self.render()
        time.sleep(4)

        # Reset all of the tracking variables
        self.first_grab = True
        self.mig_pred = 0
        self.error = 0
        self.epoch = epoch
        self.grabs_left = self.max_grabs

        # Reset the viewbox to its inital position
        self.view_box = ViewBox(self.observation_shape)

        # Reset the canvas to the Landsat image
        self.canvas = cv2.imread("./test_image.png")

        # Draw the elements on the canvas
        self.draw_elements_on_canvas()

        # Return the observation
        return self.canvas


    def render(self, mode = "human"):
        """
        Function to render everything on the screen to the user
        """
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.canvas

    def get_action_meanings(self):
        """
        Dictionary of action meanings
        """
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Select"}
    
    def close(self):
        """
        Destroys the interactive window
        """
        cv2.destroyAllWindows()


    def step(self, action):

        """
        Function to handle what happens each time the agent makes a move
        """

        # Assert that the action is valid
        assert self.action_space.contains(action), "Invalid Action"

        # Flag that marks the termination of an episode
        done = False        

        # If the action is a select...
        if action == 4:

            # Update the number of selects left
            self.grabs_left -= 1

            # Move the box according to the action
            self.view_box.move_box(action)

            # Get the new screen and extract the landsat from that area
            new_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            
            # Run the features through the feature extractor to get the FC layer of shape [1,512], then unsqueeze it to [1,1,512]
            fc_layer, _ = self.critic(new_screen)
            print("CRITIC PREDICTION: ", _.item())
            fc_layer = fc_layer.unsqueeze(0)

            # Save the previous prediction of the LSTM so we can use it to calculate the reward
            prev_pred = self.mig_pred

            # If it's the first grab, don't use the hidden layers (but calculate them)
            if self.first_grab:
                mig_pred, self.hidden = self.rnn(fc_layer)

            # If it's not the first grab, then use the hidden layers
            else:
                mig_pred, self.hiddden = self.rnn(fc_layer, self.hidden)

            # self.hidden = tuple([each.data for each in self.hidden])

            print("RNN MIG PRED: ", mig_pred, self.y_val)

            # Update the new error
            self.error = mig_pred - self.y_val

            # Calculate the loss and ~optimize~
            # YEAH THIS DEF ALL NEEDS TO BE UPDATED/FIXED
            rnn_loss = self.rnn_criterion(mig_pred.squeeze(0), self.y_val)
            rnn_loss.backward()
            self.rnn_optim.step()
            self.rnn_optim.zero_grad() 

            # De-tensorize lol
            self.mig_pred = mig_pred.item()

            # Update the canvas, but draw the box as green since it was a select
            self.draw_elements_on_canvas(red = False)
     
            # If there are no grabs left, update the canvas with this steps results, set the done flag to True & return 
            if self.grabs_left == 0:
                self.draw_elements_on_canvas()
                done = True
                return [1,2,done,4]

            # If there are still more grabs left, calculate the reward and return not done
            else:

                print("OVERALL MIG PRED: ", self.mig_pred)

                if abs(self.y_val - prev_pred) > abs(self.y_val - self.mig_pred):
                    reward = 20
                else:
                    reward = 0
                
                self.first_grab = False

                return [1,2,done,4]


                

        # If the action is just a simple move...
        elif action != 4:

            # Update total number of moves (not really important to acutal model training)
            self.total_moves += 1

            # Get the screen & the prediction for the current state before you take an action
            current_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            _, mig_pred_t1 = self.critic(current_screen)
            
            # Now take the action and update the view_boxes position (and therefore our state)
            self.view_box.move_box(action)

            # Draw pretty
            self.draw_elements_on_canvas()

            # Get the screen & the prediction for the current state before you take an action
            new_screen = self.to_tens(self.view_box.clip_image(cv2.imread("./test_image.png"))).unsqueeze(0)
            _, mig_pred_t2 = self.critic(new_screen)

            # If the screen after the action was taken is closer to the true value than before, give the model a reward
            if abs(self.y_val - mig_pred_t1) > abs(self.y_val - mig_pred_t2):
                reward = 10
            else:
                reward = 0

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


