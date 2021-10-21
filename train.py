from torchvision import models, transforms
import torch
import json
import cv2 

from earth_env import *
from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


def train(env, display = True):

    for epoch in range(0, 10):

        # Reset the environment to the beginning
        obs = env.reset(epoch = epoch)

        # Set done flag
        done = False

        # While there are still selects left...
        while not done:

            # Select and perform an action
            action = env.select_action()

            # Save current state so we can push it to memory in a couple lines
            current_state = env.view_box.clip_image(cv2.imread(impath))

            # Calculate the state-action pair's reward and done flag
            mp, reward, done, _ = env.step(action.item())

            # Get the new state post action
            next_state = env.view_box.clip_image(cv2.imread(impath))

            # Push all of the goodness to memory
            memory.push(to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward]))

            # Put it on da screen
            if display:
                env.render()

            # Perform one step of the optimization (on the policy network)
            env.optimize_model()

        print("Epoch: {}  |  Predicted Migrants: {}".format(epoch, mp.item()))
        # print("\n")

if __name__ == "__main__":

    with open("./migration_data.json", "r") as f:
        mig_data = json.load(f)

    impath = "./test_ims/484019039.png"
    muni_id = "484019039"
    y_val = mig_data[muni_id]

    display = True
    to_tens = transforms.ToTensor()

    env = EarthObs(impath = impath, y_val = y_val, num_channels = 3, num_actions = 5, display = display)

    train(env, display)