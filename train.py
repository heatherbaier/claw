import torch
import cv2 

from earth_env import *
from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


env = EarthObs(num_channels = 3, num_actions = 5)

for epoch in range(0, 10):

    # Reset the environment to the beginning
    obs = env.reset(epoch = epoch)

    # Capctur ethe current screen at the starting position
    current_screen = env.view_box.clip_image(cv2.imread("./test_image.png"))

    # Reset the ReplayMemory (LOOK INTO THIS)
    memory = ReplayMemory(10000)

    # Set done flag
    done = False

    # While there are still selects left...
    while not done:

        # Select and perform an action
        action = env.select_action()

        # Save current state so we can push it to memory in a couple lines
        current_state = env.view_box.clip_image(cv2.imread("./test_image.png"))

        # Calculate the state-action pair's reward and done flag
        _, reward, done, _ = env.step(action.item())

        # Get the new state post action
        next_state = env.view_box.clip_image(cv2.imread("./test_image.png"))

        # Push all of this goodness to memory
        memory.push(current_state, action, next_state, reward)

        # Put it on da screen
        env.render()

        # Perform one step of the optimization (on the policy network)
        env.optimize_model()

        print("\n")


    # env.close()