from torchvision import models, transforms
import torch
import json
import cv2 

from earth_env import *
from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


def train(env, impath, num_epochs, memory, lock, display = True):

    to_tens = transforms.ToTensor()

    for epoch in range(0, num_epochs):

        env.reset(epoch = epoch)
        done = False

        while not done:
            action = env.select_action()
            current_state = env.view_box.clip_image(cv2.imread(impath))
            mp, reward, done, _ = env.step(action.item())
            next_state = env.view_box.clip_image(cv2.imread(impath))
            memory.append((to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward])))
            # memory.push(to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward]))
            
            with lock:
                env.optimize_model(memory)

            if display:
                env.render()

        print("Epoch: {}  |  Predicted Migrants: {}".format(epoch, mp.item()))


if __name__ == "__main__":

    with open("./migration_data.json", "r") as f:
        mig_data = json.load(f)

    impath = "./test_ims/484019039.png"
    muni_id = "484019039"
    y_val = mig_data[muni_id]
    memory = ReplayMemory(10000)

    display = True
    to_tens = transforms.ToTensor()

    env = EarthObs(impath = impath, y_val = y_val, num_channels = 3, num_actions = 5, display = display)

    train(env, 10, display)
