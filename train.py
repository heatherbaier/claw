from torchvision import models, transforms
import torch
import json
import cv2 

from earth_env import *
from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


def train(env, impath, epoch, mini_epochs, shared_model, optimizer, memory, lock, epoch_preds, display = True):

    to_tens = transforms.ToTensor()

    env.reset(epoch = epoch)
    done = False

    for mini_epoch in range(mini_epochs):    

        env.reset(epoch = mini_epoch)
        done = False

        # print("mini_epoch: ", mini_epoch)

        while not done:
            action = env.select_action(shared_model)
            current_state = env.view_box.clip_image(cv2.imread(impath))
            mp, reward, done, _ = env.step(action.item(), shared_model, optimizer, epoch_preds, lock)
            next_state = env.view_box.clip_image(cv2.imread(impath))
            memory.append((to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward])))
            
            # if (done) and (mini_epoch == mini_epochs - 1):


            with lock:
                env.optimize_model(shared_model, memory, optimizer)

            if display:
                env.render()

    with lock:
        epoch_preds.append(mp.item())


    # print("Epoch: {}  |  Predicted Migrants: {}".format(epoch, mp.item()))


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
