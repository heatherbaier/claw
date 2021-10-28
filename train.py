from torchvision import models, transforms
import psutil
import socket
import torch
import json
import cv2 
import os

from earth_env import *
from ViewBox import *
from models import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


def train(env, impath, epoch, mini_epochs, shared_model, optimizer, memory, lock, epoch_preds, temp, display = True):
    
    to_tens = transforms.ToTensor()

    env.reset(epoch = epoch)
    done = False
    muni_id = str(impath.split("/")[5].replace(".png", ""))

    with open("/sciclone/home20/hmbaier/claw/runs/log_" + muni_id + "v5.txt", "a") as f:
        f.write("In training loop on epoch " + str(epoch) + " for temp variable " + str(temp) + "\n")


    for mini_epoch in range(mini_epochs):   

        with open("/sciclone/home20/hmbaier/claw/runs/log_" + muni_id + "v5.txt", "a") as f:
            f.write("Temp: " + str(temp) + "  |  Mini Epoch: " + str(mini_epoch) + "\n")
 
        env.reset(epoch = mini_epoch)
        done = False

        # print("mini_epoch: ", mini_epoch)

        while not done:

            action = env.select_action(shared_model)

            with open("/sciclone/home20/hmbaier/claw/runs/log_" + muni_id + "v5.txt", "a") as f:
                f.write("Temp: " + str(temp) + "  |  Mini Epoch: " + str(mini_epoch) + "  |  Action: " + str(action.item()) + "\n")

            current_state = env.view_box.clip_image(env.image)
            mp, reward, done, _ = env.step(action.item(), shared_model, optimizer, epoch_preds, lock)
            next_state = env.view_box.clip_image(env.image)
            memory.append((to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward])))
            
            # if (done) and (mini_epoch == mini_epochs - 1):

            # with lock:
            env.optimize_model(shared_model, memory, optimizer)

            if display:
                env.render()

    with lock:
        epoch_preds.append((impath, mp.detach()))


    with open("/sciclone/home20/hmbaier/claw/runs/log_" + muni_id + "v5.txt", "a") as f:
        f.write("MP: " + str(mp.detach()) + "  |  Epoch Preds: " + str(epoch_preds) + "\n")


    print("TEMP ID PARALLEL: ", temp)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


    # with open("/sciclone/home20/hmbaier/claw/log_inepoch_" + socket.gethostname() + "_" + str(psutil.Process().cpu_num()) + ".txt", "a") as f:
    #     f.write("Pred: " + str(mp))

