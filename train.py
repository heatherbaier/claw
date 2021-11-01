from mpi4py import MPI

from torchvision import models, transforms
import platform
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


def train(env, 
          impath, 
          epoch, 
          mini_epochs, 
          shared_model, 
          optimizer, 
          memory, 
          lock, 
          epoch_preds, 
          temp, 
          display = False, 
          valid = False):

    comm = MPI.COMM_WORLD

    env.load_image()

    to_tens = transforms.ToTensor()

    env.reset(epoch = epoch, valid = valid)
    done = False
    muni_id = str(impath.split("/")[5].replace(".png", ""))
    

    with open("/sciclone/home20/hmbaier/test/runs4/log_" + str(comm.Get_rank()) + "_" + str(MPI.Get_processor_name()) + "_" + muni_id + "_" + str(platform.node()) + "_" + str(socket.gethostname()) + ".txt", "a") as f:
        f.write(str(os.uname()) + "\n")    

    with open("/sciclone/home20/hmbaier/test/runs4/log_" + str(comm.Get_rank()) + "_" + str(MPI.Get_processor_name()) + "_" + muni_id + "_" + str(platform.node()) + "_" + str(socket.gethostname()) + ".txt", "a") as f:
        f.write("In training loop on epoch " + str(epoch) + " for temp variable " + str(temp) + "\n")


    for mini_epoch in range(mini_epochs):   

        with open("/sciclone/home20/hmbaier/test/runs4/log_" + str(comm.Get_rank()) + "_" + str(MPI.Get_processor_name()) + "_" + muni_id + "_" + str(platform.node()) + "_" + str(socket.gethostname()) + ".txt", "a") as f:
            f.write("Temp: " + str(temp) + "  |  Mini Epoch: " + str(mini_epoch) + "\n")
 
        env.reset(epoch = mini_epoch, valid = valid)
        done = False

        while not done:

            action = env.select_action(shared_model)

            with open("/sciclone/home20/hmbaier/test/runs4/log_" + str(comm.Get_rank()) + "_" + str(MPI.Get_processor_name()) + "_" + muni_id + "_" + str(platform.node()) + "_" + str(socket.gethostname()) + ".txt", "a") as f:
                f.write("Temp: " + str(temp) + "  |  Mini Epoch: " + str(mini_epoch) + "  |  Action: " + str(action.item()) + "\n")

            current_state = env.view_box.clip_image(env.image)
            mp, reward, done, _ = env.step(action.item(), shared_model, optimizer, epoch_preds, lock, valid = valid)
            next_state = env.view_box.clip_image(env.image)
            memory.append((to_tens(current_state).unsqueeze(0), action, to_tens(next_state).unsqueeze(0), torch.tensor([reward])))
            
            # if (done) and (mini_epoch == mini_epochs - 1):

            # with lock:
            env.optimize_model(shared_model, memory, optimizer)

            # if display:
            #     env.render()

    with lock:
        epoch_preds.append((impath, mp.detach()))


    with open("/sciclone/home20/hmbaier/test/runs4/log_" + str(comm.Get_rank()) + "_" + str(MPI.Get_processor_name()) + "_" + muni_id + "_" + str(platform.node()) + "_" + str(socket.gethostname()) + ".txt", "a") as f:
        f.write("MP: " + str(mp.detach()) + "  |  Epoch Preds: " + str(epoch_preds) + "\n")


