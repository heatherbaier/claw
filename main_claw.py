from torchvision import models, transforms
import torch.multiprocessing as mp
from copy import deepcopy
import psutil
import socket
import torch
import json
import cv2 
import os

from shared_optim import *
from dataloader import *
from earth_env import *
from ViewBox import *
from models import *
from train import *
from utils import *
from eval import *

# font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


import time

if __name__ == "__main__":

    # Set up dataset as batches
    train_data = Dataset(batch_size = 12, imagery_dir = "/sciclone/geograd/Heather/mex_imagery/", json_path = "/sciclone/home20/hmbaier/claw/migration_data.json", valid = False, EPS_DECAY = 150)
    # val_data = Dataset(batch_size = 2, imagery_dir = "/sciclone/home20/hmbaier/claw/val_ims", json_path = "/sciclone/home20/hmbaier/claw/migration_data.json", valid = True, EPS_START = 0.05)

    print(len(train_data.data))
    # print(len(val_data.data))


    # Using the Manager allows us to share the ReplayMemory list between processes
    with mp.Manager() as manager:

        # Initialize variables
        device = "cpu"
        processes = []
        epochs = 20
        mini_epochs = 5
        batch_criterion = torch.nn.L1Loss()

        # Set up shared model and params
        shared_model = DQN(128, 128, n_actions).to(device)
        shared_model.share_memory()
        optimizer = SharedAdam(shared_model.parameters(), lr = 0.0001)
        optimizer.share_memory()

        # Set up the Replay Memory as a list that can be shared amongst all threads
        memory = manager.list()
        lock = mp.Lock()
        val_preds = manager.list()

        for epoch in range(0, epochs):

            epoch_ys = []
            epoch_preds = manager.list()

            """ TRAINING DATASET """

            start = time.perf_counter()

            # For every list of lists within the dataloader (i.e. for every batch...)
            for batch in train_data.data:

                # For each observation/image in the batch, start a training prcoess for it
                for count, (obs) in enumerate(batch):

                    epoch_ys.append((obs[0], obs[1]))

                    p = mp.Process(target = train, args=(obs[2], obs[0], epoch, mini_epochs, shared_model, optimizer, memory, lock, epoch_preds, count, False, ))
                    p.start()
                    processes.append(p)
            
                # Don't let another batch start until the first has finished
                for p in processes:
                    p.join()

                print("end of batch!")

            end = time.perf_counter()

            print("Time to finish with WITH mp & join: ", end - start)

            epoch_preds = list(epoch_preds)
            epoch_preds.sort(key = lambda i: i[0])
            epoch_ys.sort(key = lambda i: i[0])

            epoch_preds = [i[1] for i in epoch_preds]
            epoch_ys = [i[1] for i in epoch_ys]

            preds = torch.tensor(epoch_preds).view(-1, 1)
            trues = torch.tensor(epoch_ys).view(-1, 1)

            # print("preds: ", preds, "trues: ", trues)

            # print(preds, trues)

            training_loss = batch_criterion(preds, trues)

            with open("/sciclone/home20/hmbaier/claw/log_v5.txt", "a") as f:
                f.write("Epoch: " + str(epoch) + "  |  Training Loss: " + str(round(training_loss.item(), 4)) + "\n")



            print("Epoch: ", epoch, "  |  Training Loss: ", round(training_loss.item(), 4))


