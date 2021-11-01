
from mpi4py import MPI

from torchvision import models, transforms
import torch.multiprocessing as mp
from copy import deepcopy
import numpy as np
import torch
import math
import time
import sys

from config import get_config
from shared_optim import *
from dataloader import *
from earth_env import *
from ViewBox import *
from models import *
from train import *
from utils import *
from eval import *


def mpiabort_excepthook(type, value, traceback):
    
    print("Failed with information: ")
    print(type, value, traceback)
    comm.Abort()
    sys.__excepthook__(type, value, traceback)


def main():


    if rank == 0:

        # data = [(i+1)**2 for i in range(size)]
        # print("Data in rank 0: ", data)

        # Make the batch_size so that the number of batches is equivalent to the number of CPU's available
        # batch_size = math.ceil(len(os.listdir(config.imagery_dir)) / size)
        batch_size = math.floor(80 / size)
        print("SIZE: ", size, "  |  BATCH SIZE: ", batch_size)

        # Load in the data
        data = Dataset(batch_size = batch_size,
                       num_workers = size,
                       imagery_dir = config.imagery_dir,
                       json_path = config.json_path,
                       split = config.tv_split,
                       eps_decay = config.eps_decay)

        train_dl = data.train_data
        val_dl = data.val_data

        print("Rank 0 Training Data Length: ", len(train_dl))
        print("Rank 0 Validation Data Length: ", len(val_dl))

    else:

        train_dl = None

    # Scatter each batch to a unique worker
    train_dl = comm.scatter(train_dl, root = 0)

    # print("Data in rank: ", rank, [type(i[2]) for i in train_dl])

    # Once the data has been equally scattered to all workers (including rank 0), have each worker call train
    # Using the Manager allows us to share the ReplayMemory list between processes
    with mp.Manager() as manager:

        # Initialize variables
        device = "cpu"
        processes = []
        epochs = config.n_epochs
        mini_epochs = config.n_mini_epochs
        batch_criterion = torch.nn.L1Loss()

        # Set up shared model and params
        shared_model = DQN(128, 128, config.n_actions).to(device)
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
            # for batch in train_dl:

            # For each observation/image in the batch, start a training prcoess for it
            for count, (obs) in enumerate(train_dl):

                epoch_ys.append((obs[0], obs[1]))
                p = mp.Process(target = train, args=(obs[2], obs[0], epoch, mini_epochs, shared_model, optimizer, memory, lock, epoch_preds, count, config.display, False, ))
                p.start()
                processes.append(p)
        
            # Don't let another batch start until the first has finished
            for p in processes:
                p.join()

        end = time.perf_counter()





if __name__ == "__main__":

    # All process get these objects
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    config, _ = get_config()

    sys.excepthook = mpiabort_excepthook
    main()
    sys.excepthook = sys.__excepthook__