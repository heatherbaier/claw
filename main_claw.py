from torchvision import models, transforms
import torch.multiprocessing as mp
import torch
import json
import cv2 

from dataloader import *
from earth_env import *
from ViewBox import *
from models import *
from train import *
from utils import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


def run_batch(env, memory):
    train(env = env[2], impath = env[0], num_epochs = 5, memory = memory, display = True)


if __name__ == "__main__":

    data = Dataset(batch_size = 2, imagery_dir = "test_ims", json_path = "migration_data.json")

    with mp.Manager() as manager:

        processes = []
        # memory = ReplayMemory(10000)
        memory = manager.list()
        lock = mp.Lock()

        # For every list of lists within the dataloader (i.e. for every batch...)
        for batch in data.data:

            # For each observation/image in the batch, start a training prcoess for it
            for obs in batch:

                p = mp.Process(target = train, args=(obs[2], obs[0], 5, memory, lock, True, ))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        print("yo")