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


def run_batch(env):
    train(env = env[2], impath = env[0], num_epochs = 5, display = True)


if __name__ == "__main__":

    data = Dataset(batch_size = 2, imagery_dir = "test_ims", json_path = "migration_data.json")

    processes = []

    # For every list of lists within the dataloader (i.e. for every batch...)
    for batch in data.data:

        # For each observation/image in the batch, start a training prcoess for it
        for obs in batch:

            p = mp.Process(target = run_batch, args=(obs, ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print("yo")