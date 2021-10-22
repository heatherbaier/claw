from torchvision import models, transforms
import torch.multiprocessing as mp
import torch
import json
import cv2 

from shared_optim import *
from dataloader import *
from earth_env import *
from ViewBox import *
from models import *
from train import *
from utils import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    # Set up dataset as batches
    data = Dataset(batch_size = 2, imagery_dir = "test_ims2", json_path = "migration_data.json")

    # Using the Manager allows us to share the ReplayMemory list between threads
    with mp.Manager() as manager:

        # Initialize variables
        device = "cpu"
        processes = []

        # Set up shared model and params
        shared_model = DQN(128, 128, n_actions).to(device)
        shared_model.share_memory()
        optimizer = SharedAdam(shared_model.parameters(), lr = 0.01)
        optimizer.share_memory()

        # Set up the Replay Memory as a list that can be shared amongst all threads
        memory = manager.list()
        lock = mp.Lock()

        for epoch in range(0, 5):

            epoch_ys = []
            epoch_preds = manager.list()

            # For every list of lists within the dataloader (i.e. for every batch...)
            for batch in data.data:

                # For each observation/image in the batch, start a training prcoess for it
                for obs in batch:

                    epoch_ys.append(obs[1])

                    # args = (env, impath, num_epochs, shared ReplayMemory, lock for Replay Memory update, display)
                    p = mp.Process(target = train, args=(obs[2], obs[0], epoch, shared_model, optimizer, memory, lock, epoch_preds, True, ))
                    p.start()
                    processes.append(p)
            
                # Don't let another batch start until the first has finished
                for p in processes:
                    p.join()

            print(list(epoch_preds))
            print(type(list(epoch_preds)[0]))

            print("Epoch: ", epoch, "  |  Total Y: ", np.sum(epoch_ys), "  |  Total Predicted Migrants: ", int(np.sum(np.array(list(epoch_preds)))), "  |  Difference: ", np.sum(epoch_ys) - int(np.sum(np.array(list(epoch_preds)))))
            



