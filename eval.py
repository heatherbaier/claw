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

def validate(env, impath, shared_model, lock, epoch_preds, display = True):

    optimizer = None

    to_tens = transforms.ToTensor()

    env.reset(epoch = 0)
    done = False

    while not done:
        action = env.select_action(shared_model)
        current_state = env.view_box.clip_image(cv2.imread(impath))
        mp, reward, done, _ = env.step(action.item(), shared_model, optimizer, epoch_preds, lock)
        next_state = env.view_box.clip_image(cv2.imread(impath))

        if display:
            env.render()

    with lock:
        epoch_preds.append((impath, mp.item()))

if __name__ == "__main__":

    # Set up dataset as batches
    data = Dataset(batch_size = 2, 
                   imagery_dir = "test_ims2", 
                   json_path = "migration_data.json", 
                   valid = True,
                   EPS_START = 0.05)

    shared_model = DQN(128, 128, n_actions).to(device)

    checkpoint = torch.load("./models/model_v2.torch")
    shared_model.load_state_dict(checkpoint['model_state_dict'])

    batch_criterion = torch.nn.L1Loss()

    processes = []
    lock = mp.Lock()

    with mp.Manager() as manager:

        epoch_ys = []
        epoch_preds = manager.list()

        for batch in data.data:

            # For each observation/image in the batch, start a training prcoess for it
            for obs in batch:

                print(obs[2].EPS_START)

                epoch_ys.append((obs[0], obs[1]))

                # args = (env, impath, num_epochs, shared ReplayMemory, lock for Replay Memory update, display)
                p = mp.Process(target = validate, args=(obs[2], obs[0], shared_model, lock, epoch_preds, True, ))
                p.start()
                processes.append(p)

            # Don't let another batch start until the first has finished
            for p in processes:
                p.join()


        epoch_preds = list(epoch_preds)
        epoch_preds.sort(key = lambda i: i[0])
        epoch_ys.sort(key = lambda i: i[0])

        epoch_preds = [i[1] for i in epoch_preds]
        epoch_ys = [i[1] for i in epoch_ys]

        preds = torch.tensor(epoch_preds).view(-1, 1)
        trues = torch.tensor(epoch_ys).view(-1, 1)

        print(preds, trues)

        batch_loss = batch_criterion(preds, trues)
        # print("Epoch: ", epoch, "  |  Loss: ", batch_loss.item())

