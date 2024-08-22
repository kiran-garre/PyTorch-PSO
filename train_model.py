# Imports
import torch
import numpy as np
import random
from tqdm import tqdm
import gc
import torch.multiprocessing as mp
from queue import Empty
from tensorlist import multi_copy

def train_model(model, eval_queue, optim_queue, dataloader, num_epochs):
    counter = 0
    while counter < num_epochs:
        # Poll eval queue for weights to evaluate
        try:
            weights, index = eval_queue.get_nowait()
        except Empty:
            continue

        # Evaluate weights
        model.train()
        multi_copy(model.parameters(), weights)
        epoch_loss = 0
        for img, box in (dataloader):
            prediction = model(img)
            loss = torch.nn.functional.mse_loss(prediction[:, 0], box)
            epoch_loss += loss.item()

        # Send weights, index, and loss back to optimizer
        optim_queue.put((weights, index, epoch_loss / len(dataloader)))
        counter += 1

    # When epochs are completed
    optim_queue.put(("END", -1, -1))
