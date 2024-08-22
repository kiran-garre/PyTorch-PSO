# Imports
import torch
import numpy as np
from tensorlist import TensorList, multi_copy
import gc
from queue import Empty
import matplotlib.pyplot as plt
import math

class PSO:
    def __init__(self, shape_like, num_models, random_range=0.1):

        # Realtime parameters
        self.velocities = [TensorList.zeros_like(shape_like) for _ in range(num_models)]
        self.pbests = [TensorList.empty_like(shape_like) for _ in range(num_models)]
        self.gbest = None
        self.prev_loss = [float('inf') for _ in range(num_models)]

        self.counter = 0
        self.best_loss = float('inf')
        self.num_models = num_models
        self.rr = random_range
        self.vmax = 0.3

        # Constants for velocity update and constriction
        self.c1 = 2.05 # how much the particle's best influences its velocity
        self.c2 = 2.05 # how much the global best influences its velocity
        phi = (self.c1 + self.c2)
        self.chi = 2 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))

        self.loss_tracker = []

    def mini_step(self, tensors, idx, loss):

        # Print loss status every full epoch
        if self.counter % self.num_models == 0:
            print(f"Epoch {self.counter // self.num_models} Average: {np.mean(self.prev_loss)}; Best: {self.best_loss}")
            self.loss_tracker.append(self.best_loss)
        self.counter += 1

       # Update pbests, gbest
        if loss < self.prev_loss[idx]:
            self.pbests[idx].multi_copy_(tensors)
            self.prev_loss[idx] = loss
            if loss < self.best_loss:
                self.gbest = [tensors,] * self.num_models
                self.best_loss = loss

        # Update velocities
        r1 = np.random.rand()
        r2 = np.random.rand()
        self.velocities[idx] = self.chi*(self.velocities[idx] + self.c1*r1*(self.pbests[idx] - tensors) + self.c2*r2*(self.gbest[idx] - tensors))
        self.velocities[idx] = TensorList.clip(self.velocities[idx], -self.vmax, self.vmax)
        random_jiggle = TensorList.normal_like(tensors) * self.rr

        # Copy new weights into tensor
        temp = tensors + self.velocities[idx] + random_jiggle
        tensors.multi_copy_(temp)

        gc.collect()

    def save_best(self, model):
        multi_copy(model.parameters(), self.gbest[0])
        torch.save(model.state_dict(), 'best_model.pt')


# Wrapper function for PSO stepping in multiproccessing 
def step_wrapper(optimizer, eval_queue, optim_queue, save=None, graph_loss=True):
    processes_ended = 0
    while True:
        # Poll optim queue for weights to update
        try:
            tensors, index, loss = optim_queue.get_nowait()
        except Empty:
            continue
        if tensors == "END":
            processes_ended += 1
            # When all processes have completed
            if processes_ended == optimizer.num_models:
                if save:
                    optimizer.save_best(save) # save gbest weights to file
                if graph_loss:
                    plt.plot(range(len(optimizer.loss_tracker)), optimizer.loss_tracker)
                    plt.show()
                return
        else:
            # Update weights and send back to eval processes
            optimizer.mini_step(tensors, index, loss)
            eval_queue.put((tensors, index))
