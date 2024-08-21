# Imports
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from data import generate_points, SimpleDataset
from model import SimpleModel
from tensorlist import TensorList
from pso import PSO, step_wrapper
from train_model import train_model

if __name__ == '__main__':

    # Initialize simple data
    def f(x1, x2):
        return x1 + (x2 - 2)**2
    n_dims = 2
    n_points = 100
    point_range = [-5, 5]

    X, Y = generate_points(f, n_dims, n_points, point_range)
    X = X.type(torch.float32)

    # Initialize Datasets and DataLoaders
    train_dataset = SimpleDataset(X, Y)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize queues
    eval_queue = mp.Queue()
    optim_queue = mp.Queue()

    # Initialize models and optimizer
    num_particles = 20
    num_epochs = 30
    model_list = []
    with torch.no_grad():
        for i in range(num_particles):
            model = SimpleModel()
            for param in model.parameters():
                param.requires_grad_(False)
            model_list.append(model)
            eval_queue.put(
                (TensorList(model.parameters()), i)
            )
    shape_like = list(model.parameters())
    optimizer = PSO(shape_like, num_particles)

    # Prepare training arguments
    processes = []
    for i in range(num_particles):
        args=(model_list[i], eval_queue, optim_queue, train_dataloader, num_epochs)
        p = mp.Process(target=train_model, args=args)
        processes.append(p)

    # Define separate process for optimizer
    optim_process = mp.Process(target=step_wrapper, args=(optimizer, eval_queue, optim_queue, model_list[0]))

    # Start training processes
    for p in processes:
        p.start()
    print("Eval processes started")

    optim_process.start()
    print("Optimizer started")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"Optimizer returned with exit code {optim_process.exitcode}")
