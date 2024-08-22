# PyTorch PSO

This is an implementation of particle swarm optimization (PSO) that can be used to train PyTorch models.

### A quick overview

Each "particle" represents a set of candidate model weights. Each particle has its own process (using `torch.multiprocessing`), and there is a separate process for the optimizer.
The only communication between the processes is two queues: an evaluation queue and an optimization queue. Once a particle process evaluates a set of weights, the weights and their corresponding loss are 
sent to the optimizer through the optimizer queue. Once the optimizer updates the weights, the new weights are put into the evaluation queue to be evaluated. This repeats for a specified number of epochs.

### The TensorList class
A PyTorch model's weights can be easily accessed using the `torch.nn.Module.parameters()` function, which returns an iterator. While this iterator can be turned into a list, PyTorch does not provide any interface for 
operations between lists of tensors of different sizes (that I know of). Since these operations are crucial for PSO, I created the TensorList class to allow for clean operations between model weights during optimization. The implementation does not necessarily provide any runtime benefits, but it makes operations much more readable and concise (rather than writing a for-loop every time).

### My choice of model
The model, target function, and loss function I used are all glaringly simple. The code I wrote is more of a proof of concept; it runs, and it works. PSO has been studied analytically and empirically, and my 
implementation would probably work with better tuning and fewer hardware constraints. However, for now, I've shown that it can be built in PyTorch.

### Lack of CUDA support
Originally, I tried to allow CUDA operations to speed up model evaluations. However, CUDA with multiprocessing is a bad combination, and it didn't work on any of the notebooks I tried. I tried it to run it on a local GPU, but 
each tensor would get zeroed out every time it moved between processes. This didn't work for me, and since my model was very simple anyways, I decided to only use CPU to train.

### Why PSO?
The main selling point of PSO is that it does not rely on any gradient calculations throughout the entire process. This means that non-differentiable loss functions can be used; the only requirement is that the function is continuous. This is even more useful when you consider that PyTorch gradient calculations only work with `torch.Tensor` objects. If your loss function uses NumPy arrays or does not keep track of the gradients, you can't use popular PyTorch built-ins like Adam or SGD. But you _can_ use PSO.

### Future developments
There are two main directions I want to take this project in the future:
1. There are many variants of PSO that claim faster convergence, so I'll try to implement those by modifying my current code.
2. Currently, each particle evaluates each set of weights on the entire dataset. When you're training 10, 20, or 100 particles, this can get pretty intense for any hardware, especially as the size of the model grows. However, something I want to explore is evaluating each model on only one mini-batch at a time, similarly to how SGD updates weights after each mini-batch instead of the whole dataset. If this method can approximate the loss function
accurately enough for PSO, then it could lead to huge performance boosts.
