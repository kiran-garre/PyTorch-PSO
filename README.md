# PyTorch PSO

This is an implementation of particle swarm optimization (PSO) that can be used to train PyTorch models.

### A quick overview

Each "particle" represents a set of candidate model weights. Each particle has its own process (using `torch.multiprocessing`), and there is a separate process for the optimizer.
The only communication between the processes is two queues: an evaluation queue and an optimization queue. Once a particle process evaluates a set of weights, the weights and their corresponding loss are 
sent to the optimizer through the optimizer queue. Once the optimizer updates the weights, the new weights are put into the evaluation queue to be evaluated. This repeats for a specified number of epochs.

### The TensorList class:
A PyTorch model's weights can be easily accessed using the `torch.nn.Module.parameters()` function, which returns an iterator. While this iterator can be turned into a list, PyTorch does not provide any interface for 
operations between lists of tensors of different sizes (that I know of). Since these operations are crucial for PSO, I created the TensorList class which allows for clean operations between model weights during optimization. The implementation
does not necessarily provide any runtime benefits, but it makes operations much more readable and concise (rather than writing a for-loop every time).
