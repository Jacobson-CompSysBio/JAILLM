# A Guide to Distributed LLM Training and Inference

**Question:** Why do we even care? What's the benefit of distribution? 
**Answer:** Well, in the JAIL world, we deal with huge models and datasets. In the rare case that this data fits on a single node or processor, it takes a *long* time to process. Many times, you'll be hit with a dreaded *Out-of-Memory (OOM)* error instead, and you can't even run your code. Parallelism makes it possible to run scripts that are so huge they won't work. It's essential for us!

This document serves as a beginner's guide to distributed LLM training. This process involves partitioning a training workload across multiple processors, known as worker nodes. Worker nodes perform their own tasks at the same time (in parallel) to accelerate the training process.

We broadly define the two major subsets of distributed training: data parallelism and model parallelism.

## Data Parallelism

![data parallelism](images/dataparallelism.webp "Data Parallelism")

In data parallelism, training data is split across multiple machines, and a copy of the model is made on each machine and trained with its own portion of the data. Models are then synchronized (see [Synchronization](#synchronization)) to ensure that they have the same weights.

## Model Parallelism

![model parallelism](images/modelparallelism.webp "Model Parallelism")

Model parallelism splits the model itself across multiple machines, where these chunks are trained separately. 

## Synchronization
Since things are distributed, how do we end up synchronizing them again? 

### Gradient Averaging
This concept applies to data parallelism. Gradients are computed on each processor containing separate portions of the data, and these gradients are then averaged to produce the overall model gradients. The overall gradients are then broadcast to each processor. 

### Parameter Server
In a parameter-server scheme, nodes are divided between worker nodes (actually training the model) and parameter server nodes, which track the global model parameters. Worker nodes calculate gradients on their data and send them to the parameter server, which updates the global model.

Parameter-server setups have a few drawbacks. First, when parameters are shared synchronously, efficiency is lost, as each worker node needs to wait for the other workers to finish in order to move to the next iteration. Also, with a single parameter-server, adding more nodes does not increase speed, as each worker must communicate with the server, and that doesn't scale well.

### All-reduce
With all-reduce, each node both computes and stores gradients. Different algorithms dictate how gradients are passed between nodes. Unlike parameter-server schemes, adding more machines does not constrain the model, meaning that it scales better.

## [Fully-Sharded Data Parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
One downside of traditional data parallel is that a model must be copied to each device in order for training to occur. However, FSDP circumvents this by sharding the model as well as the data. Meta describes the process as such:

>*In FSDP, only a shard of the model is present on a GPU. Then, locally, all weights are gathered from the other GPUs — by means of an all-gather step — to calculate the forward pass. This gathering of weights is then performed again before the backward pass. After that backward pass, the local gradients are averaged and sharded across the GPUs by means of a reduce-scatter step, which allows each GPU to update its local weight shard*.

In psuedocode:
```
FSDP forward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        forward pass for layer_i
        discard full weights for layer_i

FSDP backward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        backward pass for layer_i
        discard full weights for layer_i
        reduce-scatter gradients for layer_i
```
