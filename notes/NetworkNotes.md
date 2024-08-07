# Network Embedding Notes
Below, we cover a handful of methods for embedding networks into vector space. Each section contains an explanation of the method and suggested framework for implementation.

## 1. [Node2Vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

Node2Vec is a framework for continuous, learned, and unsupervised embedding of nodes into some latent space. The method uses an objective function designed to preserve neighborhoods of nodes in a *d*-dimensional feature space.
    1. First, a biased random walk is used to traverse the graph. Multiple parameters are used to control the walk:
        * **Return Parameter (p):** The return parameter dictates how likely the walker is to return to a previous node.
        * **In-out Parameter (q):** The in-out parameter dictates how likely the walker is to stay near the previous node vs traveling farther away.
        * **Walk Length (l):**
    2. Repeat the random walk *r* times
    3. After repetition, optimize the objective Skip-Gram function with SGD
        * Maximize the log-probability of observing a network neighorhood (*N_S(u)*) for node *U* conditioned on its feature representation *f*.

## 2. [Graph Autoencoder (GAE)](https://arxiv.org/abs/1611.07308)

## 3. [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907)

## 4. [GraphLLM Embedding](https://arxiv.org/abs/2310.05845)
`