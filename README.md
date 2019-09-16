# MHRW
metropolis hastings random walk implementation with pyspark

![pseudocode](https://imgur.com/lCK1sW5)

# CLI

```console
Usage: mhrw_sample.py [OPTIONS] EDGES_PATH SAMPLE_SAVE_PATH

  The MHRW algorithm is an application of the Metropolis-Hastings algorithm
  for unbiased graph sampling. It makes use of a modified random walk to
  draw nodes from the graph. In particular, it modifies the transition
  probabilities of a random walk so that the walk converges into a uniform
  distribution.

  First, node seed node u is sampled. When MH draws a node v from N(u) with
  probability 1/d_u, MH accepts v as a sample with probability min{1,
  d_u/d_v}, and reject it with probability 1 − min{1, d_u/d_v}, where N(u)
  is set of u neighbors and d_i degree of i's node.

Options:
  --seed_ratio FLOAT    The fraction of seed nodes in the graph for which
                        random walk will be started.
  --budget_ratio FLOAT  The fraction of the nodes that need to be sampled.
                        reaching it, the random walk will stop.
  --method TEXT         The method by which the samples will be sampled. It
                        determines with what probability the node will be
                        accepted or rejected on each iteration. Two methods
                        are available:
                        - Metropolis-Hastings Random Walk
                        (MHRW)
                        - Rejection-Controlled Metropolis-Hastings
                        Algorithm (RCMH).
  --alpha FLOAT         Only for RCMH. Alpha ∈ [0, 1] which parameterizes the
                        acceptance function of node acceptance or rejection on
                        each iteration. When alpha = 1 the algorithm is
                        reduced to the original MHRW, when alpha = 0 to the
                        simply random walk
  --help                Show this message and exit.
```

# References

- Wang, T., Chen, Y., Zhang, Z., Xu, T., Jin, L., Hui, P., … Li, X. (2011). Understanding Graph Sampling Algorithms for Social Network Analysis. 
    2011 31st International Conference on Distributed Computing Systems Workshops, 123–128. 
    https://doi.org/10.1109/ICDCSW.2011.34https://doi.org/10.1109/ICDE.2015.7113345

- Li, R., Yu, J. X., Qin, L., Mao, R., & Jin, T. (2015). On random walk based graph sampling. 
    2015 IEEE 31st International Conference on Data Engineering, 927–938. 
    https://doi.org/10.1109/ICDE.2015.7113345
