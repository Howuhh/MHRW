import click
import logging as log

from config import CONF_PARAMS
from mhrw.utils import start_session
from mhrw import MHRWSampler

@click.command()
@click.argument('edges_path')
@click.argument('sample_save_path')
@click.option('--seed_ratio', default=0.001, help="The fraction of seed nodes in the graph for which random walk will be started.")
@click.option('--budget_ratio', default=0.1, help="The fraction of the nodes that need to be sampled. reaching it, the random walk will stop.")
@click.option('--method', default="MHRW", help="""The method by which the samples will be sampled. It determines with what probability the node will be accepted or rejected on each iteration. Two methods are available: 
                                                - Metropolis-Hastings Random Walk (MHRW)
                                                - Rejection-Controlled Metropolis-Hastings Algorithm (RCMH).""")
@click.option('--alpha', default=0.5, help="Only for RCMH. Alpha ∈ [0, 1] which parameterizes the acceptance function of node acceptance or rejection on each iteration. When alpha = 1 the algorithm is reduced to the original MHRW, when alpha = 0 to the simply random walk.")
def main(edges_path: str, sample_save_path: str, seed_ratio: float, 
         budget_ratio: float, method: str, alpha: float):
    """
    The MHRW algorithm is an application of the Metropolis-Hastings algorithm 
    for unbiased graph sampling. It makes use of a modified random walk to 
    draw nodes from the graph. In particular, it modifies the transition 
    probabilities of a random walk so that the walk converges into a uniform
    distribution.

    First, node seed node u is sampled. When MH draws a node v from N(u) with 
    probability 1/d_u, MH accepts v as a sample with probability min{1, d_u/d_v},
    and reject it with probability 1 − min{1, d_u/d_v}, where N(u) is set of u neighbors
    and d_i degree of i's node.

    Above is a description of how the algorithm works in relation to one 
    node. In this implementation, a certain percentage of seed nodes is 
    selected first, and then random walk begins for each one. Due to the 
    fact that all this is done for each node independently, the 
    parallelism and distributedness of the algorithm is achieved with 
    the help of PySpark. All unique nodes on every iteration are added to the 
    pool.
    """
    spark, sc = start_session("mhrwGraphSample", **CONF_PARAMS)
    edge_list = spark.read.parquet(edges_path)

    log.info(f"edges in full graph: {edge_list.count()}")

    sampler = MHRWSampler(spark, "tmp", seed_ratio, budget_ratio, method, alpha).fit(edge_list)
    sampled_graph = sampler.transform(edge_list)

    log.info(f"edges in sampled graph: {sampled_graph.count()}")
    sampled_graph.write.parquet(sample_save_path, mode="overwrite")
    sampler.clear_ids()


if __name__ == "__main__":
    main()