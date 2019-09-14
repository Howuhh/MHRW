import logging
import numpy as np

import pyspark.sql.functions as F

from itertools import count
from pyspark.sql.types import IntegerType
from .utils import arr_choice, arr_len, rename_columns


def MHRW(graph, path, seed_ratio=0.001, budget_ratio=0.1, accept_func="MHRW", alpha=0.5):
    """
    Set docstring here.

    Parameters
    ----------
    graph: 
    path: 
    seed_ratio=0.001: 
    budget_ratio=0.1: 
    accept_func="MHRW": 
    alpha=0.5: 

    Returns
    -------

    """
    if seed_ratio >= 1 or seed_ratio <= 0:
        raise ValueError("seed ratio should be in range (0, 1)")

    if budget_ratio >= 1 or budget_ratio <= 0:
        raise ValueError("budget ratio should be in range (0, 1)")

    sample_seed = np.random.random()
    budget = int(graph.count() * budget_ratio)

    seed_nodes = graph.sample(False, seed_ratio, seed=sample_seed).persist()
    seed_nodes.select("user").write.parquet(path, mode="overwrite")

    cache = [seed_nodes]
    nodes_sampled_count = seed_nodes.count()
    print(f"INFO: sampled {nodes_sampled_count} seed nodes, budget: {budget}")

    for iter_ in count():
        if nodes_sampled_count < budget:
            new_nodes, nodes_count = _iter_MHRW(graph, cache[iter_], accept_func=accept_func, alpha=alpha)

            cache.append(new_nodes.persist())
            nodes_sampled_count += nodes_count
            cache[iter_].unpersist()

            new_nodes.select("user").write.parquet(path, mode="append")
            print(f"INFO: iteration {iter_}, sampled {nodes_sampled_count} nodes, progress: {round((nodes_sampled_count / budget)*100, 2)}%")
        else:
            print(f"INFO: the budget has been spent! stop sampling")
            break

    seed_nodes.unpersist()

    return None


def _iter_MHRW(user_neigh, seed_nodes, accept_func="MHRW", alpha=0.5):
    """
    Set docstring here.

    Parameters
    ----------
    user_neigh: 
    seed_nodes: 
    accept_func="MHRW": 
    alpha=0.5: 

    Returns
    -------

    """
    if accept_func == "MHRW":
        if_select = F.udf(lambda q_ratio: 1 if np.random.uniform(0, 1) <= min(1, q_ratio) else 0, IntegerType())
    elif accept_func == "RCMH":
        if_select = F.udf(lambda q_ratio: 1 if np.random.uniform(0, 1) <= q_ratio**alpha else 0, IntegerType())
    else:
        raise ValueError("wrong accept_func option, only: MHRW, RCMH (with alpha correction)")

    seed_nodes = seed_nodes.withColumn("cand", arr_choice("neigh"))
    cand_nodes = rename_columns(user_neigh, {"user": "cand_user", "neigh": "cand_neigh"})
    
    seed_nodes = (seed_nodes
                    .join(cand_nodes, seed_nodes["cand"] == cand_nodes["cand_user"])
                    .drop("cand_user")
                 )
    
    seed_nodes = seed_nodes.withColumn("q_ratio", arr_len(seed_nodes["neigh"]) / arr_len(seed_nodes["cand_neigh"]))
    seed_nodes = seed_nodes.withColumn("if_selected", if_select("q_ratio"))
    
    new_nodes_count = seed_nodes.filter(F.col("if_selected") == 1).count()
    
    new_nodes = (seed_nodes
                         .withColumn("user", 
                                     F.when(F.col("if_selected") == 1, F.col("cand")).otherwise(F.col("user")))
                          .withColumn("neigh", 
                                     F.when(F.col("if_selected") == 1, F.col("cand_neigh")).otherwise(F.col("neigh")))

                    ).select("user", "neigh")
    
    return new_nodes, new_nodes_count