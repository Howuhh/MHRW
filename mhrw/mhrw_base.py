import shutil

import logging as log 
import numpy as np

import pyspark.sql.functions as F

from itertools import count
from typing import Tuple
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from .utils import arr_choice, arr_len, rename_columns

log.basicConfig(level=log.INFO, 
                    format='%(levelname)s - %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# TODO: добавить загрузку айдишек + уникальные в return, так будет проще и для модели и для использования, т.к. 
# для других не очевидно, что в файлик сохраняется адишки с повторениями.
def MHRW(graph: DataFrame, ids_path: str, seed_ratio: float=0.001, 
        budget_ratio: float=0.1, accept_func: str="MHRW", alpha: float=0.5) -> None:
    """
    Set docstring here.

    Parameters
    ----------
    graph: 
    tmp_path: 
    seed_ratio=0.001: 
    budget_ratio=0.1: 
    accept_func="MHRW": 
    alpha=0.5: 

    Returns
    -------

    """
    if not 0 <= seed_ratio <= 1:
        raise ValueError("seed ratio should be in range (0, 1)")

    if not 0 <= budget_ratio <= 1:
        raise ValueError("budget ratio should be in range (0, 1)")

    sample_seed = np.random.random()
    budget = int(graph.count() * budget_ratio)

    seed_nodes = graph.sample(False, seed_ratio, seed=sample_seed).persist()
    seed_nodes.select("user").write.parquet(ids_path, mode="overwrite")

    nodes_sampled_count = seed_nodes.count()
    log.info(f"sampled {nodes_sampled_count} seed nodes, budget: {budget}")

    progress_total = 0.0
    cache = [seed_nodes]

    for iter_ in count():
        if nodes_sampled_count < budget:
            new_nodes, nodes_count = _iter_MHRW(graph, cache[iter_],
                                                accept_func=accept_func, alpha=alpha)

            cache.append(new_nodes.persist())
            cache[iter_].unpersist()

            # use accumulator?
            nodes_sampled_count += nodes_count

            new_nodes.select("user").write.parquet(ids_path, mode="append")

            progress_iter = round((nodes_sampled_count / budget)*100, 2)

            if abs(progress_total - progress_iter) < 0.5:
                log.warn(f"gain lower than 0.5%, stop sampling!")
                break
            else:
                log.info(f"iteration {iter_ + 1}, sampled {nodes_sampled_count} nodes, progress: {progress_iter}%")
                progress_total = progress_iter
        else:
            log.info(f"the budget has been spent! stop sampling")
            break

    seed_nodes.unpersist()

    return None


def _iter_MHRW(user_neigh: DataFrame, seed_nodes: DataFrame, accept_func: str="MHRW", alpha: float=0.5) -> Tuple[DataFrame, int]:
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
