import shutil

import pyspark.sql.functions as F

from .mhrw_base import MHRW
from .utils import graph_filter_ids

from pyspark.sql import SparkSession, DataFrame


class MHRWSampler:
    def __init__(self, spark: SparkSession, ids_path: str, seed_ratio: float=0.001,
                 budget_ratio: float=0.1, method: str="MHRW", alpha: float=0.5):
        self.spark = spark
        self.ids_path = ids_path

        self.seed_ratio = seed_ratio
        self.budget_ratio = budget_ratio

        self.method = method
        self.alpha = alpha

        self._ids = None

    @property
    def sampled_ids(self) -> DataFrame:
        # return lazy dataframe
        if self._ids is None:
            try:
                sample_ids = self.spark.read.parquet(self.ids_path)
            except Exception:
                raise ValueError("Sample does not exists. First, fit the model")
            
            sample_ids = sample_ids.distinct().withColumnRenamed("user", "id")
            self._ids = sample_ids

            return self._ids
        else:
            return self._ids

    def fit(self, edge_list: DataFrame, cache: bool=True) -> 'MHRWSampler':
        self._ids = None
        graph = (edge_list
                    .toDF("user", "item")
                    .groupBy("user")
                    .agg(F.collect_list("item").alias("neigh")))
        
        if cache: 
            graph.persist()

        MHRW(graph, self.ids_path, seed_ratio=self.seed_ratio, 
            budget_ratio=self.budget_ratio, accept_func=self.method, alpha=self.alpha)

        if cache: 
            graph.unpersist()

        return self

    def transform(self, edge_list: DataFrame) -> DataFrame:
        sampled_graph = graph_filter_ids(edge_list, self.sampled_ids)

        return sampled_graph

    def clear_ids(self) -> str:
        shutil.rmtree(self.ids_path)

        return self.ids_path
