from .mhrw_base import MHRW
from .utils import graph_filter_ids


class MHRWSampler(object):
    def __init__(self, graph, path, method, spark):
        """
        Set docstring here.

        Parameters
        ----------
        self: 
        graph: 
        path: 
        method: 
        spark: 

        Attributes
        ----------

        Notes
        -----

        """   self.path = path
        self.graph = graph
        self.method = method
        self.spark = spark

        self._train = None
        self._ids = None

    @property
    def sampled_ids(self):
        if self._train is None:
            raise ValueError("Sample does not exists. First, fit the model")

        if self._ids is None:
            sample_ids = self.spark.read.parquet(self.path)
            sample_ids = sample_ids.select("user").distinct().withColumnRenamed("user", "user_id") 

            self._ids = sample_ids

            return sample_ids
        else:
            return self._ids

    def fit(self, seed_ratio, budget_ratio, alpha=0.5):
        MHRW(self.graph, self.path, 
             seed_ratio=seed_ratio, budget_ratio=budget_ratio, 
             accept_func=self.method, alpha=alpha)

        self._train = True

        return self

    def transform(self, edge_list):
        sample_graph = graph_filter_ids(edge_list, self.sampled_ids)

        return sample_graph
