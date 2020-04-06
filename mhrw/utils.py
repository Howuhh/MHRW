import numpy as np

import pyspark.sql.functions as F

from pyspark import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, LongType

from typing import Dict, Tuple


arr_len = F.udf(lambda arr: len(arr), IntegerType())
arr_choice = F.udf(lambda arr: int(np.random.choice(arr)), LongType())


def start_session(task_name: str, **kwargs) -> Tuple[SparkSession, SparkContext]:
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf()
    for param in kwargs:
        assert isinstance(param, str), "conf param should be string"
        try:
            conf.set(param, kwargs[param])
        except KeyError:
            raise KeyError(f"Unknown configuration parameter: {param}")

    spark = (SparkSession
                .builder
                .config(conf=conf)
                .appName(task_name)
                .getOrCreate())
    sc = spark.sparkContext

    return spark, sc


def graph_filter_ids(edge_list: DataFrame, filter_ids: DataFrame) -> DataFrame:
    edge_list_filt = (edge_list
            .join(filter_ids, edge_list["user"] == filter_ids["id"])
            .drop("id")
            .join(filter_ids, edge_list["item"] == filter_ids["id"])
            .select("user", "item"))

    return edge_list_filt


def rename_columns(df: DataFrame, columns: Dict[str, str]) -> DataFrame:
    if isinstance(columns, dict):
        for old_name, new_name in columns.items():
            df = df.withColumnRenamed(old_name, new_name)
        return df
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")


if __name__ == "__main__":
    spark, sc = start_session("test_session")
    print(spark)

    print(sc.getConf().getAll())
