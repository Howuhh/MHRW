import click
import logging as log

from config import CONF_PARAMS
from mhrw.utils import start_session
from mhrw import MHRWSampler


def main():
    spark, sc = start_session("mhrwGraphSample", **CONF_PARAMS)
    pass


if __name__ == "__main__":
    main()