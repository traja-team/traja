from .accessor import TrajaAccessor
from .frame import TrajaDataFrame, TrajaCollection
from .parsers import read_file, from_df
from .plotting import *
from .trajectory import *
from traja import models
from traja import dataset

import logging

__author__ = "justinshenk"
__version__ = "0.2.3"

logging.basicConfig(level=logging.INFO)


def set_traja_axes(axes: list):
    TrajaAccessor._set_axes(axes)
