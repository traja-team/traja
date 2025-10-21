import logging

from traja import dataset, models

from .accessor import TrajaAccessor
from .frame import TrajaCollection, TrajaDataFrame
from .parsers import from_df, read_file
from .plotting import *
from .trajectory import *

__author__ = "justinshenk"
__version__ = "25.0.0"

logging.basicConfig(level=logging.INFO)


def set_traja_axes(axes: list):
    TrajaAccessor._set_axes(axes)
