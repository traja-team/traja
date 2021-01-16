import logging

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from torch.utils.data.sampler import WeightedRandomSampler

logger = logging.getLogger(__name__)


def get_class_distribution(targets):
    """Compute class distribution, returns number of classes and their count in the targets

    Args:
        targets ([type]): [description]

    Returns:
        [type]: [description]
    """
    targets_ = np.unique(targets, return_counts=True)
    return targets_[0], targets_[1]


def generate_dataset(df, n_past: int, n_future: int, stride: int = None, parameter_columns: list = list()):
    """
    df : Dataframe
    n_past: Number of past observations
    n_future: Number of future observations
    stride: Size of the sliding window. Defaults to sequence_length
    Returns:
    X: Past steps
    Y: Future steps (Sequence target)
    Z: Sequence category"""

    # Split the dataframe with respect to IDs
    series_ids = dict(
        tuple(df.groupby("ID"))
    )  # Dict of ids as keys and x,y,id as values

    train_data, target_data, target_category, target_parameters = list(), list(), list(), list()

    if stride is None:
        stride = n_past + n_future

    assert n_past >= 1, 'n_past has to be positive!'
    assert n_future >= 1, 'n_past has to be positive!'
    assert stride >= 1, 'Stride has to be positive!'

    sequences_in_categories = list()

    for ID in series_ids.keys():
        xx, yy, zz, ww = list(), list(), list(), list()
        # Drop the column ids and convert the pandas into arrays
        non_parameter_columns = [column for column in df.columns if column not in parameter_columns]
        series = series_ids[ID].drop(columns=['ID'] + parameter_columns).to_numpy()
        parameters = series_ids[ID].drop(columns=non_parameter_columns).to_numpy()[0, :]
        window_start = 0
        sequences_in_category = 0
        while window_start <= len(series):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if not future_end >= len(series):
                # slicing the past and future parts of the window
                past, future = series[window_start:past_end, :], series[past_end:future_end, :]
                # past, future = series[window_start:future_end, :], series[past_end:future_end, :]
                xx.append(past)
                yy.append(future)
                # For each sequence length set target category
                zz.append(int(ID), )
                ww.append(parameters)
                sequences_in_category += 1
            window_start += stride

        train_data.extend(np.array(xx))
        target_data.extend(np.array(yy))
        target_category.extend(np.array(zz))
        target_parameters.extend(np.array(ww))
        sequences_in_categories.append(sequences_in_category)
    return train_data, target_data, target_category, target_parameters, sequences_in_categories


def get_indices_from_categories(categories: list, sequences_in_categories: list):
    indices = list()
    sequence_index = 0
    start_index = 0
    for category in categories:
        while sequence_index < len(sequences_in_categories) and sequence_index < category:
            start_index += sequences_in_categories[sequence_index]
            sequence_index += 1
        if sequence_index >= len(sequences_in_categories):
            break
        if sequence_index == category:
            indices += list(range(start_index, start_index + sequences_in_categories[sequence_index]))
        start_index += sequences_in_categories[sequence_index]
        sequence_index += 1
    return indices
