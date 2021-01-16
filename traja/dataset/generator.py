import logging
from collections import defaultdict

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
    Z: Sequence ID"""

    # Split the dataframe with respect to IDs
    sequence_ids = dict(
        tuple(df.groupby("ID"))
    )  # Dict of ids as keys and x,y,id as values

    train_data, target_data, target_category, target_parameters = list(), list(), list(), list()

    if stride is None:
        stride = n_past + n_future

    assert n_past >= 1, 'n_past has to be positive!'
    assert n_future >= 1, 'n_past has to be positive!'
    assert stride >= 1, 'Stride has to be positive!'

    samples_in_sequence_id = list()

    for ID in sequence_ids.keys():
        xx, yy, zz, ww = list(), list(), list(), list()
        # Drop the column ids and convert the pandas into arrays
        non_parameter_columns = [column for column in df.columns if column not in parameter_columns]
        series = sequence_ids[ID].drop(columns=['ID'] + parameter_columns).to_numpy()
        parameters = sequence_ids[ID].drop(columns=non_parameter_columns).to_numpy()[0, :]
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
        samples_in_sequence_id.append(sequences_in_category)
    return train_data, target_data, target_category, target_parameters, samples_in_sequence_id


def get_indices_from_sequence_ids(sequence_ids: list, samples_in_sequence_id: list):
    indices = list()

    # We compute weights since it is cheap and they are used when weighing samples.
    weights = defaultdict(float)
    sequence_index = 0
    start_index = 0

    for sequence_id in sequence_ids:
        # We need to compute the start of each sequence's samples. To do this, we
        # compute the start of all sequences' sample starts. start_index
        # keeps track of where each sequence's samples start.
        while sequence_index < len(samples_in_sequence_id) and sequence_index < sequence_id:
            start_index += samples_in_sequence_id[sequence_index]
            sequence_index += 1
        if sequence_index >= len(samples_in_sequence_id):
            break
        if sequence_index == sequence_id:
            # The weight is simply one over the number of samples in this sequence.
            if samples_in_sequence_id[sequence_index]:
                weights[sequence_id] = 1.0 / samples_in_sequence_id[sequence_index]
            else:
                weights[sequence_id] = 0
            indices += list(range(start_index, start_index + samples_in_sequence_id[sequence_index]))
        start_index += samples_in_sequence_id[sequence_index]
        sequence_index += 1
    return indices, weights
