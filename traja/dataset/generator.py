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

    sequences_in_categories= list()

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


def shuffle_split(
        train_data: np.array,
        target_data: np.array,
        target_category: np.array,
        train_ratio: float,
        split: bool = True,
):
    """[summary]

    Args:
        train_data (np.array): [description]
        target_data (np.array): [description]
        target_category (np.array): [description]
        train_ratio (float): [description]
        split (bool): If True, split the data into train and test, else only shuffle the dataset and return it for training 

    Returns:
        [type]: [description]
    """

    # Shuffle the IDs and the corresponding sequence , preserving the order
    train_data, target_data, target_category = shuffle(
        train_data, target_data, target_category
    )

    assert train_ratio > 0, "Train data ratio should be greater than zero"
    assert train_ratio <= 1.0, "Train data ratio should be less than or equal to 1 "
    if split:
        # Train test split
        split = int(train_ratio * len(train_data))

        train_x = train_data[:split]
        train_y = target_data[:split]
        train_z = target_category[:split]

        test_x = train_data[split:]
        test_y = target_data[split:]
        test_z = target_category[split:]

        return [train_x, train_y, train_z], [test_x, test_y, test_z]
    else:
        return train_data, target_data, target_category


def scale_data(data, sequence_length):
    """[summary]

    Args:
        data ([type]): [description]
        sequence_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert len(data[0].shape) == 2
    scalers = {}
    data = np.vstack(data)

    for i in range(data.shape[1]):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        s_s = scaler.fit_transform(data[:, i].reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scalers["scaler_" + str(i)] = scaler
        data[:, i] = s_s
    # Slice the data into batches
    data = [data[i: i + sequence_length] for i in range(0, len(data), sequence_length)]
    return data, scalers


def weighted_random_samplers(train_z, test_z):
    """[summary]

    Args:
        train_z ([type]): [description]
        test_z ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Prepare weighted random sampler:
    train_target_list = torch.tensor(train_z).type(torch.LongTensor)
    test_target_list = torch.tensor(test_z).type(torch.LongTensor)

    # Number of classes and their frequencies
    train_targets_, train_class_count = get_class_distribution(train_target_list)
    test_targets_, test_class_count = get_class_distribution(test_target_list)

    # Compute class weights
    train_class_weights = 1.0 / torch.tensor(train_class_count, dtype=torch.float)
    test_class_weights = 1.0 / torch.tensor(test_class_count, dtype=torch.float)

    # Assign weights to original target list
    train_class_weights_all = train_class_weights[
        train_target_list - 1
        ]  # Note the targets start from 1, to python idx
    # to apply,-1
    test_class_weights_all = test_class_weights[test_target_list - 1]

    # Weighted samplers
    train_weighted_sampler = WeightedRandomSampler(
        weights=train_class_weights_all,
        num_samples=len(train_class_weights_all),
        replacement=True,
    )
    test_weighted_sampler = WeightedRandomSampler(
        weights=test_class_weights_all,
        num_samples=len(test_class_weights_all),
        replacement=True,
    )
    return train_weighted_sampler, test_weighted_sampler
