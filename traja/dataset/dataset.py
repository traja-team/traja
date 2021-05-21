"""
Modified from https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py.

This module contains:

Classes:
1. Pytorch Time series dataset class instance
2. Weighted train and test dataset loader with respect to class distribution

Helpers:
1. Class distribution in the dataset

"""
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from traja.dataset import generator
from traja.dataset.generator import get_indices_from_sequence_ids

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    r"""Pytorch Dataset object

    Args:
        Dataset (torch.utils.data.Dataset): Pyptorch dataset object
    """

    def __init__(
        self,
        data,
        target,
        sequence_ids=None,
        parameters=None,
        classes=None,
        scaler: TransformerMixin = None,
    ):
        r"""
        Args:
            data (array): Data
            target (array): Target
            sequence_ids (array): Sequence ID
            parameters (array): Parameters
            classes (array): Sequence classes
            scaler (sklearn.base.TransformerMixin)
        """

        self.data = data
        self.target = target
        self.sequence_ids = sequence_ids
        self.parameters = parameters
        self.classes = classes
        self.scaler = scaler

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        ids = self.sequence_ids[index] if self.sequence_ids else torch.zeros(1)
        parameters = self.parameters[index] if self.parameters else torch.zeros(1)
        classes = self.classes[index] if self.classes else torch.zeros(1)

        if self.scaler is not None:
            data = torch.tensor(self.scaler.transform(data))
            target = torch.tensor(self.scaler.transform(target))
        return data, target, ids, parameters, classes

    def __len__(self):
        return len(self.data)


class MultiModalDataLoader:
    """
    MultiModalDataLoader wraps the following data preparation steps,

    1. Data generator: Extract x and y time series and corresponding ID (sequence_id) in the dataset. This process split the dataset into
                        i) Train samples with sequence length equals n_past
                        ii) Target samples with sequence length equals n_future
                        iii) Target sequence_id(ID) of both train and target data
    2. Data scalling: Scale the train and target data columns between the range (-1,1) using MinMaxScalers; TODO: It is more optimal to scale data for each ID(sequence_id)
    3. Data shuffling: Shuffle the order of samples in the dataset without loosing the train<->target<->sequence_id combination
    4. Create train test split: Split the shuffled batches into train (data, target, sequence_id) and test(data, target, sequence_id)
    5. Weighted Random sampling: Apply weights with respect to sequence_id counts in the dataset: category_sample_weight = 1/num_category_samples; This avoid model overfit to sequence_id appear often in the dataset
    6. Create pytorch Dataset instances
    7. Returns the train and test data loader instances along with their scalers as a dictionaries given the dataset instances and batch size

        Args:
            df (pd.DataFrame): Dataset
            batch_size (int): Number of samples per batch of data
            n_past (int): Input sequence length. Number of time steps from the past.
            n_future (int): Target sequence length. Number of time steps to the future.
            num_workers (int): Number of cpu subprocess occupied during data loading process
            train_split_ratio (float):Should be between 0.0 and 1.0 and represent the proportion of the dataset-validation_dataset
                                      to include in the train split.
            validation_split_ratio (float): Should be between 0.0 and 1.0 and represent the proportion of the dataset
                                      to include in the validation split.
            stride: Size of the sliding window. Defaults to sequence_length
            split_by_id (bool): Whether to split data based on the sequence's ID (default) or split each sequence
                                length-wise.
            scale (bool): If True, scale the input and target and return the corresponding scalers in a dict.
            parameter_columns (list): Columns in data frame with regression parameters.
            weighted_sampling (bool): Whether to weigh the likelihood of picking each sample by the sequence length.
                                      This balances the accuracy if trajectories have different lengths.

        Usage:
        ------
        dataloaders, scalers = MultiModalDataLoader(df = data_frame, batch_size=32, n_past = 20, n_future = 10, num_workers=4)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        n_past: int,
        n_future: int,
        num_workers: int = 1,
        train_split_ratio: float = 0.4,
        validation_split_ratio: float = 0.2,
        stride: int = None,
        split_by_id: bool = True,
        scale: bool = True,
        test: bool = True,
        parameter_columns: list = [],
        weighted_sampling: bool = False,
    ):
        self.df = df
        self.batch_size = batch_size
        self.n_past = n_past
        self.n_future = n_future
        self.num_workers = num_workers
        self.test = test
        self.train_split_ratio = train_split_ratio
        self.validation_split_ratio = validation_split_ratio
        self.split_by_id = split_by_id
        self.scale = scale
        self.stride = stride

        # Train and test data from df-val_df
        (
            train_data,
            target_data,
            target_ids,
            target_parameters,
            target_classes,
            samples_in_sequence_id,
        ) = generator.generate_dataset(
            self.df,
            self.n_past,
            self.n_future,
            stride=self.stride,
            parameter_columns=parameter_columns,
        )

        if self.scale:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(np.vstack(train_data + target_data))
        else:
            scaler = None

        # Dataset
        dataset = TimeSeriesDataset(
            train_data,
            target_data,
            target_ids,
            target_parameters,
            target_classes,
            scaler=scaler,
        )

        # We initialise sample weights in case we need them to weigh samples.
        train_weights = defaultdict(float)
        test_weights = defaultdict(float)
        validation_weights = defaultdict(float)

        if self.split_by_id:
            ids = list(set(target_ids))
            np.random.shuffle(ids)

            train_split_index = round(train_split_ratio * len(ids))
            validation_split_index = round((1 - validation_split_ratio) * len(ids))

            train_ids = np.sort(ids[:train_split_index])
            test_ids = np.sort(ids[train_split_index:validation_split_index])
            validation_ids = np.sort(ids[validation_split_index:])

            train_indices, train_weights = get_indices_from_sequence_ids(
                train_ids, samples_in_sequence_id
            )
            test_indices, test_weights = get_indices_from_sequence_ids(
                test_ids, samples_in_sequence_id
            )
            validation_indices, validation_weights = get_indices_from_sequence_ids(
                validation_ids, samples_in_sequence_id
            )

        else:  # Do not sample by sequence ID
            if stride is None:
                stride = n_past + n_future

            sequence_length = n_past + n_future
            train_indices = list()
            test_indices = list()
            validation_indices = list()
            id_start_index = 0
            for sequence_index, sequence_count in enumerate(samples_in_sequence_id):
                overlap = math.ceil(sequence_length / stride)

                start_test_index = round(sequence_count * train_split_ratio)
                end_train_index = start_test_index - overlap

                start_validation_index = round(
                    sequence_count * (1 - validation_split_ratio)
                )
                end_test_index = start_validation_index - overlap

                train_indices.extend(
                    list(range(id_start_index, id_start_index + end_train_index))
                )
                test_indices.extend(
                    list(
                        range(
                            id_start_index + start_test_index,
                            id_start_index + end_test_index,
                        )
                    )
                )
                validation_indices.extend(
                    list(
                        range(
                            id_start_index + start_validation_index,
                            id_start_index + sequence_count,
                        )
                    )
                )

                train_weights[sequence_index] = (
                    1.0 / end_train_index if end_train_index > 0 else 0
                )
                test_weights[sequence_index] = (
                    1.0 / (end_test_index - start_test_index)
                    if (end_test_index - start_test_index) > 0
                    else 0
                )
                validation_weights[sequence_index] = (
                    1.0 / (sequence_count - start_validation_index)
                    if (sequence_count - start_validation_index) > 0
                    else 0
                )

                id_start_index += sequence_count

        sequential_train_dataset = torch.utils.data.Subset(
            dataset, np.sort(train_indices[:])
        )
        sequential_test_dataset = torch.utils.data.Subset(
            dataset, np.sort(test_indices[:])
        )
        sequential_validation_dataset = torch.utils.data.Subset(
            dataset, np.sort(validation_indices[:])
        )

        if weighted_sampling:
            train_index_weights = list()
            test_index_weights = list()
            validation_index_weights = list()

            for (
                data,
                target,
                sequence_id,
                parameters,
                classes,
            ) in sequential_train_dataset:
                train_index_weights.append(train_weights[sequence_id])
            for (
                data,
                target,
                sequence_id,
                parameters,
                classes,
            ) in sequential_test_dataset:
                test_index_weights.append(test_weights[sequence_id])
            for (
                data,
                target,
                sequence_id,
                parameters,
                classes,
            ) in sequential_validation_dataset:
                validation_index_weights.append(validation_weights[sequence_id])

            train_dataset = sequential_train_dataset
            test_dataset = sequential_test_dataset
            validation_dataset = sequential_validation_dataset

            train_sampler = WeightedRandomSampler(
                weights=train_index_weights,
                num_samples=len(train_index_weights),
                replacement=True,
            )
            test_sampler = WeightedRandomSampler(
                weights=test_index_weights,
                num_samples=len(test_index_weights),
                replacement=True,
            )
            validation_sampler = WeightedRandomSampler(
                weights=validation_index_weights,
                num_samples=len(validation_index_weights),
                replacement=True,
            )

        else:
            train_dataset = dataset
            test_dataset = dataset
            validation_dataset = dataset

            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            np.random.shuffle(validation_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            validation_sampler = SubsetRandomSampler(validation_indices)

        # Dataloader
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=test_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=validation_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        self.sequential_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=num_workers,
        )
        self.sequential_train_loader = torch.utils.data.DataLoader(
            dataset=sequential_train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=num_workers,
        )
        self.sequential_test_loader = torch.utils.data.DataLoader(
            dataset=sequential_test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=num_workers,
        )
        self.sequential_validation_loader = torch.utils.data.DataLoader(
            dataset=sequential_validation_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=num_workers,
        )

        self.dataloaders = {
            "train_loader": self.train_loader,
            "test_loader": self.test_loader,
            "validation_loader": self.validation_loader,
            "sequential_loader": self.sequential_loader,
            "sequential_train_loader": self.sequential_train_loader,
            "sequential_test_loader": self.sequential_test_loader,
            "sequential_validation_loader": self.sequential_validation_loader,
        }

    def __new__(
        cls,
        df: pd.DataFrame,
        batch_size: int,
        n_past: int,
        n_future: int,
        num_workers: int,
        split_by_id: bool = True,
        stride: int = None,
        train_split_ratio: float = 0.4,
        validation_split_ratio: float = 0.2,
        scale: bool = True,
        parameter_columns: list = list(),
        weighted_sampling: bool = False,
    ):
        """Constructor of MultiModalDataLoader"""
        # Loader instance
        loader_instance = super(MultiModalDataLoader, cls).__new__(cls)
        loader_instance.__init__(
            df,
            batch_size,
            n_past,
            n_future,
            num_workers,
            train_split_ratio=train_split_ratio,
            validation_split_ratio=validation_split_ratio,
            split_by_id=split_by_id,
            stride=stride,
            scale=scale,
            parameter_columns=parameter_columns,
            weighted_sampling=weighted_sampling,
        )
        # Return train and test loader attributes
        return loader_instance.dataloaders
