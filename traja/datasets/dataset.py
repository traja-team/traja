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
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd
from sklearn.utils import shuffle
from traja.datasets import utils
import random

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloader for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out


class TimeSeriesDataset(Dataset):
    r"""Pytorch Dataset object

    Args:
        Dataset (torch.utils.data.Dataset): Pyptorch dataset object
    """

    def __init__(self, data, target, category=None, parameters=None):
        r"""
        Args:
            data (array): Data
            target (array): Target
            category (array): Category
            parameters (array): Parameters
        """

        self.data = data
        self.target = target
        self.category = category
        self.parameters = parameters

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        z = self.category[index] if self.category else torch.zeros(1)
        w = self.parameters[index] if self.parameters else torch.zeros(1)
        return x, y, z, w

    def __len__(self):
        return len(self.data)


class MultiModalDataLoader:
    """
    MultiModalDataLoader wraps the following data preparation steps,
    
    1. Data generator: Extract x and y time series and corresponding ID (category) in the dataset. This process split the dataset into 
                        i) Train samples with sequence length equals n_past
                        ii) Target samples with sequence length equals n_future 
                        iii) Target category(ID) of both train and target data
    2. Data scalling: Scale the train and target data columns between the range (-1,1) using MinMaxScalers; TODO: It is more optimal to scale data for each ID(category)
    3. Data shuffling: Shuffle the order of samples in the dataset without loosing the train<->target<->category combination
    4. Create train test split: Split the shuffled batches into train (data, target, category) and test(data, target, category)
    5. Weighted Random sampling: Apply weights with respect to category counts in the dataset: category_sample_weight = 1/num_category_samples; This avoid model overfit to category appear often in the dataset 
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
            num_target_categories: If validation_split_criteria is "category", then num_classes_in_validation_data should be not None. 
                                            N number of classes in dataset will be used in validation dataset
            scale (bool): If True, scale the input and target and return the corresponding scalers in a dict. 

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
        num_workers: int,
        train_split_ratio: float,
        validation_split_ratio: float = None,
        num_val_categories: int = None,
        scale: bool = True,
        test: bool = True,
    ):
        self.df = df
        self.batch_size = batch_size
        self.n_past = n_past
        self.n_future = n_future
        self.num_workers = num_workers
        self.test = test
        self.train_split_ratio = train_split_ratio
        self.validation_split_ratio = validation_split_ratio
        self.scale = scale
        self.num_val_categories = num_val_categories

        if self.num_val_categories is not None:
            assert (
                self.validation_split_ratio is not None
            ), "Invalid validation argument, validation_split_ratio not supported for category based validation split"

            self.set_validation()
        if self.validation_split_ratio is not None:
            assert (
                self.validation_split_ratio is not None
            ), "Invalid validation argument, num_val_categories not supported for sequence based validation split"
            self.set_validation()

        # Train and test data from df-val_df
        train_data, target_data, target_category = utils.generate_dataset(
            self.df, self.n_past, self.n_future
        )

        if test:
            # Shuffle and split the data
            [train_x, train_y, train_z], [test_x, test_y, test_z] = utils.shuffle_split(
                train_data,
                target_data,
                target_category,
                train_ratio=self.train_split_ratio,
                split=True,
            )
        else:
            [train_x, train_y, train_z] = utils.shuffle_split(
                train_data, target_data, target_category, train_ratio=None, split=False
            )

        # Scale data
        if self.scale and self.test:
            (train_x, self.train_x_scaler), (train_y, self.train_y_scaler) = (
                utils.scale_data(train_x, sequence_length=self.n_past),
                utils.scale_data(train_y, sequence_length=self.n_future),
            )

            (test_x, self.test_x_scaler), (test_y, self.test_y_scaler) = (
                utils.scale_data(test_x, sequence_length=self.n_past),
                utils.scale_data(test_y, sequence_length=self.n_future),
            )

            # Weighted Random Sampler
            (
                train_weighted_sampler,
                test_weighted_sampler,
            ) = utils.weighted_random_samplers(train_z, test_z)

        if self.scale and not self.test:
            (train_x, self.train_x_scaler), (train_y, self.train_y_scaler) = (
                utils.scale_data(train_x, sequence_length=self.n_past),
                utils.scale_data(train_y, sequence_length=self.n_future),
            )

            # Weighted Random Sampler
            (
                train_weighted_sampler,
                test_weighted_sampler,
            ) = utils.weighted_random_samplers(train_z, test_z)

        # Weighted Random Sampler
        train_weighted_sampler, test_weighted_sampler = utils.weighted_random_samplers(
            train_z, test_z
        )

        # Dataset
        train_dataset = TimeSeriesDataset(train_x, train_y, train_z)
        test_dataset = TimeSeriesDataset(test_x, test_y, test_z)

        # Dataloader with weighted samplers
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=train_weighted_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=test_weighted_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        if self.validation_split_ratio is not None and self.test:

            self.dataloaders = {
                "train_loader": self.train_loader,
                "test_loader": self.test_loader,
                "validation_loader": self.validation_loader,
            }
            if self.scale:

                self.scalers = {
                    "train_data_scaler": self.train_x_scaler,
                    "train_target_scaler": self.train_y_scaler,
                    "test_data_scaler": self.test_x_scaler,
                    "test_target_scaler": self.test_y_scaler,
                    "val_data_scaler": self.val_x_scaler,
                    "val_target_scaler": self.val_y_scaler,
                }
            else:
                self.scalers = None
                
        elif self.test and self.validation_split_ratio is None:

            self.dataloaders = {
                "train_loader": self.train_loader,
                "test_loader": self.test_loader,
            }
            if self.scale:

                self.scalers = {
                    "train_data_scaler": self.train_x_scaler,
                    "train_target_scaler": self.train_y_scaler,
                    "test_data_scaler": self.test_x_scaler,
                    "test_target_scaler": self.test_y_scaler,
                }
            else:
                self.scalers = None
                
        elif self.validation_split_ratio is not None and not self.test:
             
            
            

    def set_validation(self):
        """[summary]

        Args:
            target_categories (list, optional): [description]. Defaults to None.
        """

        if self.validation_split_ratio is None and self.num_val_categories is not None:
            max_ID = self.df["ID"].max()
            val_categories = random.sample(range(1, max_ID), self.num_val_categories)
            self.df_val = self.df.loc[self.df["ID"].isin(val_categories)]

        if self.validation_split_ratio is not None and self.num_val_categories is None:
            # Prepare validation data before train and test and their splits
            self.df_val = self.df.groupby("ID").tail(
                self.validation_split_ratio * len(self.df)
            )

        # Generate validation dataset
        val_x, val_y, val_z = utils.generate_dataset(
            self.df_val, self.n_past, self.n_future
        )
        if self.scale:
            # Scale validation data:
            (val_x, self.val_x_scaler), (val_y, self.val_y_scaler) = (
                utils.scale_data(val_x, sequence_length=self.n_past),
                utils.scale_data(val_y, sequence_length=self.n_future),
            )
        # Generate Pytorch dataset
        val_dataset = TimeSeriesDataset(val_x, val_y, val_z)

        self.validation_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            sampler=None,
            drop_last=True,
            num_workers=self.num_workers,
        )

        # Create new df for train and test; Difference of df with df_val
        self.df = self.df.loc[self.df.index.difference(self.df_val.index)]

    def __new__(
        cls,
        df: pd.DataFrame,
        batch_size: int,
        n_past: int,
        n_future: int,
        num_workers: int,
        train_split_ratio: float,
        validation_split_ratio: float = None,
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
            train_split_ratio,
            validation_split_ratio,
        )
        # Return train and test loader attributes
        return loader_instance.dataloaders, loader_instance.scalers

