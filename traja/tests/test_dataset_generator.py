import pandas as pd
import numpy as np

from traja.dataset import dataset


def test_time_based_sampling_dataloaders_do_not_overlap():
    data = list()
    num_ids = 140
    sequence_length = 2000

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.501
    validation_split_ratio = 0.25

    split_by_id = False  # The test condition

    # The train[0] column should contain only 1s, the test column should contain 2s and the
    # validation column set should contain 3s.
    # When scaled, this translates to -1., 0 and 1. respectively.
    for sample_id in range(num_ids):
        for element in range(round(sequence_length * train_split_ratio)):
            data.append([1, element, sample_id])
        for element in range(round(sequence_length * (1 - train_split_ratio - validation_split_ratio))):
            data.append([2, element, sample_id])
        for element in range(round(sequence_length * validation_split_ratio)):
            data.append([3, element, sample_id])

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    dataloaders = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio,
                                               split_by_id=split_by_id)

    for data, target, ids, parameters in dataloaders['train_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == -1.
        for sequence in target:
            for sample in sequence:
                assert sample[0] == -1.

    for data, target, ids, parameters in dataloaders['test_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == 0
        for sequence in target:
            for sample in sequence:
                assert sample[0] == 0

    for data, target, ids, parameters in dataloaders['validation_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == 1
        for sequence in target:
            for sample in sequence:
                assert sample[0] == 1


def test_time_based_sampling_dataloaders_with_short_stride_do_not_overlap():
    data = list()
    num_ids = 140
    sequence_length = 2000

    # Hyperparameters
    batch_size = 15
    num_past = 10
    num_future = 5
    train_split_ratio = 0.498
    validation_split_ratio = 0.25

    stride = 5

    split_by_id = False  # The test condition

    # The train[0] column should contain only 1s, the test column should contain 2s and the
    # validation column set should contain 3s.
    # When scaled, this translates to -1., 0 and 1. respectively.
    for sample_id in range(num_ids):
        for element in range(round(sequence_length * train_split_ratio) - 6):
            data.append([1, element, sample_id])
        for element in range(round(sequence_length * (1 - train_split_ratio - validation_split_ratio)) + -4):
            data.append([2, element, sample_id])
        for element in range(round(sequence_length * validation_split_ratio) + 10):
            data.append([3, element, sample_id])

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    dataloaders = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio,
                                               split_by_id=split_by_id,
                                               stride=stride)

    for data, target, ids, parameters in dataloaders['train_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == -1.
        for sequence in target:
            for sample in sequence:
                assert sample[0] == -1.

    for data, target, ids, parameters in dataloaders['test_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == 0
        for sequence in target:
            for sample in sequence:
                assert sample[0] == 0

    for data, target, ids, parameters in dataloaders['validation_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == 1
        for sequence in target:
            for sample in sequence:
                assert sample[0] == 1


def test_time_based_weighted_sampling_dataloaders_do_not_overlap():
    pass


def test_id_wise_sampling_with_few_ids_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 5
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40 + int(sequence_id / 14)):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    # Hyperparameters
    batch_size = 1
    num_past = 10
    num_future = 5
    train_split_ratio = 0.5
    validation_split_ratio = 0.2

    dataloaders = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio,
                                               scale=False)

    verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids)


def test_id_wise_sampling_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 150
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio,
                                               scale=False)

    verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids)


def verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids):
    train_ids = []  # We check that the sequence IDs are not mixed
    train_sample_ids = []  # We also check that the sample IDs do not overlap
    for data, target, ids, parameters in dataloaders['train_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in train_ids:
                train_ids.append(sequence_id)
            train_sample_ids.append(int(data[index][0][1]))

    assert len(train_ids) == round(train_split_ratio * num_ids), 'Wrong number of training ids!'

    test_ids = []
    test_sample_ids = []
    for data, target, ids, parameters in dataloaders['test_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in test_ids:
                test_ids.append(sequence_id)
            test_sample_ids.append(int(data[index][0][1]))

            assert sequence_id not in train_ids, 'Found test data in train loader!'


    validation_ids = []
    validation_sample_ids = []
    for data, target, ids, parameters in dataloaders['validation_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in validation_ids:
                validation_ids.append(sequence_id)
            validation_sample_ids.append(int(data[index][0][1]))

            assert sequence_id not in train_ids, 'Found validation data in train loader!'
            assert sequence_id not in test_ids, 'Found validation data in test loader!'

    assert len(validation_ids) == round(
        validation_split_ratio * num_ids), 'Wrong number of validation ids!'
    assert len(train_ids) + len(test_ids) + len(
        validation_ids) == num_ids, 'Wrong number of ids!'

    train_sequential_sample_ids = []
    for data, target, ids, parameters in dataloaders['sequential_train_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            train_sequential_sample_ids.append(int(data[index][0][1]))
            assert sequence_id in train_ids, f'train_ids missing id {sequence_id}!'

    train_sample_ids = sorted(train_sample_ids)
    assert len(train_sample_ids) == len(train_sequential_sample_ids), 'train and sequential_train loaders have different lengths!'
    for index in range(len(train_sample_ids)):
        assert train_sample_ids[index] == train_sequential_sample_ids[index], f'Index {train_sample_ids[index]} is not equal to {train_sequential_sample_ids[index]}!'

    test_sequential_sample_ids = []
    for data, target, ids, parameters in dataloaders['sequential_test_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            test_sequential_sample_ids.append(int(data[index][0][1]))
            assert sequence_id in test_ids, f'test_ids missing id {sequence_id}!'

    test_sample_ids = sorted(test_sample_ids)
    assert len(test_sample_ids) == len(
        test_sequential_sample_ids), 'test and sequential_test loaders have different lengths!'
    for index in range(len(test_sample_ids)):
        assert test_sample_ids[index] == test_sequential_sample_ids[
            index], f'Index {test_sample_ids[index]} is not equal to {test_sequential_sample_ids[index]}!'

    validation_sequential_sample_ids = []
    for data, target, ids, parameters in dataloaders['sequential_validation_loader']:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            validation_sequential_sample_ids.append(int(data[index][0][1]))
            assert sequence_id in validation_ids, f'validation_ids missing id {sequence_id}!'

    validation_sample_ids = sorted(validation_sample_ids)
    assert len(validation_sample_ids) == len(validation_sequential_sample_ids), 'validation and sequential_validation loaders have different lengths!'
    for index in range(len(validation_sample_ids)):
        assert validation_sample_ids[index] == validation_sequential_sample_ids[index], f'Index {validation_sample_ids[index]} is not equal to {validation_sequential_sample_ids[index]}!'


def test_sequential_data_loader_indices_are_sequential():
    data = list()
    num_ids = 46

    for sample_id in range(num_ids):
        for sequence in range(40 + int(sample_id / 14)):
            data.append([sequence, sequence, sample_id])

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    # Hyperparameters
    batch_size = 18
    num_past = 13
    num_future = 8
    train_split_ratio = 0.5
    validation_split_ratio = 0.2
    stride = 1

    dataloaders = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio,
                                               stride=stride)

    current_id = 0
    for data, target, ids, parameters in dataloaders['sequential_train_loader']:
        for id in ids:
            id = int(id)
            if id > current_id:
                current_id = id
            assert id == current_id, 'IDs in sequential train loader should increase monotonically!'

    current_id = 0
    for data, target, ids, parameters in dataloaders['sequential_test_loader']:
        for id in ids:
            id = int(id)
            if id > current_id:
                current_id = id
            assert id == current_id, 'IDs in sequential test loader should increase monotonically!'
