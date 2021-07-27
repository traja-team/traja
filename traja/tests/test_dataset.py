import os
import pandas as pd
import pytest

from traja.dataset import dataset
from traja.dataset.pituitary_gland import create_latin_hypercube_sampled_pituitary_df


@pytest.mark.skipif(os.name == 'nt', reason="hangs on Windows for unknown reason")
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
        for element in range(
            round(sequence_length * (1 - train_split_ratio - validation_split_ratio))
        ):
            data.append([2, element, sample_id])
        for element in range(round(sequence_length * validation_split_ratio)):
            data.append([3, element, sample_id])

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        split_by_id=split_by_id,
    )

    for data, target, ids, parameters, classes in dataloaders["train_loader"]:
        for sequence in data:
            assert all(sample == -1.0 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == -1.0 for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["test_loader"]:
        for sequence in data:
            assert all(sample == 0 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 0 for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["validation_loader"]:
        for sequence in data:
            assert all(sample == 1 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 1 for sample in sequence[:,0])


def test_time_based_sampling_dataloaders_do_not_overlap():
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
        for element in range(
            round(sequence_length * (1 - train_split_ratio - validation_split_ratio))
            + -4
        ):
            data.append([2, element, sample_id])
        for element in range(round(sequence_length * validation_split_ratio) + 10):
            data.append([3, element, sample_id])

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        split_by_id=split_by_id,
        stride=stride,
    )

    for data, target, ids, parameters, classes in dataloaders["train_loader"]:
        for sequence in data:
            assert all(sample == -1. for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == -1. for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["test_loader"]:
        for sequence in data:
            assert all(sample == 0 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 0 for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["validation_loader"]:
        for sequence in data:
            assert all(sample == 1 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 1 for sample in sequence[:,0])

def test_time_based_sampling_dataloaders_with_stride_one_do_not_overlap():
    data = list()
    num_ids = 2
    sequence_length = 200

    # Hyperparameters
    batch_size = 15
    num_past = 10
    num_future = 5
    train_split_ratio = 0.5
    validation_split_ratio = 0.25

    stride = 1

    split_by_id = False  # The test condition

    # The train[0] column should contain only 1s, the test column should contain 2s and the
    # validation column set should contain 3s.
    # When scaled, this translates to -1., 0 and 1. respectively.
    for sample_id in range(num_ids):
        for element in range(round(sequence_length * train_split_ratio) - 8):
            data.append([1, element, sample_id])
        for element in range(
            round(sequence_length * (1 - train_split_ratio - validation_split_ratio))
            - 4
        ):
            data.append([2, element, sample_id])
        for element in range(round(sequence_length * validation_split_ratio) + 12):
            data.append([3, element, sample_id])

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=4,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        split_by_id=split_by_id,
        stride=stride,
    )

    for data, target, ids, parameters, classes in dataloaders["train_loader"]:
        for sequence in data:
            assert all(sample == -1. for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == -1. for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["test_loader"]:
        for sequence in data:
            assert all(sample == 0 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 0 for sample in sequence[:,0])

    for data, target, ids, parameters, classes in dataloaders["validation_loader"]:
        for sequence in data:
            assert all(sample == 1 for sample in sequence[:,0])
        for sequence in target:
            assert all(sample == 1 for sample in sequence[:,0])


def test_time_based_weighted_sampling_dataloaders_do_not_overlap():
    data = list()
    num_ids = 232
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40 + (int(sequence_id * 2.234) % 117)):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        scale=False,
        split_by_id=False,
        weighted_sampling=True,
        stride=1,
    )

    train_ids = extract_sample_ids_from_dataloader(dataloaders["train_loader"])
    test_ids = extract_sample_ids_from_dataloader(dataloaders["test_loader"])
    validation_ids = extract_sample_ids_from_dataloader(
        dataloaders["validation_loader"]
    )
    sequential_train_ids = extract_sample_ids_from_dataloader(
        dataloaders["sequential_train_loader"]
    )
    sequential_test_ids = extract_sample_ids_from_dataloader(
        dataloaders["sequential_test_loader"]
    )
    sequential_validation_ids = extract_sample_ids_from_dataloader(
        dataloaders["sequential_validation_loader"]
    )

    verify_that_indices_belong_to_precisely_one_loader(
        train_ids, test_ids, validation_ids
    )
    verify_that_indices_belong_to_precisely_one_loader(
        sequential_train_ids, sequential_test_ids, sequential_validation_ids
    )


def test_id_wise_sampling_with_few_ids_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 5
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40 + int(sequence_id / 14)):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 1
    num_past = 10
    num_future = 5
    train_split_ratio = 0.5
    validation_split_ratio = 0.2

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        scale=False,
    )

    verify_sequential_id_sampled_sequential_dataloaders_equal_dataloaders(
        dataloaders, train_split_ratio, validation_split_ratio, num_ids
    )


def test_id_wise_sampling_with_short_sequences_does_not_divide_by_zero():
    data = list()
    num_ids = 283
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(
            1 + (sequence_id % 74)
        ):  # Some sequences will generate zero time series
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 1
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        scale=False,
    )

    verify_sequential_id_sampled_sequential_dataloaders_equal_dataloaders(
        dataloaders,
        train_split_ratio,
        validation_split_ratio,
        num_ids,
        expect_all_ids=False,
    )


def test_id_wise_sampling_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 150
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        scale=False,
    )

    verify_sequential_id_sampled_sequential_dataloaders_equal_dataloaders(
        dataloaders, train_split_ratio, validation_split_ratio, num_ids
    )


def test_id_wise_weighted_sampling_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 150
    sample_id = 0

    for sequence_id in range(num_ids):
        for sequence in range(40 + (int(sequence_id * 2.234) % 117)):
            data.append([sequence, sample_id, sequence_id])
            sample_id += 1

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        scale=False,
        weighted_sampling=True,
        stride=1,
    )

    verify_id_wise_sampled_dataloaders_do_not_overlap(
        dataloaders, train_split_ratio, validation_split_ratio, num_ids
    )


def extract_sample_ids_from_dataloader(dataloader):
    sample_ids = list()
    for data, target, ids, parameters, classes in dataloader:
        for index, sequence_id in enumerate(ids):
            sample_ids.append(int(data[index][0][1]))
    return sample_ids


def verify_id_wise_sampled_dataloaders_do_not_overlap(
    dataloaders, train_split_ratio, validation_split_ratio, num_ids, expect_all_ids=True
):
    train_ids = []  # We check that the sequence IDs are not mixed
    train_sample_ids = []  # We also check that the sample IDs do not overlap
    for data, target, ids, parameters, classes in dataloaders["train_loader"]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in train_ids:
                train_ids.append(sequence_id)
            train_sample_ids.append(int(data[index][0][1]))

    test_ids = []
    test_sample_ids = []
    for data, target, ids, parameters, classes in dataloaders["test_loader"]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in test_ids:
                test_ids.append(sequence_id)
            test_sample_ids.append(int(data[index][0][1]))

            assert sequence_id not in train_ids, "Found test data in train loader!"

    validation_ids = []
    validation_sample_ids = []
    for data, target, ids, parameters, classes in dataloaders["validation_loader"]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            if sequence_id not in validation_ids:
                validation_ids.append(sequence_id)
            validation_sample_ids.append(int(data[index][0][1]))

            assert (
                sequence_id not in train_ids
            ), "Found validation data in train loader!"
            assert sequence_id not in test_ids, "Found validation data in test loader!"

    if expect_all_ids:
        assert len(train_ids) == round(
            train_split_ratio * num_ids
        ), "Wrong number of training ids!"
        assert len(validation_ids) == round(
            validation_split_ratio * num_ids
        ), "Wrong number of validation ids!"
        assert (
            len(train_ids) + len(test_ids) + len(validation_ids) == num_ids
        ), "Wrong number of ids!"

    return (
        train_ids,
        train_sample_ids,
        test_ids,
        test_sample_ids,
        validation_ids,
        validation_sample_ids,
    )


def verify_sequential_id_sampled_sequential_dataloaders_equal_dataloaders(
    dataloaders, train_split_ratio, validation_split_ratio, num_ids, expect_all_ids=True
):
    (
        train_ids,
        train_sample_ids,
        test_ids,
        test_sample_ids,
        validation_ids,
        validation_sample_ids,
    ) = verify_id_wise_sampled_dataloaders_do_not_overlap(
        dataloaders, train_split_ratio, validation_split_ratio, num_ids, expect_all_ids
    )

    # We check that all sample IDs are present in the sequential samplers and vice versa
    train_sequential_sample_ids = []
    for data, target, ids, parameters, classes in dataloaders[
        "sequential_train_loader"
    ]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            train_sequential_sample_ids.append(int(data[index][0][1]))
            assert sequence_id in train_ids, f"train_ids missing id {sequence_id}!"

    train_sample_ids = sorted(train_sample_ids)
    assert len(train_sample_ids) == len(
        train_sequential_sample_ids
    ), "train and sequential_train loaders have different lengths!"
    for index in range(len(train_sample_ids)):
        assert (
            train_sample_ids[index] == train_sequential_sample_ids[index]
        ), f"Index {train_sample_ids[index]} is not equal to {train_sequential_sample_ids[index]}!"

    test_sequential_sample_ids = []
    for data, target, ids, parameters, classes in dataloaders["sequential_test_loader"]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            test_sequential_sample_ids.append(int(data[index][0][1]))
            assert sequence_id in test_ids, f"test_ids missing id {sequence_id}!"

    test_sample_ids = sorted(test_sample_ids)
    assert len(test_sample_ids) == len(
        test_sequential_sample_ids
    ), "test and sequential_test loaders have different lengths!"
    for index in range(len(test_sample_ids)):
        assert (
            test_sample_ids[index] == test_sequential_sample_ids[index]
        ), f"Index {test_sample_ids[index]} is not equal to {test_sequential_sample_ids[index]}!"

    validation_sequential_sample_ids = []
    for data, target, ids, parameters, classes in dataloaders[
        "sequential_validation_loader"
    ]:
        for index, sequence_id in enumerate(ids):
            sequence_id = int(sequence_id)
            validation_sequential_sample_ids.append(int(data[index][0][1]))
            assert (
                sequence_id in validation_ids
            ), f"validation_ids missing id {sequence_id}!"

    validation_sample_ids = sorted(validation_sample_ids)
    assert len(validation_sample_ids) == len(
        validation_sequential_sample_ids
    ), "validation and sequential_validation loaders have different lengths!"
    for index in range(len(validation_sample_ids)):
        assert (
            validation_sample_ids[index] == validation_sequential_sample_ids[index]
        ), f"Index {validation_sample_ids[index]} is not equal to {validation_sequential_sample_ids[index]}!"

    verify_that_indices_belong_to_precisely_one_loader(
        train_sample_ids, test_sample_ids, validation_sample_ids
    )
    # Check that all indices belong to precisely one loader
    # Note that (because some samples are dropped and because we only check the first value in data)
    # not all indices are in a loader.
    train_index = 0
    test_index = 0
    validation_index = 0
    for index in range(
        len(train_sample_ids) + len(test_sample_ids) + len(validation_sample_ids)
    ):
        if train_sample_ids[train_index] < index:
            train_index += 1
        if test_sample_ids[test_index] < index:
            test_index += 1
        if validation_sample_ids[validation_index] < index:
            validation_index += 1
        index_is_in_train = train_sample_ids[train_index] == index
        index_is_in_test = test_sample_ids[test_index] == index
        index_is_in_validation = validation_sample_ids[validation_index] == index

        assert not (
            index_is_in_train and index_is_in_test
        ), f"Index {index} is in both the train and test loaders!"
        assert not (
            index_is_in_train and index_is_in_validation
        ), f"Index {index} is in both the train and validation loaders!"
        assert not (
            index_is_in_test and index_is_in_validation
        ), f"Index {index} is in both the test and validation loaders!"


def verify_that_indices_belong_to_precisely_one_loader(
    train_sample_ids, test_sample_ids, validation_sample_ids
):
    # Check that all indices belong to precisely one loader
    # Note that (because some samples are dropped and because we only check the first value in data)
    # not all indices are in a loader.
    train_index = 0
    test_index = 0
    validation_index = 0
    for index in range(
        len(train_sample_ids) + len(test_sample_ids) + len(validation_sample_ids)
    ):
        if train_sample_ids[train_index] < index:
            train_index += 1
        if test_sample_ids[test_index] < index:
            test_index += 1
        if validation_sample_ids[validation_index] < index:
            validation_index += 1
        index_is_in_train = train_sample_ids[train_index] == index
        index_is_in_test = test_sample_ids[test_index] == index
        index_is_in_validation = validation_sample_ids[validation_index] == index

        assert not (
            index_is_in_train and index_is_in_test
        ), f"Index {index} is in both the train and test loaders!"
        assert not (
            index_is_in_train and index_is_in_validation
        ), f"Index {index} is in both the train and validation loaders!"
        assert not (
            index_is_in_test and index_is_in_validation
        ), f"Index {index} is in both the test and validation loaders!"


def test_sequential_data_loader_indices_are_sequential():
    data = list()
    num_ids = 46

    for sample_id in range(num_ids):
        for sequence in range(40 + int(sample_id / 14)):
            data.append([sequence, sequence, sample_id])

    df = pd.DataFrame(data, columns=["x", "y", "ID"])

    # Hyperparameters
    batch_size = 18
    num_past = 13
    num_future = 8
    train_split_ratio = 0.5
    validation_split_ratio = 0.2
    stride = 1

    dataloaders = dataset.MultiModalDataLoader(
        df,
        batch_size=batch_size,
        n_past=num_past,
        n_future=num_future,
        num_workers=1,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        stride=stride,
    )

    current_id = 0
    for data, target, ids, parameters, classes in dataloaders[
        "sequential_train_loader"
    ]:
        for id in ids:
            id = int(id)
            if id > current_id:
                current_id = id
            assert (
                id == current_id
            ), "IDs in sequential train loader should increase monotonically!"

    current_id = 0
    for data, target, ids, parameters, classes in dataloaders["sequential_test_loader"]:
        for id in ids:
            id = int(id)
            if id > current_id:
                current_id = id
            assert (
                id == current_id
            ), "IDs in sequential test loader should increase monotonically!"


def test_pituitary_gland_latin_hypercube_generator_gives_correct_number_of_samples():
    num_samples = 30
    _, num_samples_out = create_latin_hypercube_sampled_pituitary_df(samples=num_samples)

    assert num_samples == num_samples_out, "Hypercube sampler returned the wrong number of samples!"
