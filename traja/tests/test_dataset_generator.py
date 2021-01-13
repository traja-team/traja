import pandas as pd

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

    for data, target, category, parameters in dataloaders['train_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == -1.
        for sequence in target:
            for sample in sequence:
                assert sample[0] == -1.

    for data, target, category, parameters in dataloaders['test_loader']:
        for sequence in data:
            for sample in sequence:
                assert sample[0] == 0
        for sequence in target:
            for sample in sequence:
                assert sample[0] == 0

    for data, target, category, parameters in dataloaders['validation_loader']:
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

    for sample_id in range(num_ids):
        for sequence in range(40 + int(sample_id / 14)):
            data.append([sequence, sequence, sample_id])

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
                                               validation_split_ratio=validation_split_ratio)
    verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids)


def test_id_wise_sampling_does_not_put_id_in_multiple_dataloaders():
    data = list()
    num_ids = 150

    for sample_id in range(num_ids):
        for sequence in range(40):
            data.append([sequence, sequence, sample_id])

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
                                               validation_split_ratio=validation_split_ratio)

    verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids)


def verify_id_wise_sampled_dataloaders_do_not_overlap(dataloaders, train_split_ratio, validation_split_ratio, num_ids):
    train_categories = []
    for data, target, categories, parameters in dataloaders['train_loader']:
        for category in categories:
            if category not in train_categories:
                train_categories.append(category)

    assert len(train_categories) == round(train_split_ratio * num_ids), 'Wrong number of training categories!'

    test_categories = []
    for data, target, categories, parameters in dataloaders['test_loader']:
        for category in categories:
            if category not in test_categories:
                test_categories.append(category)

        assert category not in train_categories, 'Found test data in train loader!'

    validation_categories = []
    for data, target, categories, parameters in dataloaders['validation_loader']:
        for category in categories:
            if category not in validation_categories:
                validation_categories.append(category)

        assert category not in train_categories, 'Found validation data in train loader!'
        assert category not in test_categories, 'Found validation data in test loader!'

    assert len(validation_categories) == round(
        validation_split_ratio * num_ids), 'Wrong number of validation categories!'
    assert len(train_categories) + len(test_categories) + len(
        validation_categories) == num_ids, 'Wrong number of categories!'
