import pandas as pd

from traja.dataset import dataset


def test_category_wise_sampling_few_categories():
    data = list()
    num_categories = 5

    for category in range(num_categories):
        for sequence in range(40 + int(category / 14)):
            data.append([sequence, sequence, category])

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    # Hyperparameters
    batch_size = 1
    num_past = 10
    num_future = 5
    train_split_ratio = 0.5
    validation_split_ratio = 0.2

    dataloaders, scalers = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio)
    verify_category_wise_sampled_dataloaders(dataloaders, train_split_ratio, validation_split_ratio, num_categories)


def test_category_wise_sampling():
    data = list()
    num_categories = 150

    for category in range(num_categories):
        for sequence in range(40):
            data.append([sequence, sequence, category])

    df = pd.DataFrame(data, columns=['x', 'y', 'ID'])

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    train_split_ratio = 0.333
    validation_split_ratio = 0.333

    dataloaders, scalers = dataset.MultiModalDataLoader(df,
                                               batch_size=batch_size,
                                               n_past=num_past,
                                               n_future=num_future,
                                               num_workers=1,
                                               train_split_ratio=train_split_ratio,
                                               validation_split_ratio=validation_split_ratio)

    verify_category_wise_sampled_dataloaders(dataloaders, train_split_ratio, validation_split_ratio, num_categories)


def verify_category_wise_sampled_dataloaders(dataloaders, train_split_ratio, validation_split_ratio, num_categories):
    train_categories = []
    for data, target, categories, parameters in dataloaders['train_loader']:
        for category in categories:
            if category not in train_categories:
                train_categories.append(category)

    assert len(train_categories) == round(train_split_ratio * num_categories), 'Wrong number of training categories!'

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
        validation_split_ratio * num_categories), 'Wrong number of validation categories!'
    assert len(train_categories) + len(test_categories) + len(
        validation_categories) == num_categories, 'Wrong number of categories!'
