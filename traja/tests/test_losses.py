import torch

from traja.models.losses import Criterion


def test_forecasting_loss_yields_correct_value():
    criterion = Criterion()

    predicted = torch.ones((1, 8))
    target = torch.zeros((1, 8))

    manhattan_loss = criterion.forecasting_criterion(
        predicted, target, loss_type="manhattan"
    )  # 8
    huber_low_loss = criterion.forecasting_criterion(
        predicted * 0.5, target, loss_type="huber"
    )  # ~1
    huber_high_loss = criterion.forecasting_criterion(
        predicted * 2, target, loss_type="huber"
    )  # ~12
    mse_low_loss = criterion.forecasting_criterion(
        predicted * 0.5, target, loss_type="mse"
    )  # 0.25
    mse_high_loss = criterion.forecasting_criterion(
        predicted * 2, target, loss_type="mse"
    )  # 4

    assert manhattan_loss == 8
    assert huber_low_loss == 1
    assert huber_high_loss == 12
    assert mse_low_loss == 0.25
    assert mse_high_loss == 4
