import warnings

import matplotlib
import numpy as np
import numpy.testing as npt

from traja.dataset import dataset
from traja.dataset.example import jaguar
from traja.models.generative_models.vae import MultiModelVAE
from traja.models.train import HybridTrainer
from traja.plotting import plot_prediction

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import traja

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

df = traja.generate(n=10)


def test_stylize_axes():
    collection = traja.plot(df, interactive=False)
    traja.plotting.stylize_axes(collection.axes)


def test_color_dark():
    df = traja.generate(n=10)
    index = pd.DatetimeIndex(range(10))
    df.index = index
    ax = plt.gca()
    try:
        traja.color_dark(df.x, ax)
    except ValueError as e:
        # catch unexplained datetime value error in travis
        if "view limit minimum" in str(e):
            pass


def test_sans_serif():
    traja.plotting.sans_serif()


def test_plot_3d():
    traja.plot_3d(df, interactive=False)


def test_plot_flow():
    traja.plot_flow(df, interactive=False)


def test_plot_contour():
    ax = traja.plot_contour(df, interactive=False)


def test_plot_surface():
    ax = traja.plot_surface(df, interactive=False)


def test_plot_stream():
    ax = traja.plot_stream(df, interactive=False)


def test_trip_grid():
    traja.plotting.trip_grid(df, interactive=False)


def test_label_axes():
    df.traja.plot(interactive=False)
    ax = plt.gca()
    traja.plotting._label_axes(df, ax)


def test_plot_actogram():
    df = traja.generate(n=1000)
    index = pd.date_range("2018-01-01", periods=1000, freq="T")
    df.index = index
    activity = traja.calc_displacement(df)
    activity.name = "activity"
    traja.plotting.plot_actogram(df.x, interactive=False)


def test_plot_xy():
    traja.plotting.plot_xy(df, interactive=False)


def test_polar_bar():
    traja.plotting.polar_bar(df, interactive=False)


def test_find_runs():
    actual = traja.find_runs(df.x)
    expected = (
        np.array(
            [
                0.0,
                1.323_370_69,
                2.275_837_54,
                2.274_285_61,
                0.336_029_88,
                -1.455_690_92,
                -3.544_442_11,
                -5.386_597_93,
                -7.508_544_4,
                -9.353_255_17,
            ]
        ),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    for i in range(len(actual)):
        npt.assert_allclose(actual[i], expected[i])


def test_plot_clustermap():
    trjs = [traja.generate(seed=i) for i in range(20)]

    # Calculate displacement
    displacements = [trj.traja.calc_displacement() for trj in trjs]

    traja.plot_clustermap(displacements)


def test_plot():
    ax = traja.plotting.plot(df, interactive=False)
    path = ax.get_paths()[0]
    npt.assert_allclose(
        path._vertices[:5],
        np.array(
            [
                [0.0, -0.5],
                [0.132_601_55, -0.5],
                [0.259_789_94, -0.447_316_85],
                [0.353_553_39, -0.353_553_39],
                [0.447_316_85, -0.259_789_94],
            ]
        ),
    )


def test_plot_prediction():
    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 10

    input_size = 2
    lstm_hidden_size = 512
    lstm_num_layers = 4
    batch_first = True
    reset_state = True
    output_size = 2
    num_classes = 9
    latent_size = 20
    dropout = 0.1
    bidirectional = False

    # Prepare the dataloader
    df = jaguar()
    data_loaders = dataset.MultiModalDataLoader(
        df, batch_size=batch_size, n_past=num_past, n_future=num_future, num_workers=1
    )

    model = MultiModelVAE(
        input_size=input_size,
        output_size=output_size,
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=lstm_num_layers,
        num_classes=num_classes,
        latent_size=latent_size,
        dropout=dropout,
        num_classifier_layers=4,
        classifier_hidden_size=32,
        batch_size=batch_size,
        num_future=num_future,
        num_past=num_past,
        bidirectional=bidirectional,
        batch_first=batch_first,
        reset_state=reset_state,
    )

    trainer = HybridTrainer(model=model, optimizer_type="Adam", loss_type="huber")

    model_save_path = "./model.pt"

    plot_prediction(model, data_loaders["sequential_test_loader"], 1)
