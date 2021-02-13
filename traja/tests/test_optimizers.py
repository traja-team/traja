from traja.models.optimizers import Optimizer
from traja.models.predictive_models.ae import MultiModelAE


def test_get_optimizers():
    # Test
    model_type = "custom"
    model = MultiModelAE(
        input_size=2,
        num_past=10,
        batch_size=5,
        num_future=5,
        lstm_hidden_size=32,
        num_lstm_layers=2,
        output_size=2,
        latent_size=10,
        batch_first=True,
        dropout=0.2,
        reset_state=True,
        bidirectional=True,
        num_classifier_layers=4,
        classifier_hidden_size=32,
        num_classes=10,
        num_regressor_layers=2,
        regressor_hidden_size=32,
        num_regressor_parameters=3,
    )

    # Get the optimizers
    opt = Optimizer(model_type, model, optimizer_type="RMSprop")
    model_optimizers = opt.get_optimizers(lr=0.1)
    model_schedulers = opt.get_lrschedulers(factor=0.1, patience=10)

    print(model_optimizers, model_schedulers)
