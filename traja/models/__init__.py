try:
    from traja.models.generative_models.vae import MultiModelVAE
    from traja.models.generative_models.vaegan import MultiModelVAEGAN
    from traja.models.predictive_models.ae import MultiModelAE
    from traja.models.predictive_models.lstm import LSTM

    from .inference import *
    from .train import HybridTrainer
    from .utils import TimeDistributed, load, read_hyperparameters, save
except ImportError:
    # torch not available, models won't be available
    import warnings
    warnings.warn(
        "PyTorch not installed. Deep learning models not available. "
        "Install with: pip install torch or pip install traja[dl]"
    )
