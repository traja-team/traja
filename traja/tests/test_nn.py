import pandas as pd
import traja
from traja import models
from traja import datasets
from traja.datasets import dataset
from traja.models.train import LSTMTrainer, HybridTrainer, CustomTrainer



def test_from_df():
    data_url = "https://raw.githubusercontent.com/traja-team/traja-research/dataset_und_notebooks/dataset_analysis/jaguar5.csv"
    df = pd.read_csv(data_url, error_bad_lines=False)
    model_save_path = 'model'

    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5

    # Prepare the dataloader
    train_loader, test_loader = dataset.MultiModalDataLoader(df,
                                                     batch_size=batch_size,
                                                     n_past=num_past,
                                                     n_future=num_future,
                                                     num_workers=2)
    
    trainer = HybridTrainer(model_type='vae',  # "ae" or "vae"
                      optimizer_type='Adam',   # ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'RMSprop', 'Rprop','LBFGS', 'ASGD', 'Adamax']
                      input_size=2,  
                      output_size=2, 
                      lstm_hidden_size=32, 
                      num_lstm_layers=2,
                      reset_state=True, 
                      latent_size=10, 
                      dropout=0.1, 
                      num_classes=9,  # Uncomment to create and train classifier network
                      num_classifier_layers=4,
                      classifier_hidden_size= 32, 
                      batch_size=batch_size, 
                      num_future=num_future, 
                      num_past=num_past,
                      bidirectional=False, 
                      batch_first=True,
                      loss_type='huber') # 'rmse' or 'huber'


    # Train the model
    trainer.fit(train_loader, test_loader, model_save_path, epochs=10, training_mode='forecasting')
    trainer.fit(train_loader, test_loader, model_save_path, epochs=10, training_mode='classification')


if __name__ == '__main__':
    test_from_df()
