import pandas as pd
from models.train import Trainer
from datasets.dataset import MultiModalDataLoader

# Downlaod dataset (9 jaguars) from github
data_url = "https://raw.githubusercontent.com/traja-team/traja-research/dataset_und_notebooks/dataset_analysis/jaguar5.csv" 
df = pd.read_csv(data_url, error_bad_lines=False)

# Hyperparameters
batch_size = 10
num_past = 10
num_future = 5
model_save_path = './model.pt'

# Prepare the dataloader
train_loader, test_loader = MultiModalDataLoader(df,
                                                 batch_size=batch_size, 
                                                 n_past=num_past, 
                                                 n_future=num_future, 
                                                 num_workers=1)

# Initialize the models
trainer = Trainer(model_type='vae', 
                  device='cpu', 
                  input_size=2, 
                  output_size=2, 
                  lstm_hidden_size=512,
                  lstm_num_layers=4,
                  reset_state=True, 
                  num_classes=9, 
                  latent_size=10, 
                  dropout=0.1, 
                  num_layers=4, 
                  epochs=10, 
                  batch_size=batch_size, 
                  num_future=num_future, 
                  sequence_length=num_past,
                  bidirectional=False, 
                  batch_first=True,
                  loss_type='huber')

# Train the model
trainer.train_latent_model(train_loader, test_loader, model_save_path)

