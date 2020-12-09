"""Generate time series from model"""

import plotly.express as px
import torch
from .ae import MultiModelAE
from .vae import MultiModelVAE
from .vaegan import MultiModelVAEGAN
from .irl import MultiModelIRL
from .utils import load_model
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def timeseries(model_type:str, model_hyperparameters:dict, model_path:str, batch_size:int, num_future:int, ):
    # Generating few samples
    batch_size = model_hyperparameters.batch_size # Number of samples
    num_future = model_hyperparameters.num_future # Number of time steps in each sample
    if model_type == 'ae':
        model = MultiModelAE(**model_hyperparameters)
        
    if model_type == 'vae':
        model = MultiModelVAE(**model_hyperparameters)
        
    if model_type == 'vaegan':
        model = MultiModelVAEGAN(**model_hyperparameters)
        return NotImplementedError
    
    if model_type == 'irl':
        model = MultiModelIRL(**model_hyperparameters)
        return NotImplementedError
    
    # Load the model from model path:
    model = load_model(model, model_hyperparameters, model_path)
    # z = torch.randn((batch_size, latent_size)).to(device)
    z = torch.empty(10, model_hyperparameters.latent_size).normal_(mean=0,std=.1).to(device)
    # Category from the noise
    cat = model.classifier(z)
    # Generate trajectories from the noise
    out = model.decoder(z,num_future).cpu().detach().numpy()
    out = out.reshape(out.shape[0]*out.shape[1],out.shape[2])

    # for index, i in enumerate(train_df.columns):
    #     scaler = scalers['scaler_'+i]  
    #     out[:,index] = scaler.inverse_transform(out[:,index].reshape(1, -1))
    print('IDs in this batch of synthetic data',torch.max(cat,1).indices+1)
    plt.figure(figsize=(12,4))
    plt.plot(out[:,0], label='Generated x: Longitude')
    plt.plot(out[:,1], label='Generated y: Latitude')
    plt.legend()

    fig, ax = plt.subplots(nrows=2, ncols= 5, figsize=(16, 5), sharey=True)
    # plt.ticklabel_format(useOffset=False)
    fig.set_size_inches(20,5)
    for i in range(2):
        for j in range(5):
            ax[i,j].plot(out[:,0][(i+j)*num_future:(i+j)*num_future + num_future],out[:,1][(i+j)*num_future:(i+j)*num_future+ num_future],label = 'Animal ID {}'.format((torch.max(cat,1).indices+1).detach()[i+j]),color='g')
            ax[i,j].legend()
    plt.show()
    
    return out
 

