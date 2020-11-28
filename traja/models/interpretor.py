"""Model interpretion and Visualization"""
import plotly.express as px

from .ae import MultiModelAE
from .vae import MultiModelVAE
from .vaegan import MultiModelVAEGAN
from .irl import MultiModelIRL

def DisplayLatentDynamics(latent):
    r"""Visualize the dynamics of combination of latents 
    Args:
    latent(tensor): Each point in the list is latent's state at the end of a sequence of each batch.
    Latent shape (batch_size, latent_dim)
    Usage:
    DisplayLatentDynamics(latent)"""
    
    latents = {}
    latents.fromkeys(list(range(latent.shape[1]))) 
    for i in range(latent.shape[1]):
        latents[f'{i}']=latent[:,i].cpu().detach().numpy()
    fig= px.scatter_matrix(latents)
    fig.update_layout(
    autosize=False,
    width=1600,
    height=1000,)
    return fig.show()