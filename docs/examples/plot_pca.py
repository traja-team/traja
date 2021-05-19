"""
Plot PCA with traja
-----------------------------------
Plot PCA of a trip grid with :meth:`traja.plotting.plot_pca`

Wrapper for pandas :meth:`pandas.plotting.autocorrelation_plot`.

"""
import traja

# Load sample jaguar dataset with trajectories for 9 animals
df = traja.dataset.example.jaguar()

# Bin trajectory into a trip grid then perform PCA
traja.plotting.plot_pca(df, id_col="ID", bins=(8,8))   