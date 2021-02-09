""" This module contains classes and methods for dimensionality reduction
techniques and visualization API"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import scipy
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csgraph
from sklearn import neighbors
from sklearn.neighbors import radius_neighbors_graph

np.set_printoptions(
    suppress=True, precision=3,
)
style.use("ggplot")


def display_latent_dynamics(latent):

    """Visualize the dynamics in latent space. Compatible only with the RNN latents
    Args:
        latent(tensor): Each point in the list is latent's state at the end
                        of a sequence of each batch.
        Latent shape (batch_size, latent_dim)
    Return: Relative plots of latent unit activations
    Usage:
    ======
    DisplayLatentDynamics(latent)
    """

    latents = {}
    latents.fromkeys(list(range(latent.shape[1])))
    for i in range(latent.shape[1]):
        latents[f"{i}"] = latent[:, i].cpu().detach().numpy()
    fig = px.scatter_matrix(latents)
    fig.update_layout(
        autosize=False, width=1600, height=1000,
    )
    return fig.show()


class DirectedNetwork(object):
    """Wrapper to plot show the activation of Recurrent networks at each time step"""

    def __init__(self):
        pass

    def show(self, states, weight, fig):
        """
        Args:
            states (array): Recurrent network states

            weight (array): Tensor of connection weights

            fig (int): Figure number

        Return:
            Boolean: Figure close status : Open - False/ Close - True
        """
        assert type(states, np.array)
        np.random.seed(70001)
        # Set up hidden states
        state_dict = {i: states[i] for i in range(0, len(states))}
        # Intialize graph
        _g = nx.from_numpy_matrix(
            weight, create_using=nx.MultiDiGraph, parallel_edges=True
        )

        edge_colors = weight.tolist()
        edge_colors_ = [float("%.8f" % j) for i in edge_colors for j in i]

        # Set up nodes
        neuron_color = [state_dict.get(node, 0.25) for node in _g.nodes()]

        # Set colrmap
        vmin = np.min(states)
        vmax = np.max(states)
        cmap = plt.cm.coolwarm
        edge_cmap = plt.cm.Spectral
        nx.draw(
            _g,
            with_labels=True,
            cmap=cmap,
            node_color=neuron_color,
            node_size=200,
            linewidths=5,
            edge_color=edge_colors_,
            edge_cmap=edge_cmap,
            font_size=10,
            connectionstyle="arc3, rad=0.3",
        )

        _sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        _sm.set_array([])

        # State of streaming plot
        if plt.fignum_exists(fig.number):
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.clear()

            # Plot is not closed
            return False
        else:
            return True

class LocalLinearEmbedding(object):
    """Perform Local Linear Embeddings of the input data"""

    def local_linear_embedding(self, _x, _d, _k, alpha=0.1):

        """
        Args:
            _x(numpy.ndarray): Input data matrix mxD , m data points with D dimensions

            _d(int): Target dimensions

            _k(int): Number of neighbors

            alpha(float): Tikhonov coefficient regularization

        Returns:
            y(numpy.ndarray): matrix m row, d attributes are reduced dimensional
        """
        # Find the nearest neighbor
        x_neighbors = neighbors.kneighbors_graph(_x, n_neighbors=_k)

        _m = len(_x)

        # Init weights
        _w = np.zeros(shape=(_m, _m))

        for i, nbor_row in enumerate(x_neighbors):
            # Get the kneighboring indexes of i
            k_indices = nbor_row.indices

            # Calculate the Z matrix
            z_i = _x[k_indices] - _x[i]

            # Calculate the matrix G
            g_i = z_i @ z_i.T

            # Weights between neigbors
            w_i = scipy.linalg.pinv(g_i + alpha * np.eye(_k)) @ np.ones(_k)
            _w[i, k_indices] = w_i / w_i.sum()

        # Calculate matrix M
        _m = (np.eye(_m) - _w).T @ (np.eye(_m) - _w)
        _m = _m.T

        # Calculate Eigen vectors
        _, vectors = scipy.linalg.eigh(_m, eigvals=(0, _d))

        # Return the vectors and discard the first column of the matrix
        return vectors[:, 1:]

    def show(self, p_c, fig2):

        """
        Args:
            p_c (array): First three principle components of the data array
            fig2 (matplotlib.pyplot):  Figure instance
        """

        ax_ = Axes3D(fig2)
        _f = ax_.scatter(p_c[:, 0], p_c[:, 1], p_c[:, 2], s=40, c=p_c[:, 2])
        for i in range(len(p_c)):
            ax_.plot3D(
                p_c[i:, 0],
                p_c[i:, 1],
                p_c[i:, 2],
                alpha=i / len(p_c),
                color="red",
                linewidth=1,
            )
        fig2.colorbar(_f)
        #         plt.pause(0.0001)
        # State of streaming plot
        if plt.fignum_exists(fig2.number):
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            fig2.clear()

            # Plot is not closed
            return False
        else:
            return True


class SpectralEmbedding(object):
    """Spectral Clustering is a non-linear dimensionality reduction technique"""
    def spectral_embedding(self, _x, rad):
        """
        Args:
            _x(numpy.ndarray): Input data matrix mxn , m data points with n dimensions
            rad(float): Radius for neighbor search

        Returns:
            Y(numpy.ndarray): matrix m row, d attributes are reduced dimensional
        """
        # Get the adjacency matrix/nearest neighbor graph; neighbors within the radius of 0.4
        _a = radius_neighbors_graph(
            _x.T,
            rad,
            mode="distance",
            metric="minkowski",
            p=2,
            metric_params=None,
            include_self=False,
        )
        _a = _a.toarray()

        # Find the laplacian of the neighbour graph
        # L = D - A ; where D is the diagonal degree matrix
        _l = csgraph.laplacian(_a, normed=False)
        # Embedd the data points i low dimension using the Eigen values/vectos
        # of the laplacian graph to get the most optimal partition of the graph
        eigval, eigvec = np.linalg.eig(_l)
        # the second smallest eigenvalue represents sparsest cut of the graph.
        np.where(eigval == np.partition(eigval, 1)[1])
        # Partition the graph using the smallest eigen value
        y_spec = eigvec[:, 1].copy()
        y_spec[y_spec < 0] = 0
        y_spec[y_spec > 0] = 1
        return y_spec

    def show(self, _x, spec_embed, fig3):
        """Plot spectral embeddings

        Args:
            _x (array): Data array
            spec_embed (array): Spectral embeddings
            fig3 (matplotlib.pyplot): Figure instance

        Returns:
            plot: Spectral embeddings in 3d plot
        """

        ax3 = fig3.add_subplot()
        _x = _x.T
        f_i = ax3.scatter(x=_x[:, 0], y=_x[:, 1], c=spec_embed, s=30, cmap=plt.cm.Spectral)
        for i in range(len(_x[:, 0])):
            ax3.annotate(i, (_x[:, 0][i], _x[:, 1][i]))
        fig3.colorbar(f_i)

        # State of streaming plot
        if plt.fignum_exists(fig3.number):
            fig3.canvas.draw()
            fig3.canvas.flush_events()
            fig3.clear()

            # Plot is not closed
            return False
        else:
            return True
