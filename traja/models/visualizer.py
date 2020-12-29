import os, sys
from matplotlib.pyplot import figimage
import networkx as nx
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn import neighbors
from scipy.sparse import csgraph
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse, copy, h5py, os, sys, time, socket
import tensorflow as tf
import torch, torchvision, torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import ticker, colors
import plotly.express as px


import matplotlib

# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib import style

# plt.switch_backend("TkAgg")


np.set_printoptions(
    suppress=True, precision=3,
)
style.use("ggplot")


def DisplayLatentDynamics(latent):
    """Visualize the dynamics of combination of latents 
    Args:
        latent(tensor): Each point in the list is latent's state at the end of a sequence of each batch.
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
    def __init__(self):
        super().__init__()
        pass

    def show(self, states, weight, fig):
        """

        :param states: list - Hidden states
        :param weight: numpy.ndarray - Array of connection weights
        :param fig: Figure number

        :return: boolean: Figure close status : Open - False/ Close - True

        """
        np.random.seed(70001)
        # Set up hidden states
        state_dict = {i: states[i] for i in range(0, len(states))}

        # Set up links
        self_connections = [weight[i][i] for i in range(len(weight))]

        # Intialize graph
        G = nx.from_numpy_matrix(
            weight, create_using=nx.MultiDiGraph, parallel_edges=True
        )

        edge_colors = weight.tolist()
        edge_colors_ = [float("%.8f" % j) for i in edge_colors for j in i]

        # Set up nodes
        neuron_color = [state_dict.get(node, 0.25) for node in G.nodes()]

        # Set colrmap
        vmin = np.min(states)
        vmax = np.max(states)
        cmap = plt.cm.coolwarm
        edge_cmap = plt.cm.Spectral
        nx.draw(
            G,
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

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, orientation="vertical", pad=0.1)

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
    def __init__(self):
        super(LocalLinearEmbedding, self).__init__()
        pass

    def local_linear_embedding(self, X, d, k, alpha=0.1):
        """
            Local Linear Embeddings

            :param X: numpy.ndarray - input data matrix mxD , m data points with D dimensions
            :param d: int - target dimensions
            :param k: int -number of neighbors
            :param alpha: float - Tikhonov coefficient regularization

            :return Y: numpy.ndarray - matrix m row, d attributes are reduced dimensional
            """
        # Find the nearest neighbor
        x_neighbors = neighbors.kneighbors_graph(X, n_neighbors=k)

        m = len(X)

        # Init weights
        W = np.zeros(shape=(m, m))

        for i, nbor_row in enumerate(x_neighbors):
            # Get the kneighboring indexes of i
            k_indices = nbor_row.indices

            # Calculate the Z matrix
            Z_i = X[k_indices] - X[i]

            # Calculate the matrix G
            G_i = Z_i @ Z_i.T

            # Weights between neigbors
            w_i = scipy.linalg.pinv(G_i + alpha * np.eye(k)) @ np.ones(k)
            W[i, k_indices] = w_i / w_i.sum()

        # Calculate matrix M
        M = (np.eye(m) - W).T @ (np.eye(m) - W)
        M = M.T

        # Calculate Eigen vectors
        _, vectors = scipy.linalg.eigh(M, eigvals=(0, d))

        # Return the vectors and discard the first column of the matrix
        return vectors[:, 1:]

    def show(self, pc, fig2):
        """[summary]

        Args:
            pc ([type]): [description]
            fig2 ([type]): [description]
        """

        ax = Axes3D(fig2)
        f = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=40, c=pc[:, 2])
        for i in range(len(pc)):
            ax.plot3D(
                pc[i:, 0],
                pc[i:, 1],
                pc[i:, 2],
                alpha=i / len(pc),
                color="red",
                linewidth=1,
            )
        fig2.colorbar(f)
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
    def __init__(self):
        super(SpectralEmbedding, self).__init__()
        pass

    def spectral_embedding(self, X, rad):
        """
            Spectral Clustering

            :param X: numpy.ndarray - input data matrix mxn , m data points with n dimensions
            :param rad: float -radius for neighbor search

            :return Y: numpy.ndarray - matrix m row, d attributes are reduced dimensional
            """
        # Get the adjacency matrix/nearest neighbor graph; neighbors within the radius of 0.4
        A = radius_neighbors_graph(
            X.T,
            rad,
            mode="distance",
            metric="minkowski",
            p=2,
            metric_params=None,
            include_self=False,
        )
        A = A.toarray()

        # Find the laplacian of the neighbour graph
        # L = D - A ; where D is the diagonal degree matrix
        L = csgraph.laplacian(A, normed=False)
        # Embedd the data points i low dimension using the Eigen values/vectos
        # of the laplacian graph to get the most optimal partition of the graph
        eigval, eigvec = np.linalg.eig(L)
        # the second smallest eigenvalue represents sparsest cut of the graph.
        np.where(eigval == np.partition(eigval, 1)[1])
        # Partition the graph using the smallest eigen value
        y_spec = eigvec[:, 1].copy()
        y_spec[y_spec < 0] = 0
        y_spec[y_spec > 0] = 1
        return y_spec

    def show(self, X, spec_embed, fig3):
        """[summary]

        Args:
            X ([type]): [description]
            spec_embed ([type]): [description]
            fig3 ([type]): [description]

        Returns:
            [type]: [description]
        """

        ax3 = fig3.add_subplot()
        X = X.T
        fi = ax3.scatter(x=X[:, 0], y=X[:, 1], c=spec_embed, s=30, cmap=plt.cm.Spectral)
        for i in range(len(X[:, 0])):
            ax3.annotate(i, (X[:, 0][i], X[:, 1][i]))
        fig3.colorbar(fi)

        # State of streaming plot
        if plt.fignum_exists(fig3.number):
            fig3.canvas.draw()
            fig3.canvas.flush_events()
            fig3.clear()

            # Plot is not closed
            return False
        else:
            return True


if __name__ == "__main__":
    # create the coordinates
    numebr_of_points = 21
    small_range = -1.0
    large_range = 1.0

    xcoordinates = np.linspace(small_range, large_range, num=numebr_of_points)
    ycoordinates = np.linspace(small_range, large_range, num=numebr_of_points)

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    inds = np.array(range(numebr_of_points ** 2))
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]
    coordinate = np.c_[s1, s2]
    print(
        "From ",
        small_range,
        " to ",
        large_range,
        " with ",
        numebr_of_points,
        " total number of coordinate: ",
        numebr_of_points ** 2,
    )


class Network:
    def __init__(self, activity, weights):

        pass

    def show(self):
        fig = None
        return fig


class ShowManifold:
    def __init__(self, inputs, manifold):
        pass

    def show(self):
        fig = None
        return fig

