Clustering and Dimensionality Reduction
=======================================

Clustering Trajectories
-----------------------

Trajectories can be clustered using :func:`traja.plotting.plot_clustermap`.

Colors corresponding to each trajectory can be specified with the ``colors`` argument.

.. autofunction:: traja.plotting.plot_clustermap

PCA
---

Prinicipal component analysis can be used to cluster trajectories based on grid cell occupancy. 
PCA is computed by converting the trajectory to a trip grid (see :meth:`traja.plotting.trip_grid`) followed by PCA (:class:`sklearn.decomposition.PCA`).

.. autofunction:: traja.plotting.plot_pca