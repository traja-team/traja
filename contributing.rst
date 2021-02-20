Contributing to Traja
=====================


Current status
--------------

Traja is currently undergoing active development with approximately
60 % of features present. Significant interface changes are still
possible, however we avoid these unless absolutely necessary.

The work is currently focussed on reaching version 1.0 with feature
completeness and 95 % test coverage.

The following features are required for feature completeness:

* Latent space visualisers
   * Eigenspace-based
   * Colour-coded to visualise evolution over time
   * Delay coordinate embeddings
* State-space visualisers
* Additional encoder and decoder options in AE and VAE models
   * MLP
   * 1d convolution
* Pituitary gland example dataset
* Regression output visualisers
* VAE GAN models
* Additional VAE latent-space shapes
   * Uniform
   * A shape that works for periodic trajectories (Torus?)
* Delay coordinate embeddings
   * Persistent homology diagrams of the embeddings
* Automatic code formatter
* Tutorials
   * Find time of day based on activity
   * Recover parameters from Pituitary ODE

How to contribute
-----------------

Traja welcomes contributions! To get started, pick up any issue
labeled with `good first issue`! Alternatively you can read some
background material or try a tutorial.

Background material
-------------------

This is a collection of papers and resources that explain the
main problems we are working on with Traja.

Analysis of mice that have suffered a stroke:

    @article{10.3389/fnins.2020.00518,
      author={Justin Shenk and
              Klara J. Lohkamp and
              Maximilian Wiesmann and
              Amanda J. Kiliaan},
      title={Automated Analysis of Stroke Mouse Trajectory Data With Traja},
      journal={Frontiers in Neuroscience},
      volume={14},
      pages={518},
      year={2020},
      url={https://www.frontiersin.org/article/10.3389/fnins.2020.00518},
      doi={10.3389/fnins.2020.00518},
      issn={1662-453X},
    }


Understanding the parameter space of the pituitary gland ODE (https://www.math.fsu.edu/~bertram/papers/bursting/JCNS_16.pdf):


    @article{10.1007/s10827-016-0600-1,
      author = {Fletcher, Patrick and Bertram, Richard and Tabak, Joel},
      title = {From Global to Local: Exploring the Relationship between Parameters and Behaviors in Models of Electrical Excitability},
      year = {2016},
      publisher = {Springer-Verlag},
      address = {Berlin, Heidelberg},
      volume = {40},
      number = {3},
      issn = {0929-5313},
      url = {https://doi.org/10.1007/s10827-016-0600-1},
      doi = {10.1007/s10827-016-0600-1},
      journal = {J. Comput. Neurosci.},
      month = June,
      pages = {331–345},
    }


Style guide
-----------
TODO