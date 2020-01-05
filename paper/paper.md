---
title: 'Traja: A Python toolbox for animal trajectory analysis'
tags:
  - Python
  - animal
  - trajectory
  - prediction
authors:
  - name: Justin Shenk
    orcid: 0000-0002-0664-7337
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Donders Institute for Brain, Cognition and Behavior, Radboud University Nijmegen
   index: 1
   
date: 5 January 2020
bibliography: paper.bib
---

# Summary

Animal tracking is important for fields as diverse as ethology, optimal 
foraging theory, and neuroscience. In recent years, advances in machine
learning have led to breakthroughs in pattern recognition and data modeling.
A tool that support modeling in the language of state-of-the-art predictive
models [@socialways; @next; @TraPHic] and which provides researchers with a high-level
API for feature extraction, modeling and visualization is needed.

``Traja`` is a Python package for trajectory analysis. ``Traja`` extends
the familiar pandas [@pandas] methods by providing a pandas accessor 
to the ``df.traja`` namespace upon import. 
The API for ``Traja`` was designed to provide a object-oriented and
user-friendly interface to common methods in analysis and visualization
of animal trajectories.
``Traja`` also interfaces well with relevant spatial analysis packages in R
(e.g., ``trajr`` [@trajr], ``adehabitat`` [@adehabitat]) and ``Shapely`` [@shapely], allowing rapid prototyping
and comparison of relevant methods in Python.

![Example plot of generated random walk](figure.png)

# References