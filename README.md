# ClustAssessPy

[![Downloads](https://static.pepy.tech/badge/ClustAssessPy)](https://pepy.tech/project/ClustAssessPy)
[![Downloads](https://static.pepy.tech/badge/ClustAssessPy/week)](https://pepy.tech/project/ClustAssessPy)

ClustAssessPy offers a data-driven approach for optimizing parameter values across all stages of graph-based community detection clustering in single-cell datasets. This Python package is a lighter adaptation of ClustAssess (R) [1], incorporating its main functions and can be used by the Scanpy community to guide robust clustering through data-driven selection in all community detection clustering steps:

- **Dimensionality Reduction:** Selection of the base embedding (UMAP vs PCA) and the number and type of features (e.g., highly-variable vs most abundant).
- **Graph Type:** Choice of graph type for the adjacency matrix (nearest neighbors vs shared nearest neighbors) and the number of neighbors.
- **Clustering:** Identification of the most stable algorithm (Leiden or Louvain) and the appropriate resolution value.

## Installation

ClustAssessPy requires Python 3.7 or newer.

### Dependencies

- numpy
- pandas
- scanpy
- umap-learn
- seaborn
- matplotlib
- scipy
- networkx
- plotnine
- pynndescent
- leidenalg
- louvain
- igraph

### User Installation

We recommend that you download ClustAssessPy on a virtual environment (venv or Conda).

```sh
pip install ClustAssessPy
```

## Getting Started

Documentation for the main functions is available [here](https://core-bioinformatics.github.io/ClustAssessPy/ClustAssessPy.html). For a detailed tutorial, click [here](https://github.com/Core-Bioinformatics/ClustAssessPy/blob/main/Examples/tutorial_cuomo.ipynb).

## References

[1] Shahsavari, A., Munteanu, A., & Mohorianu, I. (2022). ClustAssess: Tools for Assessing the Robustness of Single-Cell Clustering. https://doi.org/10.1101/2022.01.31.478592
