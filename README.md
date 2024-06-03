# ClustAssessPy

ClustAssessPy is a lighter Python adaptation of ClustAssess (R). This Python version includes ClustAssess's main functions, such as calculating all ECS-related metrics and evaluating and plotting clustering stability in the dimensionality reduction, graph building, and graph clustering components.

The package allows for a data-driven assessment of optimal parameter values for dimensionality reduction (e.g. choosing between UMAP and PCA), graph type (e.g., shared-nearest-neighbors vs nearest neighbors), identification of the most stable community detection algorithm (e.g., leiden vs louvain), resolution that produces the most stable partitions.
