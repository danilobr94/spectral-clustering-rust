# Spectral Clustering for Rust
### Efficient Linear Algebra and Machine Learning for CL SS2018
#### Tuebingen University
#### Danilo Brajovic and Tristan Baumann
## Description
The crate provides an implementation of spectral clustering in rust. Spectral clustering is an unsupervised
learning algorithm able to cluster nonlinear data. The general steps are: compute the graph representation
of the data as adjacency matrix, transform it into an graph Laplacian and cluster the largest eigenvectors
with K-Means. We provide a function to cluster the data using a 2D ndarray of the raw data (automatically
building the knn-Graph) or functions to manually build a graph and functions to cluster data from a graph
representation, giving the user the flexibility to pass their own graph. Two different graph types can currently
be build using implemented functions: a knn graph and an epsilon graph. Additionally, it is possible to select
the distance function for graph building (currently only norms are supported L1, L2).

## Usage
The crate is designed to work with the ndarray-crate for matrix representation. The easiest way to use it is
to call the method

````
spectral_clustering::cluster(x, k_knn, n_clusters, normalized)
````

with a dataset of row-wise point dimension representation x and a desired k_knn for the knn-graph creation that produces the needed
weight matrix. The number of desired clusters can also be specified, but the implementation will attempt
to find a good number of clusters by itself if the value n_clusters is set to a negative number. The boolean variable normalized
specifies whether the Laplace matrix is normalized or not, which may change the clustering outcome.
If the data is already in the form of a weight/adjacency matrix, spectral clustering can also be performed directly by calling

````
spectral-clustering::cluster-from-graph()
````

It is also possible to separately create the weight matrix either by nearest neighbors

````
knn::build(*)
````

a distance threshold

````
epsilon::build(*)
````

or both 
````
knn::build_knn_with_epsilon(*)
````

Finally, the crate offers the possibility of non-euclidean data distances by specifying a p-norm.

The project additionally includes a method for reading files and a method for plotting the results of the
clustering; These are only part of the project to provide a working example (see main method). They should
not be thought of as parts of the crate.
Further method usage is detailed in each method’s description in the code.

## Further Steps
Currently, we only support norms for graph building. As further improvement, a custom distance function could be passed.
The eigendecomposition and K-Means use two different crates. This is because the decomposition of rusty-
machine fails to compute the eigenvalues. Our graphs are spare, hence an own eigendecomposition might be
beneficial in terms of runtime. Alternatively, switching the crate when a new one is available might be enough.
