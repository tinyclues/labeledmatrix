# Labeled Matrix
## Why Labeled Matrix ?
`labeledmatrix` introduces routines to work with 2-dimensional data using user-defined labels as coordinates.
In particular all matrix operations will automatically align row and column labels for the calculations.

Another distinguishing feature of `labeledmatrix` is the support of sparse matrices and effective implementation of
variety of operations on them. 

## What are available operations and use-cases ?
Initially, this library is implemented for Recommender Systems use-case, where a matrix represents users rating
or purchase history (with user ids used as row labels and item ids used as column labels). But it may be directly
applied to problems with 2-dimensional data:
* Graph adjacency matrix
* Pairwise similarity matrix
* TODO other examples

In particular, there are algorithms for various use cases, such as
* similarity measures
* clustering
* matrix factorization

See full list in [algorithms.md](documentation/algorithms.md)

## Is it efficient ?
For sparse input matrices we use efficient implementations of algorithms written in Cython and for some operations
their implementations are much faster than the ones in standard libraries, like scipy or pandas
(due to the usage of OpenMP parallelism).  
TODO add a link to benchmarks

## Get started
### Installation
TODO
### First examples
TODO

## Repository structure
### cyperf sub-package :
Parallelized (with OpenMP) version of standard CSR and CSC sparse matrix format with some additional useful methods.

### labeledmatrix sub-package
Library for matrix operations using dense (`numpy.ndarray`) and sparse (`cyperf.KarmaSparse`) backends.
