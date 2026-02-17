# MAGIC-FB

This repository contains the code for the paper "Multiresolution Adaptive Block-Coordinate Forward-Backward for Image Reconstruction". The paper introduces MAGIC-FB, an algorithm that exploits a multiresolution decomposition of the image to perform block-coordinate updates where block selection is adaptive, with the goal of matching the fastest known method in every degradation regime. The method is evaluated on deblurring problems with various levels of degradation.

## Algorithmic framework

The repository includes a general framework for block-coordinate forward-backward algorithms on wavelet blocks. As of now, the degradation model $A$ is limited to convolutional operators that can be implemented as Kronecker products of 1D convolutions.

The minimization problem solved by the BC-FB algorithms is of the form

$$
\min_{\mathbf{w} \in \mathbb{R}^n} f(\mathbf{w}) + g(\mathbf{w}),
$$

where $\mathbf{w} = Wx$ are the wavelet coefficients of the image $x$ to be reconstructed, $f$ is a data fidelity term, and $g$ is a regularization term that promotes certain properties of the wavelet coefficients. In the experiments of the paper, the data fidelity term is the squared $\ell_2$ norm $\frac{1}{2}\|AW^\top \mathbf{w} - \mathbf{y}\|^2$, and the regularization term is the $\ell_1$ norm applied to the detail coefficients of the wavelet decomposition.

The regularization term is assumed to be separable across the wavelet coefficients, *i.e* that

$$
\forall \mathbf{w} \in \mathbb{R}^{n}, \quad g(\mathbf{w}) = \sum_{i=0}^{J} g_i(w_i).
$$

## Repository structure

The repository is organized as follows:

- `minimal.ipynb`: A minimal example of how to use the BC-FB framework for a deblurring problem.
- `block.py`: Contains the implementation of the BlockCoordinateDescent class, which implements the block-coordinate forward-backward algorithm and encapsulates different block selections.
- `utils`: Contains utility functions for creating matricial representations of physics operators, wavelet transforms, and plotting results.

## Acknowledgements

This repository uses the following libraries:
- [DeepInverse](https://deepinv.github.io/deepinv/): a Torch-based library for inverse problems.
- [LazyLinop](https://faustgrp.gitlabpages.inria.fr/lazylinop/): a toolbox to accelerate computations with linear operators.