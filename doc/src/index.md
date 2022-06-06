```@meta
CurrentModule = Flou
```

# Documentation

```@contents
```

## Motivation

Flou is a solver for hyperbolic partial differential equations, using a *Discontinuous Galerkin Spectral Element Method* (DGSEM) approach.

## Roadmap

- Add test cases
- Add orientation of faces
- Add periodic boundary conditions
- Add elliptic equations
- Add multi-process support

## Hyperbolic equations

### Mathematical background

Weak form in 2D for tensor-product elements:
```math
M \dot{\boldsymbol{Q}}_{ij} = \sum_k^{N_x} K^{(j)}_{x,ik} \tilde{\boldsymbol{f}}_{kj} + \sum_k^{N_y} K^{(i)}_{y,jk} \tilde{\boldsymbol{g}}_{ik} - \sum_d^2 L_{\omega_x,ij}^{(d)} \boldsymbol{f}_{n,i}^\star - \sum_d^2 L_{\omega_y,ij}^{(d)} \boldsymbol{g}_{n,j}^\star
```

Strong form in 2D for tensor-product elements:
```math
M \dot{\boldsymbol{Q}}_{ij} = \sum_k^{N_x} \hat{K}^{(j)}_{x,ik} \tilde{\boldsymbol{f}}_{kj} + \sum_k^{N_y} \hat{K}^{(i)}_{y,jk} \tilde{\boldsymbol{g}}_{ik} - \sum_d^2 L_{\omega_x,j}^{(d)} \boldsymbol{f}_{n,i}^\star - \sum_d^2 L_{\omega_y,ij}^{(d)} \boldsymbol{g}_{n,j}^\star
```

The matrices are defined as follows:

```math
\begin{gathered}
M = \text{diag}(J_{ij}\omega_{x,i}\omega_{y,j}), \\
K^{(j)}_{x,ik} = \omega_{y,j}\omega_{x,k}D_{x,ki}, \\
K^{(i)}_{y,jk} = \omega_{x,i}\omega_{y,k}D_{y,kj}, \\
\hat{K}^{(j)}_{x,ik} = -\omega_{y,j}\omega_{x,i}D_{x,ik} + \omega_{y,j}\left[l_{x,i}(1)l_{x,k}(1) - l_{x,i}(-1)l_{x,k}(-1)\right], \\
\hat{K}^{(i)}_{y,jk} = -\omega_{x,i}\omega_{y,j}D_{y,jk} + \omega_{x,i}\left[l_{y,j}(1)l_{y,k}(1) - l_{y,j}(-1)l_{y,k}(-1)\right], \\
L_{\omega_x,ij}^{(1)} = \omega_{y,j}l_{x,i}(-1), \\
L_{\omega_x,ij}^{(2)} = \omega_{y,j}l_{x,i}(1), \\
L_{\omega_y,ij}^{(1)} = \omega_{x,i}l_{y,j}(-1) \\
L_{\omega_y,ij}^{(2)} = \omega_{x,i}l_{y,j}(1).
\end{gathered}
```

!!! note
    Notice that the operators in 2D are very similar to the ones in 1D. They are simply the tensor product of their 1D versions with the vector ``\vec{\omega}`` corresponding to the other direction. For example, ``K^{\text{2D}}_x = K^{\text{1D}}_x \otimes \omega_y``, which is a third order tensor with size ``N_x \times N_x \times N_y``.

### Divergence operators

```@autodocs
Modules = [Flou]
Pages = ["Divergence.jl"]
```

## Developers

### Standard regions -- `StdRegions.jl`

Any standard region extending the `AbstractStdRegion` struct must implement the `is_tensor_product` trait. The idea behind is to use the tensor product structure of the nodes to employ a faster approach.

```@docs
is_tensor_product
```

## Index

```@index
```