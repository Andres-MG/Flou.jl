<!--
Copyright (C) 2023 Andrés Mateo Gabín

This file is part of Flou.jl.

Flou.jl is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Flou.jl is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Flou.jl. If
not, see <https://www.gnu.org/licenses/>.
-->

```@meta
CurrentModule = Flou
```

# Introduction

*Flou* is a solver for hyperbolic partial differential equations, using a flux reconstruction approach.

## Hyperbolic equations

Equations of this type represent conservation laws: ``\boldsymbol{q}_t + \nabla\cdot\vec{\boldsymbol{f}} = \boldsymbol{s}``, where the integral values of ``\boldsymbol{q}`` are only modified by the fluxes through the boundaries or the sources/drains represented by ``\boldsymbol{s}``.

The numerical resolution of this equations requires the discretization of the variables, i.e. the components of ``\boldsymbol{q}`` and the time, ``t``. We rely on [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) for the temporal term, and *Flou* provides different spatial discretizations that can be directly integrated in time.

### Flux reconstruction (FR)

Starting with the 1D version of a generic conservation law, ``q_t + f_x = s``, defined on a certain region of the space, this domain is subdivided in non-overlapping elements. Inside them we represent the spatial distribution of any magnitude as a linear combination of a set of nodal functions, ``\phi_i(x)``,
```math
q \approx \sum_{i=0}^n q_i \phi_i(x), \quad \phi_i(x_j) = \delta_{ij}.
```

The fluxes require a little bit more of work. Since they appear in the equation as a derivative, its function approximation must be consistent with the approximation of the rest of the terms. This means that if ``\phi_i(x) \in \mathbb{P}^n``, the fluxes should be approximated with a different basis that, when derived, contains ``\{\phi_i(x)\}_i``,
```math
f \approx \sum_{i=0}^m f_i \psi_i(x), \quad \psi_i(x_j) = \delta_{ij}.
```

Considering this, and since we are seeking the time derivative of ``q_i`` at the different solution points ``\{x_0, \dots, x_n\}``, the discretization would be as follows,
```math
J_i(q_t)_i \approx -\sum_{j=0}^m f_j \psi_j'(x_i) + s(x_i), \quad i = 0, \dots, n.
```

Now we recall that our spatial domain is divided in a set of elements, and we have not imposed that the appoximations inside our elements are continous. In general, the numerical solution that we obtain with these methods is not continuous accross elements. However, information must be able to travel throughout the domain and we introduce this in our mathematical discretization with numerical fluxes, ``f^\star``. They act on two different states, ``q_l`` and ``q_r``, and return a sort of "common" flux by approximating the Riemann problem defined by the discontinuity between ``q_l`` and ``q_r``.

We use this at the element interfaces, so that the information arriving at one of the element boundaries is transmited by means of the discontinuity that it creates. To apply this to our numerical discretization, we first compute the values of ``q`` at a new set of points, ``\{\xi_0, \dots \xi_m\}`` that includes the boundaries, ``\xi=-1, \xi=1``. We compute the fluxes there as ``f_i = f(\xi_i) = f(q(\xi_i))`` and also use the values of ``q`` at the boundaries to obtain ``f^\star_l`` and ``f^\star_r``. We are now ready to represent the approximation of the flux taking into account the neighbouring elements,
```math
f \approx \sum_{i=0}^m f_i \psi_i(x) + \left(f^\star_l - \sum_{i=0}^m f_i \psi_i(-1)\right)g_l(x) + \left(f^\star_r - \sum_{i=0}^m f_i \psi_i(+1)\right)g_r(x).
```

The functions ``g_l(x)`` and ``g_r(x)`` are called reconstruction functions and there are several options. They are simply required to be valued ``1`` on "their" side and ``0`` on the other. In the next chapters we will see that other spatial discretizations can be recasted into the flux reconstruction framework by a certain choice of reconstruction functions. Finally, the FR spatial discretization is,
```math
J_i(q_t)_i \approx -\sum_{j=0}^m f_j \psi_j'(x_i) + \left(f^\star_l - \sum_{i=0}^m f_i \psi_i(-1)\right)\left.\frac{dg_l}{dx}\right\rvert_{x_i} + \left(f^\star_r - \sum_{i=0}^m f_i \psi_i(+1)\right)\left.\frac{dg_r}{dx}\right\rvert_{x_i} + s(x_i).
```

### Discontinuous Galerkin Spectral Element Method (DGSEM)

### Extension to higher dimensions in tensor-product elements

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

## Divergence operators

```@autodocs
Modules = [Flou.FlouSpatial]
Pages = ["OpDivergence.jl"]
```
