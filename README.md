# Flou

[![arXiv:2211.05066](https://img.shields.io/badge/arXiv-2211.05066-green?style=flat-square)](https://arxiv.org/abs/2211.05066)

Flou is a (very incomplete) framework to solve partial differential equations. It currently
implements solvers for hyperbolic equations using a high-order flux-reconstruction approach,
supporting only the *Discontinuous Galerkin Spectral Element Method* (DGSEM) reconstruction
functions.

This repository contains the code to reproduce the results of Fig. 2 in <https://arxiv.org/abs/2211.05066>.
The spatial discretization is implemented in `src/FlouCommon/` and `src/FlouSpatial/`, and the
results can be recovered by running `figure2/src/convergence.jl` and `figure2/src/errors.jl`:

```console
$ cd figure2
$ julia -p <n procs> --project=. -- src/convergence.jl
$ julia -p <n procs> --project=. -- src/errors.jl
```

You might need to run ```julia -e "using Pkg; Pkg.instantiate()"``` after entering the `figure2`
folder.

## License notice

Copyright (C) 2023 Andrés Mateo Gabín

Flou.jl is free software; you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
