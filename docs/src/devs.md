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

# Developers

## Standard regions -- `StdRegions.jl`

Any standard region extending the `AbstractStdRegion` struct must implement the `is_tensor_product` trait. The idea behind is to use the tensor product structure of the nodes to employ a faster approach.

```@docs
is_tensor_product
```

# Index

```@index
```