# Developers

## Standard regions -- `StdRegions.jl`

Any standard region extending the `AbstractStdRegion` struct must implement the `is_tensor_product` trait. The idea behind is to use the tensor product structure of the nodes to employ a faster approach.

```@docs
is_tensor_product
```

# Index

```@index
```