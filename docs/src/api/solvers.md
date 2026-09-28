```@meta
CurrentModule = Soliton
```

# Solvers

[`SimParams`](@ref) combines a medium with solver and output settings. Scalar simulations default to [`ERK4IP`](@ref); vectorial simulations require [`SSFM`](@ref). See [Getting Started](../guide/basic.md) and [Birefringent Propagation](../guide/vectorial.md) for complete workflows.

```@docs
GNLSESolver
ERK4IP
SSFM
AdaptiveSSFM
SimParams
Solution
VectorialSolution
solve
solve_sweep
```
