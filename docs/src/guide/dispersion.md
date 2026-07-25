```@meta
CurrentModule = JuGNLSE
```

# Dispersion Models

JuGNLSE.jl provides three ways to specify the chromatic dispersion of your waveguide.

## Taylor Expansion

The most common approach: provide dispersion coefficients ``\beta_n`` at the center wavelength.

```julia
# betas[1] = β₂, betas[2] = β₃, ... (β₀ and β₁ excluded)
disp = TaylorDispersion([-21.5e-27, 1.4e-40, -1.8e-55])
medium = Medium(0.5, 0.002, 0.0, disp, 1550e-9)
```

Alternatively, pass `betas` directly to the `Medium` keyword constructor:

```julia
medium = Medium(;
    length  = 0.5,
    gamma   = 0.002,
    betas   = [-21.5e-27, 1.4e-40],  # β₂, β₃
    lambda0 = 1550e-9,
)
```

For birefringent fibers, a group-velocity mismatch ``\beta_1`` (walk-off) can also be specified:

```julia
disp_slow = TaylorDispersion([-21.5e-27], 0.0)       # x axis: no walk-off
disp_fast = TaylorDispersion([-21.5e-27], 1.0e-10)   # y axis: 100 fs/mm walk-off
```

## Tabulated Dispersion

When you have measured ``\beta(\omega)`` data from your waveguide characterization:

```julia
using JuGNLSE

# Relative frequencies [rad/s] and corresponding β(ω) [1/m]
detuning = range(-5e13, 5e13; length=201)
beta     = 0.5 .* (-21.5e-27) .* detuning.^2  # example: parabolic

disp = TabulatedDispersion(collect(detuning), collect(beta))
medium = Medium(1.0, 0.002, 0.0, disp, 1550e-9)
```

The tabulated dispersion is **linearly interpolated** onto the simulation grid. Outside the provided range, the nearest endpoint value is held (flat extrapolation).

## Sellmeier Dispersion

For standard glass materials where the refractive index is described by Sellmeier coefficients:

```julia
# Fused silica Sellmeier coefficients (Malitson 1965)
B = [0.6961663, 0.4079426, 0.8974794]
C = [0.0684043^2, 0.1162414^2, 9.896161^2]  # [μm²]

disp = SellmeierDispersion(B, C; microns=true)  # microns=true (default)
medium = Medium(1.0, 0.002, 0.0, disp, 1550e-9)
```

!!! note
    The Sellmeier model computes ``n(\lambda)`` from which the full ``\beta(\omega)`` is derived numerically, including all dispersion orders automatically.

## Checking Your Dispersion

Use `dispersion_operator` to inspect the computed ``D(\omega)`` operator on your grid:

```julia
grid = create_grid(2^12, 10e-12, 1550e-9)
D = dispersion_operator(grid, medium)
# D is a complex vector in monotonic-frequency order
```

The imaginary part of `D` is the propagation-constant deviation ``B(\omega)``;
the real part is ``-\alpha/2`` (loss contribution).
