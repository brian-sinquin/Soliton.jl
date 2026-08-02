```@meta
CurrentModule = GNLSE
```

# Basic Usage Guide

This guide walks through the fundamental workflow in GNLSE.jl step by step.

## Step 1: Create a Time–Frequency Grid

All simulations start with a [`Grid`](@ref) — it defines the time and frequency axes:

```julia
using GNLSE

# create_grid(N, time_window, lambda0)
# N           — number of grid points (power of 2 recommended for FFT efficiency)
# time_window — total temporal window [s]
# lambda0     — center wavelength [m]
grid = create_grid(2^12, 10e-12, 1550e-9)
```

The grid exposes:
- `grid.t` — time axis [s], monotonic
- `grid.V` — relative frequency axis ``\omega - \omega_0`` [rad/s], monotonic
- `grid.W` — absolute frequency ``\omega`` [rad/s]
- `grid.dt` — temporal sampling interval [s]
- `grid.omega0` — carrier angular frequency [rad/s]

## Step 2: Define a Fiber Medium

A [`Medium`](@ref) describes the physical fiber:

```julia
# Dispersion via Taylor coefficients: betas[1]=β₂, betas[2]=β₃, ...
medium = Medium(;
    length   = 1.0,              # [m]
    gamma    = 0.0011,           # [1/(W·m)] — telecom-fiber value
    loss     = 0.2e-3,           # [dB/m]
    betas    = [-21.5e-27],      # β₂ [s²/m] (anomalous dispersion)
    lambda0  = 1550e-9,          # [m]
)
```

## Step 3: Generate an Initial Pulse

Several pulse shapes are available:

```julia
# Hyperbolic secant: sech_pulse(grid, Pmax, FWHM)
pulse = sech_pulse(grid, 1000.0, 1e-12)   # 1 kW peak, 1 ps FWHM

# Gaussian: gaussian_pulse(grid, Pmax, FWHM)
pulse = gaussian_pulse(grid, 500.0, 500e-15)

# Lorentzian: lorentzian_pulse(grid, Pmax, FWHM)
pulse = lorentzian_pulse(grid, 100.0, 2e-12)

# Continuous wave (CW): cw_pulse(grid, power)
pulse = cw_pulse(grid, 1.0)
```

## Step 4: Configure Simulation Parameters

[`SimParams`](@ref) collects all run-time options:

```julia
params = SimParams(;
    medium          = medium,
    z_saves         = 100,       # number of output snapshots along fiber
    raman_model     = BlowWood(),# Raman model: BlowWood(), LinAgrawal(), Hollenbeck(), or nothing
    self_steepening = false,     # enable/disable shock term
)
```

## Step 5: Solve

```julia
sol = solve(pulse, params)
```

A [`Solution`](@ref) is returned:

| Field | Description |
|:---|:---|
| `sol.t` | Time axis [s] |
| `sol.W` | Frequency axis [rad/s] |
| `sol.Z` | Propagation distances [m] |
| `sol.At` | Time-domain field matrix `(N × z_saves)` |
| `sol.AW` | Freq-domain field matrix `(N × z_saves)` |

## Full Example: Soliton Propagation

```julia
using GNLSE

# Grid: 4096 points, 20 ps window, 1550 nm
grid = create_grid(2^12, 20e-12, 1550e-9)

# Anomalous-dispersion fiber (β₂ < 0), L_NL ≈ L_D → fundamental soliton
medium = Medium(;
    length   = 100.0,              # 100 m
    gamma    = 0.0011,
    loss     = 0.0,
    betas    = [-21.5e-27],        # β₂ [s²/m]
    lambda0  = 1550e-9,
)

# Fundamental soliton: N = √(γ P₀ T₀² / |β₂|) = 1
T0 = 1e-12  # soliton 1/e half-width [s]
P0 = abs(medium.dispersion.betas[1]) / (medium.gamma * T0^2)  # [W]
# sech_pulse takes the intensity FWHM; m = 2·arcsinh(1) ≈ 1.763 converts T₀ → FWHM
pulse = sech_pulse(grid, P0, 2 * log(1 + sqrt(2)) * T0)

# Solve without Raman for clean soliton
params = SimParams(; medium=medium, z_saves=200, raman_model=nothing)
sol = solve(pulse, params)

println("Max intensity at input  : ", maximum(abs2, sol.At[:, 1]))
println("Max intensity at output : ", maximum(abs2, sol.At[:, end]))
```

## Choosing a Solver

The default solver is `ERK4IP` (adaptive). You can switch to `SSFM` (fixed-step):

```julia
params = SimParams(;
    medium = medium,
    solver = SSFM(1e-4),       # step size 0.1 mm
)
```

Or tune the adaptive solver tolerances:

```julia
params = SimParams(;
    medium = medium,
    rtol   = 1e-8,             # relative tolerance
    atol   = 1e-10,            # absolute tolerance
)
```

## Parallel Parameter Sweeps (`solve_sweep`)

To sweep parameters (e.g. peak power $P_0$, fiber length $L$, or noise seeds) concurrently across all available Julia worker threads (`julia -t N`), use `solve_sweep`:

```julia
# Sweep peak power P0 from 1 kW to 10 kW in parallel
powers = range(1e3, 10e3; length=10)

solutions = solve_sweep(powers) do P0
    pulse  = sech_pulse(grid, P0, 100e-15)
    params = SimParams(; medium=medium, z_saves=20)
    return (pulse, params)
end
```

`solve_sweep` guarantees thread-safe, isolated memory allocations for each simulation trial.
