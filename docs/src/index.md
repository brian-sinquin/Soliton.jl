```@meta
CurrentModule = JuGNLSE
```

```@docs
JuGNLSE
```

# JuGNLSE.jl

*Generalized Nonlinear Schrödinger Equation solver in Julia*

---

**JuGNLSE.jl** is a high-performance Julia package for simulating the propagation of ultrashort optical pulses in nonlinear dispersive media such as optical fibers, waveguides, and birefringent media.

It implements the **Generalized Nonlinear Schrödinger Equation (GNLSE)** in natural SI units, with a rich physical model and a modern, composable API.

## Physical Effects

| Effect | Description |
|:---|:---|
| **Chromatic dispersion** | Taylor expansion (β₂, β₃, ...) or tabulated or Sellmeier dispersion |
| **Kerr nonlinearity (SPM)** | Self-phase modulation: `i γ |A|² A` |
| **Raman scattering** | Delayed molecular response (Blow–Wood, Lin–Agrawal, Hollenbeck) |
| **Self-steepening** | Shock term for sub-100 fs pulses |
| **Fiber loss** | Constant attenuation `α` [dB/m] |
| **Wavelength-dependent γ** | Frequency-dependent or Aeff-derived nonlinear coefficient |
| **Tapered/varying fiber** | z-dependent nonlinearity `γ(z)` |
| **Polarization / birefringence** | Coupled GNLSE: SPM + XPM + coherent FWM on two polarization axes |

## Solvers

| Solver | Type | Description |
|:---|:---|:---|
| `ERK4IP` | Adaptive | Embedded Runge–Kutta 4(3) in the Interaction Picture |
| `SSFM` | Fixed-step | Symmetric Split-Step Fourier Method |

## Installation

```julia
using Pkg
Pkg.add("JuGNLSE")
```

Or from the GitHub repository:
```julia
Pkg.add(url="https://github.com/brian-sinquin/JuGNLSE.jl")
```

## Quick Start

```julia
using JuGNLSE

# 1. Define the time–frequency grid
grid = create_grid(2^13, 12.5e-12, 835e-9)   # N, window [s], λ₀ [m]

# 2. Create the fiber medium
medium = Medium(;
    length   = 0.15,                         # m
    gamma    = 0.11,                         # 1/(W·m)
    loss     = 0.0,                          # dB/m
    betas    = [-11.83e-27, 8.13e-41],       # β₂ [s²/m], β₃ [s³/m]
    lambda0  = 835e-9,                       # m
)

# 3. Generate an initial pulse
pulse = sech_pulse(grid, 10000.0, 50e-15)    # Pmax [W], FWHM [s]

# 4. Configure simulation parameters and solve
params = SimParams(; medium=medium, raman_model=BlowWood(), self_steepening=true)
sol = solve(pulse, params)

# 5. Access the results
# sol.t   → time grid [s]
# sol.W   → frequency grid [rad/s]
# sol.Z   → propagation distances [m]
# sol.At  → time-domain field  (N × z_saves)
# sol.AW  → freq-domain field  (N × z_saves)
```

## Contents

```@contents
Pages = [
    "installation.md",
    "physics.md",
    "guide/basic.md",
    "guide/dispersion.md",
    "guide/raman.md",
    "guide/nonlinearity.md",
    "guide/cascading.md",
    "guide/vectorial.md",
    "api/medium.md",
    "api/grid.md",
    "api/pulse.md",
    "api/solvers.md",
    "api/analysis.md",
    "api/elements.md",
]
Depth = 2
```
