# GNLSE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brian-sinquin.github.io/GNLSE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brian-sinquin.github.io/GNLSE.jl/dev/)
[![Build status](https://github.com/brian-sinquin/GNLSE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/brian-sinquin/GNLSE.jl/actions/workflows/CI.yml)

GNLSE.jl is a Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE). It is designed to model the propagation of optical pulses in nonlinear media, such as optical fibers, with a focus on performance and numerical stability.

GNLSE.jl is the successor to [FiberNlse.jl](https://github.com/brian-sinquin/FiberNlse.jl), covering the items on that package's 2.0 roadmap (GNLSE physics, Raman scattering, self-steepening, many solvers) and then some. FiberNlse.jl may eventually be deprecated or become a lightweight, specialized wrapper around GNLSE.jl.

## Features

- Commercial fiber & glass presets (Corning, NKT, Thorlabs) and custom/tabulated dispersion
- Active amplifying fibers (EDFA/YDFA) with gain saturation and ASE noise
- Gas-filled hollow-core PCF with pressure-tunable dispersion and molecular Raman response
- Silicon photonics: two-photon absorption and free-carrier dynamics
- Coupled vectorial/birefringent GNLSE solver (SPM + XPM + FWM)
- Cascaded multi-stage pipelines (fibers, amplifiers, filters) via `|>` piping
- Adaptive (`ERK4IP`, `AdaptiveSSFM`) and fixed-step (`SSFM`) solvers, multi-threaded parameter sweeps

## Installation

The package can be installed using the Julia package manager:

```julia
using Pkg
Pkg.add("GNLSE")
```

## Basic Usage

All quantities are specified in natural SI units (seconds, meters, watts).

```julia
using GNLSE, Plots

# 1. Select a commercial fiber preset (e.g. NKT NL-PM-750 photonic crystal fiber)
medium = commercial_fiber("NKT_NL_PM_750"; length=0.15) # 15 cm fiber

# 2. Set up time-frequency grid (8192 resolution, 12.5 ps time window)
grid = create_grid(2^13, 12.5e-12, medium.lambda0)

# 3. Generate initial pulse (10 kW peak power, 50 fs sech² pulse)
pulse = sech_pulse(grid, 10000.0, 50e-15)

# 4. Propagate & solve GNLSE (Kerr SPM, Raman scattering & self-steepening)
solution = solve(pulse, SimParams(; medium=medium, raman_model=BlowWood(), self_steepening=true))

# 5. Visualize supercontinuum evolution (produces 4-panel dashboard)
plot(solution)
```

Custom fibers and measured dispersion curves are also supported via `Medium(length, gamma, loss, betas, lambda0)` or `TabulatedDispersion(detuning, beta)`.

## Documentation

For detailed information on the physical models, numerical methods, and API reference, please refer to the [documentation](https://brian-sinquin.github.io/GNLSE.jl/).
