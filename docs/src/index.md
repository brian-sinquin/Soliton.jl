```@meta
CurrentModule = Soliton
```

# Soliton.jl

*Simulate ultrashort optical pulses in Julia.*

**Soliton.jl** solves the Generalized Nonlinear Schrödinger Equation (GNLSE) for pulse propagation in optical fibers, waveguides, and birefringent media. Combine dispersion, nonlinear effects, and numerical solvers through a composable API, using SI units throughout.

## Start Here

- **New to Soliton.jl?** Install the package below, run the quick start, then follow the [Getting Started guide](guide/basic.md).
- **Looking for a simulation to adapt?** Explore the [worked examples](examples/index.md), from supercontinuum generation to pulse amplification.
- **Choosing a physical model?** Read the [Physics Background](physics.md) and the topic guides below.
- **Need function signatures and options?** Browse the API for [media and simulation parameters](api/medium.md), [pulses](api/pulse.md), and [solvers](api/solvers.md).

## Installation

Soliton.jl requires Julia 1.10 or later. Install it from the Julia REPL:

```julia
using Pkg
Pkg.add("Soliton")
```

To use the development version from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/brian-sinquin/Soliton.jl")
```

## Quick Start

Propagate a pulse through one meter of telecom fiber. Times are in seconds, lengths and wavelengths in meters, and peak power in watts.

```julia
using Soliton

# Choose a fiber and use its reference wavelength for the grid.
medium = commercial_fiber("Corning_SMF28"; length=1.0, lambda0=1550e-9)
grid = create_grid(2^12, 10e-12, medium.lambda0)

# Sech pulse: 100 W peak power, 100 fs intensity FWHM.
pulse = sech_pulse(grid, 100.0, 100e-15)

# Propagate with the default adaptive ERK4IP solver and silica Raman response.
params = SimParams(; medium=medium, raman_model=BlowWood(), z_saves=50)
sol = solve(pulse, params)

# Output temporal power profile [W].
output_power = abs2.(sol.At[:, end])
```

The returned solution contains propagation distances (`sol.Z`), the time axis (`sol.t`), and temporal and spectral fields (`sol.At`, `sol.AW`). The [Getting Started guide](guide/basic.md) explains each step and how to choose a solver.

## Explore the Documentation

| Goal | Where to go |
|:---|:---|
| Understand the governing equation and numerical methods | [Physics Background](physics.md) |
| Configure dispersion, Kerr effects, and Raman response | [Dispersion](guide/dispersion.md), [Nonlinearity](guide/nonlinearity.md), [Raman Scattering](guide/raman.md) |
| Choose a commercial fiber or model an amplifier | [Fiber Catalog](guide/fibers.md), [Amplifying Fibers](guide/edfa.md) |
| Simulate gas-filled fibers or semiconductor waveguides | [Hollow-Core PCF](guide/hollowcore.md), [Silicon & Semiconductors](guide/semiconductor.md) |
| Model polarization or multi-stage optical systems | [Birefringent Propagation](guide/vectorial.md), [Cascaded Propagation](guide/cascading.md) |
| Add noise and analyze simulation results | [Noise Modeling](guide/noise.md), [Analysis API](api/analysis.md), [Unit Conversions](api/conversions.md) |

### Try a Worked Example

- [Supercontinuum generation in a photonic crystal fiber](examples/ex1_supercontinuum.md)
- [Higher-order soliton compression](examples/ex5_soliton_compression.md)
- [EDFA pulse amplification](examples/ex9_edfa_amplifier.md)
- [Multithreaded parameter sweeps](examples/ex10_parallel_sweep.md)

## Physical Effects & Capabilities

| Feature / Model | Description | Reference Module |
|:---|:---|:---|
| **Chromatic dispersion** | Taylor expansion ($\beta_2, \beta_3, \dots$), tabulated, or Sellmeier glass presets (`FusedSilica`, `SF6`, `SF57`) | `TaylorDispersion`, `Sellmeier` |
| **Kerr nonlinearity (SPM)** | Self-phase modulation ($i \gamma \|A\|^2 A$) | `Medium` |
| **Raman scattering** | Delayed silica response (Blow–Wood, Lin–Agrawal, Hollenbeck) | `BlowWood`, `Hollenbeck` |
| **Self-steepening** | Frequency-dependent shock term $\gamma \omega / \omega_0$ | `SimParams` |
| **Commercial Fiber Catalog** | Built-in presets (`Corning_SMF28`, `NKT_NL_PM_750`, `Thorlabs_PM780`, etc.) | `commercial_fiber` |
| **Active Amplifiers (EDFA/YDFA)** | Dynamic gain saturation $g(z)$ & quantum ASE noise seeding ($F_{\text{dB}}$) | `AmplifyingMedium` |
| **Gas Hollow-Core PCF** | Marcatili-Schmeltzer capillary model, noble & molecular gas Raman ($\text{H}_2, \text{N}_2$) | `HollowCoreFiber`, `MolecularRamanGas` |
| **Silicon Photonics (PICs)** | Two- & Three-Photon Absorption (TPA $\alpha_2$, 3PA $\alpha_3$), Free-Carrier Absorption (FCA), & Refraction (FCR) | `SemiconductorMedium` |
| **Birefringence / Vectorial** | Coupled GNLSE: SPM + XPM + coherent FWM across fast and slow axes | `BirefringentMedium`, `VectorialPulse` |
| **Cascaded System Dynamics** | Multi-stage propagation & lumped element processing (`Amplifier`, `Attenuator`, `Filter`) | `LumpedElement`, `solve` |

## Solvers

| Solver | Type | Description |
|:---|:---|:---|
| `ERK4IP` | Adaptive | Embedded Runge–Kutta 4(3) in the Interaction Picture (default) |
| `SSFM` | Fixed-step | Symmetric Split-Step Fourier Method |
| `AdaptiveSSFM` | Adaptive | Phase-controlled adaptive Split-Step Fourier Method |

## Project and Support

Find the source code on [GitHub](https://github.com/brian-sinquin/Soliton.jl), report problems in the [issue tracker](https://github.com/brian-sinquin/Soliton.jl/issues), or read the [contribution guidelines](https://github.com/brian-sinquin/Soliton.jl/blob/master/CONTRIBUTING.md).

## Module Reference

```@docs
Soliton
```
