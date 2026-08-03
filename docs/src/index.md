```@meta
CurrentModule = Soliton
```

```@docs
Soliton
```

# Soliton.jl

*Generalized Nonlinear Schrödinger Equation solver in Julia*

---

**Soliton.jl** is a high-performance Julia package for simulating the propagation of ultrashort optical pulses in nonlinear dispersive media such as optical fibers, waveguides, and birefringent media.

It implements the **Generalized Nonlinear Schrödinger Equation (GNLSE)** in natural SI units, with a rich physical model and a modern, composable API.

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
| **Silicon Photonics (PICs)** | Two-Photon Absorption (TPA $\alpha_2$), Free-Carrier Absorption (FCA), & Refraction (FCR) | `SemiconductorMedium` |
| **Birefringence / Vectorial** | Coupled GNLSE: SPM + XPM + coherent FWM across fast and slow axes | `BirefringentMedium`, `VectorialPulse` |
| **Cascaded System Dynamics** | Multi-stage propagation & lumped element processing (`Amplifier`, `Attenuator`, `Filter`) | `LumpedElement`, `solve` |

## Solvers

| Solver | Type | Description |
|:---|:---|:---|
| `ERK4IP` | Adaptive | Embedded Runge–Kutta 4(3) in the Interaction Picture (default) |
| `SSFM` | Fixed-step | Symmetric Split-Step Fourier Method |
| `AdaptiveSSFM` | Adaptive | Phase-controlled adaptive Split-Step Fourier Method |

## Installation

```julia
using Pkg
Pkg.add("Soliton")
```

Or from the GitHub repository:
```julia
Pkg.add(url="https://github.com/brian-sinquin/Soliton.jl")
```

## Quick Start

```julia
using Soliton

# 1. Define time-frequency grid
grid = create_grid(2^13, 12.5e-12, 835e-9)

# 2. Select commercial fiber or custom medium
medium = commercial_fiber("Corning_SMF28"; length=1.0, lambda0=1550e-9)

# 3. Generate initial pulse
pulse = sech_pulse(grid, 100.0, 100e-15)

# 4. Solve GNLSE
sol = solve(pulse, SimParams(; medium=medium, raman_model=BlowWood()))
```

## Documentation Contents

```@contents
Pages = [
    "physics.md",
    "guide/basic.md",
    "guide/dispersion.md",
    "guide/raman.md",
    "guide/nonlinearity.md",
    "guide/cascading.md",
    "guide/vectorial.md",
    "guide/fibers.md",
    "guide/edfa.md",
    "guide/hollowcore.md",
    "guide/semiconductor.md",
    "guide/noise.md",
    "examples/index.md",
    "api/medium.md",
    "api/grid.md",
    "api/pulse.md",
    "api/solvers.md",
    "api/dispersion.md",
    "api/nonlinearity.md",
    "api/elements.md",
    "api/fibers.md",
    "api/analysis.md",
]
Depth = 2
```
