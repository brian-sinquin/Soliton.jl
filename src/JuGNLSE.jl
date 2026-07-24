"""
    JuGNLSE

Numerical solver for the Generalized Nonlinear Schrödinger Equation (GNLSE) following
gnlse-python conventions for optical pulse propagation in nonlinear dispersive media.

# Physical Effects

  - Dispersion: Arbitrary-order Taylor expansion β₂, β₃, β₄, ... [ps^n/m]
  - Kerr nonlinearity: Self-phase modulation, γ|A|² [1/(W·m)]
  - Raman scattering: Delayed nonlinear response (BlowWood, LinAgrawal, Hollenbeck models)
  - Self-steepening: Shock term for sub-100 fs pulses
  - Fiber loss: α(ω) [dB/m]

# Solvers

  - `solve()`: Adaptive ERK4IP (embedded RK4 in interaction picture) following gnlse-python

# Units

Natural SI units throughout:
  - Time: s (seconds)
  - Wavelength: m (meters)
  - Frequency: rad/s
  - Power: W (watts)
  - Distance: m (meters)
  - Dispersion: s^n/m
  - Nonlinearity: 1/(W·m)
  - Loss: dB/m

# Usage

```julia
using JuGNLSE

# Define grid (natural SI units)
grid = create_grid(2^13, 12.5e-12, 835e-9)  # resolution, time_window [s], λ [m]

# Define medium: Medium(L[m], γ[1/W/m], loss[dB/m], betas[sⁿ/m], λ[m])
medium = Medium(0.15, 0.11, 0.0, [-11.83e-27], 835e-9)

# Create pulse
pulse = sech_pulse(grid, 10000.0, 50e-15)  # Pmax [W], FWHM [s]

# Setup parameters
params = SimParams(medium=medium, z_saves=200, raman_model=BlowWood())

# Solve
solution = solve(pulse, params)
```

# Main Exports

**Types**: `Medium`, `Grid`, `Pulse`, `SimParams`, `Solution`, `RamanModel`, `BlowWood`, `LinAgrawal`, `Hollenbeck`

**Pulses**: `sech_pulse`, `gaussian_pulse`, `lorentzian_pulse`, `cw_pulse`

**Grids**: `create_grid`

**Solvers**: `solve`

**Physics**: `dispersion_operator`, `raman_response`, `build_physics_model`

# References

Adapted from gnlse-python (https://github.com/WUST-FOG/gnlse-python)
G. P. Agrawal, "Nonlinear Fiber Optics" (Academic Press, 2019)
"""
module JuGNLSE

using FFTW
using LinearAlgebra

# Physical constants - natural SI units
const c = 299792458.0  # Speed of light [m/s]

include("Types.jl")
using .Types

# All core types, physics definitions, and related functions
include("core.jl")
include("grid.jl")
include("pulses.jl")
include("dispersion.jl")
include("raman.jl")
include("nonlinearity.jl")

# Solvers
include("solvers/erk4ip.jl")

include("solver.jl")
include("analysis.jl")

# Export types
export Medium, SimParams, Grid, Pulse, Solution
export RamanModel, BlowWood, LinAgrawal, Hollenbeck
export DispersionModel, TaylorDispersion, TabulatedDispersion
export PhysicsModel  # Internal physics model struct

# Export grid functions
export create_grid, wavelength_grid

# Export pulse functions
export sech_pulse, gaussian_pulse, lorentzian_pulse, cw_pulse

# Export dispersion functions
export dispersion_operator, propagation_constant

# Export Raman functions
export raman_response

# Export solver interface
export solve

# Export physics model builder
export build_physics_model

# Export analysis functions
export pulse_energy, peak_power, fwhm, spectral_bandwidth, time_bandwidth_product
export photon_number, spectral_centroid
export dispersion_length, nonlinear_length, soliton_number
export add_noise, rin_rms, spectral_coherence

# Physical constant
export c

end # module
