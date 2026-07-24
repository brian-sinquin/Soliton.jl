module JuGNLSE

# Physical constants - natural SI units
const c = 299792458.0  # Speed of light [m/s]

# Include new modular files
include("Types.jl")
include("Pulses.jl")
include("Operators.jl")
include("Raman.jl")
include("Physics.jl")
include("Solvers.jl")

# Propagation pipeline (if separate)
include("propagation_steps.jl")

# Solvers and analysis
# include("solver.jl") # Removed: Solvers are now in Solvers.jl
include("analysis.jl")

# Using statements for internal modules (to make functions/types directly available)
using .Types
using .Pulses
using .Operators
using .Raman
using .Physics
using .Solvers
using .PropagationSteps
using .Analysis
# Exports (consolidated) - export everything publicly needed from submodules
export c

# From Types.jl
export DispersionModel, TaylorDispersion, TabulatedDispersion
export Medium, Grid, RamanModel, BlowWood, LinAgrawal, Hollenbeck
export AbstractGNLSESolver, AbstractGammaCoefficient, ConstantGamma, ZDependentGamma, WavelengthDependentGamma
export Pulse, SimParams, GNLSEProblem, Solution

# From Pulses.jl
export create_grid, wavelength_grid
export sech_pulse, gaussian_pulse, lorentzian_pulse, cw_pulse

# From Operators.jl
export propagation_constant, dispersion_operator
export gamma

# From Raman.jl
export raman_response

# From Physics.jl
export PhysicsModel, build_physics_model

# From Solvers.jl
export ERK4IP, RK4, solve

# From propagation_steps.jl (assuming these are already defined in that file)
export AbstractPropagationStep, Fiber, Loss, Filter, Amplifier, propagate!

# From analysis.jl (assuming these are already defined in that file)
export pulse_energy, peak_power, fwhm, spectral_bandwidth, time_bandwidth_product
export photon_number, spectral_centroid
export dispersion_length, nonlinear_length, soliton_number
export add_noise, rin_rms, spectral_coherence

end # module
