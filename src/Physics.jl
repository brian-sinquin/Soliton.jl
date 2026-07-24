module Physics

using ..Types: Grid, Medium, SimParams, AbstractGammaCoefficient, Pulse
using ..Operators: dispersion_operator, compute_gamma
using ..Raman: raman_response
import FFTW
using FFTW: plan_ifft, plan_fft, ifft, fft, ifftshift, fftshift # For FFT plans and shifts
using LinearAlgebra # For `mul!`
import ..JuGNLSE: c

export PhysicsModel, build_physics_model

# Conversion constant (c is now global in JuGNLSE.jl)
# const C = 299792458.0 # Speed of light in vacuum [m/s]
_omega_to_lambda(omega::Real) = 2π * c / omega

"""
    PhysicsModel

Pre-computed operators and FFT plans for GNLSE propagation.

# Fields
- `to_freq`: Plan for the time → frequency transform (`ifft`)
- `to_time`: Plan for the frequency → time transform (`fft`)
- `D`: Dispersion operator [1/m], FFT-natural order
- `gamma`: Nonlinear coefficient γ/ω₀ [s/(W·m·rad)]
- `W`: Nonlinear-term frequency factor [rad/s], FFT-natural order. Equals the
   absolute angular frequency ω₀+V when self-steepening is on, or the constant
   ω₀ when off — so the nonlinear term iγ·W reduces to iγ_phys in that case.
- `dt`: Time step [s]
- `N`: Number of grid points
- `fr`: Raman fraction
- `RW`: Raman response in frequency domain (if enabled)
- `buf_t1`, `buf_t2`: Pre-allocated time-domain buffers
- `buf_f1`: Pre-allocated frequency-domain buffer
"""
struct PhysicsModel{TF, TT, NL, GC <: AbstractGammaCoefficient}
    to_freq::TF
    to_time::TT
    D::Vector{ComplexF64}
    gamma_coefficient::GC
    W::Vector{Float64}
    dt::Float64
    N::Int
    fr::Float64
    RW::Union{Nothing, Vector{ComplexF64}}
    nonlinear_function::NL
    lambda0::Float64
    buf_t1::Vector{ComplexF64}
    buf_t2::Vector{ComplexF64}
    buf_f1::Vector{ComplexF64}
end

"""
    _spm(u, model::PhysicsModel)

Kerr (self-phase modulation) nonlinear operator.

Computes `iγ·W·to_freq(u|u|²)`. Self-steepening is carried by `model.W`: with
self-steepening off, `W = ω₀` (constant) and iγ·W = iγ_phys; with it on,
`W = ω₀+V`, giving the shock term iγ_phys(1+V/ω₀). Zero allocations.

# See also

[`_spm_raman`](@ref)
"""
function _spm(u, model::PhysicsModel, z::Float64)
    # SPM term u·|u|²
    @. model.buf_t1 = u * abs2(u)

    # Transform to the frequency domain
    mul!(model.buf_f1, model.to_freq, model.buf_t1)

    # Get gamma value using the central wavelength (for WavelengthDependentGamma simplification)
    # and the current z for ZDependentGamma
    gamma_val = compute_gamma(model.gamma_coefficient, model.lambda0, z)

    # Multiply by iγW. The W in model.W already includes omega0 or omega0+V,
    # so we just need the physical gamma.
    @. model.buf_f1 = 1.0im * gamma_val * model.W * model.buf_f1

    return model.buf_f1
end

"""
    _spm_raman(u, model::PhysicsModel, z::Float64)

SPM with Raman scattering nonlinear operator.

Includes both instantaneous Kerr response and delayed Raman response via
convolution with hᵣ(t). Raman causes intrapulse frequency shift (red-shifting spectral peak) and energy transfer to Stokes wavelengths. Zero allocations via pre-allocated buffers.

# Physics 
Total response: n₂[|E|² + fᵣ∫hᵣ(t-t')|E(t')|²dt']E where fᵣ ≈ 0.18 is the Raman fraction. Convolution implemented efficiently via FFT multiplication. 
# Implementation: 
    conv = (|u|²) ⊛ hᵣ      via  to_time(to_freq(|u|²) .* RW)     op   = u .* ((1-fr)|u|² + fr·dt·conv)     result = iγ·W · to_freq(op) 
# See also 
[`raman_response`](@ref), [`_spm`](@ref)
"""
function _spm_raman(u, model::PhysicsModel, z::Float64)
    # `_spm_raman` is only selected when Raman is enabled, so RW is a Vector.
    # The assertion narrows the Union{Nothing,Vector} field type, keeping the
    # broadcast below type-stable and allocation-free.
    RW = model.RW::Vector{ComplexF64}

    # Intensity |u|²
    @. model.buf_t1 = abs2(u)

    # Raman convolution: multiply the intensity spectrum by the Raman response
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = model.buf_f1 * RW
    mul!(model.buf_t2, model.to_time, model.buf_f1)   # buf_t2 = |u|² ⊛ h_R

    # Get gamma value for the current z and central lambda
    gamma_val = compute_gamma(model.gamma_coefficient, model.lambda0, z)

    # Total nonlinearity (instantaneous Kerr + delayed Raman) times u
    @. model.buf_t1 = u * ((1.0 - model.fr) * abs2(u) + model.fr * model.dt * model.buf_t2)

    # Transform to the frequency domain and multiply by iγW
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = 1.0im * gamma_val * model.W * model.buf_f1

    return model.buf_f1
end

"""
    choose_nonlinear_term(raman::Bool)

Select the nonlinear operator. Self-steepening is not a separate operator — it is carried by `model.W` (see [`build_physics_model`](@ref)), so only the presence of Raman scattering selects between operators. 
# See also

[`build_physics_model`](@ref), [`_spm`](@ref), [`_spm_raman`](@ref)
"""
choose_nonlinear_term(raman::Bool) = raman ? _spm_raman : _spm

"""
    build_physics_model(grid::Grid, params::SimParams) 
Construct PhysicsModel with pre-computed operators for GNLSE propagation. 
Pre-computes all frequency-domain operators, FFT plans, and selects the appropriate nonlinear function. Called once at start of solve() to enable zero-allocation propagation in the ERK4IP stepper. 
# Arguments 
  - `grid`: Time-frequency grid
  - `params`: Simulation parameters (medium, physics flags) 
# Returns 
PhysicsModel struct ready for propagation 
# Implementation Details 
  - FFT plans use FFTW with FFTW.MEASURE flag for optimization   - Dispersion operator computed via `dispersion_operator(grid, medium)`   - Raman response computed in time domain then FFT'd to frequency domain   - Self-steepening: folded into `W` (ω₀+Δω if enabled, else constant ω₀)   - Nonlinear function selected via `choose_nonlinear_term(raman)`

# See also 
[`PhysicsModel`](@ref), [`propagate_erk4ip`](@ref), [`dispersion_operator`](@ref), [`raman_response`](@ref)
"""
function build_physics_model(grid::Grid, params::SimParams, gamma_coefficient::AbstractGammaCoefficient)
    medium = params.medium
    N = grid.N

    # Extract physics flags
    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

    # Transform plans (FFTW.MEASURE for optimal performance). Standard optics
    # convention: time → frequency is ifft, frequency → time is fft.
    tmp = zeros(ComplexF64, N)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    # Compute dispersion operator. grid.V is in monotonic order, so the
    # operator must be fftshifted to FFT-natural order to align with AW.
    D = fftshift(dispersion_operator(grid, medium))

    # Gamma (nonlinear coefficient) - now stored as AbstractGammaCoefficient

    # Raman response in frequency domain (if enabled)
    raman_freq_response = nothing
    fr = 0.0
    if enable_raman
        # Compute the Raman response h_R(t) in the time domain
        fr, h_R = raman_response(grid, params.raman_model)

        # Frequency-domain Raman response. ifftshift puts the causal response
        # (zero for t<0) into FFT-natural order with zero delay at index 1.
        # RW = N·ifft(h_R) so that to_time(to_freq(I) .* RW) evaluates the
        # circular convolution I ⊛ h_R for the ifft/fft transform pair.
        raman_freq_response = N .* ifft(ifftshift(h_R))
    end

    # Frequency factor for the nonlinear term iγ·W·FFT(...).
    # With self-steepening: W = ω₀+Δω (the true absolute frequency), giving the
    # shock term iγ_phys(1+Δω/ω₀). Without it: W = ω₀ (constant), so iγ·W reduces
    # to iγ_phys. grid.W is monotonic; ifftshift puts it in FFT-natural order.
    W = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)

    # Select nonlinear function (self-steepening is carried by W, not the operator)
    nonlinear_function = choose_nonlinear_term(enable_raman)

    # Pre-allocate working buffers for zero-allocation nonlinear operators
    buf_t1 = zeros(ComplexF64, N)
    buf_t2 = zeros(ComplexF64, N)
    buf_f1 = zeros(ComplexF64, N)

    # Construct model
    PhysicsModel(
        to_freq,
        to_time,
        D,
        gamma_coefficient, # Store the gamma_coefficient directly
        W,
        grid.dt,
        N,
        fr,
        raman_freq_response,
        nonlinear_function,
        grid.lambda0, # Pass lambda0
        buf_t1,
        buf_t2,
        buf_f1,
    )
end

end # module
