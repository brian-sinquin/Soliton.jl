"""
Nonlinear operators for GNLSE propagation.

The package uses the standard optics FFT convention: the envelope spectrum is
`AW = ifft(At)` and the field is `At = fft(AW)`. So `to_freq` is `ifft` and
`to_time` is `fft`.
"""

using FFTW

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
struct PhysicsModel{TF, TT, NL, TG}
    to_freq::TF
    to_time::TT
    D::Vector{ComplexF64}
    gamma::TG
    omega0::Float64
    gamma_W::Vector{Float64}
    W::Vector{Float64}
    dt::Float64
    N::Int
    fr::Float64
    RW::Union{Nothing, Vector{ComplexF64}}
    nonlinear_function::NL
    # Pre-allocated buffers
    buf_t1::Vector{ComplexF64}
    buf_t2::Vector{ComplexF64}
    buf_f1::Vector{ComplexF64}
end

@inline function eval_gamma(gamma::Number, ::Real, ::Real)
    return gamma
end

@inline function eval_gamma(gamma::Function, z::Real, omega0::Float64)
    return gamma(z) / omega0
end

"""
    _spm(u, model::PhysicsModel, z::Real)

Kerr (self-phase modulation) nonlinear operator.

Computes `iγ·gamma_W·to_freq(u|u|²)`. Zero allocations.

# See also

[`_spm_raman`](@ref)
"""
function _spm(u, model::PhysicsModel, z::Real)
    # SPM term u·|u|²
    @. model.buf_t1 = u * abs2(u)

    # Transform to the frequency domain
    mul!(model.buf_f1, model.to_freq, model.buf_t1)

    # Get gamma at z
    gamma_z = eval_gamma(model.gamma, z, model.omega0)

    # Multiply by iγ_z * gamma_W
    @. model.buf_f1 = 1.0im * gamma_z * model.gamma_W * model.buf_f1

    return model.buf_f1
end

"""
    _spm_raman(u, model::PhysicsModel, z)

SPM with Raman scattering nonlinear operator.

Includes both instantaneous Kerr response and delayed Raman response via
convolution with hᵣ(t). Raman causes intrapulse frequency shift (red-shifting
spectral peak) and energy transfer to Stokes wavelengths. Zero allocations via
pre-allocated buffers.

# Physics

Total response: n₂[|E|² + fᵣ∫hᵣ(t-t')|E(t')|²dt']E where fᵣ ≈ 0.18 is the
Raman fraction. Convolution implemented efficiently via FFT multiplication.

# Implementation:

    conv = (|u|²) ⊛ hᵣ      via  to_time(to_freq(|u|²) .* RW)
    op   = u .* ((1-fr)|u|² + fr·dt·conv)
    result = iγ_z · gamma_W · to_freq(op)

# See also

[`raman_response`](@ref), [`_spm`](@ref)
"""
function _spm_raman(u, model::PhysicsModel, z::Real)
    # `_spm_raman` is only selected when Raman is enabled, so RW is a Vector.
    # The assertion narrows the Union{Nothing,Vector} field type, keeping the
    # broadcast below type-stable and allocation-free.
    RW = model.RW::Vector{ComplexF64}

    # Intensity |u|²
    @. model.buf_t1 = abs2(u)

    # Raman convolution: multiply the intensity spectrum by the Raman response
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = model.buf_f1 * RW
    mul!(model.buf_t2, model.to_time, model.buf_f1)   # buf_t2 = |u|² ⊛ hᵣ

    # Total nonlinearity (instantaneous Kerr + delayed Raman) times u
    @. model.buf_t1 = u * ((1.0 - model.fr) * abs2(u) + model.fr * model.dt * model.buf_t2)

    # Transform to the frequency domain and multiply by iγ_z * gamma_W
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    
    # Get gamma at z
    gamma_z = eval_gamma(model.gamma, z, model.omega0)
    
    @. model.buf_f1 = 1.0im * gamma_z * model.gamma_W * model.buf_f1

    return model.buf_f1
end

"""
    choose_nonlinear_term(raman::Bool)

Select the nonlinear operator. Self-steepening is not a separate operator —
it is carried by `model.W` (see [`build_physics_model`](@ref)), so only the
presence of Raman scattering selects between operators.

# See also

[`build_physics_model`](@ref), [`_spm`](@ref), [`_spm_raman`](@ref)
"""
choose_nonlinear_term(raman::Bool) = raman ? _spm_raman : _spm

"""
    build_physics_model(grid::Grid, params::SimParams)

Construct PhysicsModel with pre-computed operators for GNLSE propagation.

Pre-computes all frequency-domain operators, FFT plans, and selects the
appropriate nonlinear function. Called once at start of solve() to enable
zero-allocation propagation in the ERK4IP stepper.

# Arguments

  - `grid`: Time-frequency grid
  - `params`: Simulation parameters (medium, physics flags)

# Returns

PhysicsModel struct ready for propagation

# Implementation Details

  - FFT plans use FFTW with FFTW.MEASURE flag for optimization
  - Dispersion operator computed via `dispersion_operator(grid, medium)`
  - Raman response computed in time domain then FFT'd to frequency domain
  - Self-steepening: folded into `W` (ω₀+Δω if enabled, else constant ω₀)
  - Nonlinear function selected via `choose_nonlinear_term(raman)`

# See also

[`PhysicsModel`](@ref), [`propagate_erk4ip`](@ref), [`dispersion_operator`](@ref),
[`raman_response`](@ref)
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}) where {S, M <: Medium}
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

    # Gamma (nonlinear coefficient) - normalize by ω₀ if constant
    # Resolve gamma input and frequency-dependent gamma_W vector
    gamma_input = medium.gamma
    gamma_z_model, gamma_W_mon = if gamma_input isa Number
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input / grid.omega0, W_factor
    elseif gamma_input isa Function
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input, W_factor
    elseif gamma_input isa ConstantNonlinearity
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input.gamma / grid.omega0, W_factor
    elseif gamma_input isa FrequencyDependentNonlinearity
        1.0, gamma_input.gamma_function.(grid.W)
    elseif gamma_input isa NonlinearityFromEffectiveArea
        # gamma(w) = n2 * w / (c * Aeff(w))
        1.0, (gamma_input.n2 .* grid.W) ./ (c .* gamma_input.Aeff_function.(grid.W))
    else
        throw(ArgumentError("Unsupported nonlinearity type: $(typeof(gamma_input))"))
    end

    # Put gamma_W in FFT-natural order
    gamma_W = ifftshift(gamma_W_mon)

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

    # Frequency factor for the nonlinear term (for backward compatibility and test assertions)
    W = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)

    # Select nonlinear function
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
        gamma_z_model,
        grid.omega0,
        gamma_W,
        W,
        grid.dt,
        N,
        fr,
        raman_freq_response,
        nonlinear_function,
        buf_t1,
        buf_t2,
        buf_f1,
    )
end

# ===========================================================================
# Vectorial/Birefringent Coupled GNLSE Support
# ===========================================================================

"""
    VectorialPhysicsModel

Internal physics model holding operators and pre-allocated buffers for Couples GNLSE.
"""
struct VectorialPhysicsModel{TF, TT, NL, TG}
    to_freq::TF
    to_time::TT
    Dx::Vector{ComplexF64}
    Dy::Vector{ComplexF64}
    gamma::TG
    omega0::Float64
    gamma_W::Vector{Float64}
    deltabeta0::Float64
    N::Int
    nonlinear_function::NL
    buf_t1::Matrix{ComplexF64}
    buf_t2::Matrix{ComplexF64}
    buf_f1::Matrix{ComplexF64}
end

"""
    _vectorial_spm_fwm(u, model::VectorialPhysicsModel, z)

Coupled Kerr (SPM, XPM, and FWM coherent coupling) nonlinear operator.
"""
function _vectorial_spm_fwm(u::Matrix{ComplexF64}, model::VectorialPhysicsModel, z::Real)
    ux = @view u[:, 1]
    uy = @view u[:, 2]

    # Coherent coupling phase mismatch factors
    ph_x = exp(-2.0im * model.deltabeta0 * z)
    ph_y = conj(ph_x)

    # Compute SPM + XPM + FWM in time domain
    @. model.buf_t1[:, 1] = (abs2(ux) + (2.0 / 3.0) * abs2(uy)) * ux + (1.0 / 3.0) * (uy^2) * conj(ux) * ph_x
    @. model.buf_t1[:, 2] = (abs2(uy) + (2.0 / 3.0) * abs2(ux)) * uy + (1.0 / 3.0) * (ux^2) * conj(uy) * ph_y

    # Transform to frequency domain
    mul!(@view(model.buf_f1[:, 1]), model.to_freq, @view(model.buf_t1[:, 1]))
    mul!(@view(model.buf_f1[:, 2]), model.to_freq, @view(model.buf_t1[:, 2]))

    # Get gamma at z
    gamma_z = eval_gamma(model.gamma, z, model.omega0)

    # Multiply by i * gamma_z * gamma_W
    @. model.buf_f1[:, 1] = 1.0im * gamma_z * model.gamma_W * model.buf_f1[:, 1]
    @. model.buf_f1[:, 2] = 1.0im * gamma_z * model.gamma_W * model.buf_f1[:, 2]

    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:BirefringentMedium})

Construct VectorialPhysicsModel for Coupled GNLSE propagation.
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}) where {S, M <: BirefringentMedium}
    medium = params.medium
    N = grid.N
    enable_shock = params.self_steepening

    # Transform plans
    tmp_col = zeros(ComplexF64, N)
    to_freq = plan_ifft(tmp_col; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp_col; flags=FFTW.MEASURE)

    # Compute dispersion operators for x and y
    Dx = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion_x, medium.lambda0)))
    Dy = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion_y, medium.lambda0)))

    # Compute gamma and gamma_W
    gamma_input = medium.gamma
    gamma_z_model, gamma_W_mon = if gamma_input isa Number
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input / grid.omega0, W_factor
    elseif gamma_input isa Function
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input, W_factor
    elseif gamma_input isa ConstantNonlinearity
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input.gamma / grid.omega0, W_factor
    elseif gamma_input isa FrequencyDependentNonlinearity
        1.0, gamma_input.gamma_function.(grid.W)
    elseif gamma_input isa NonlinearityFromEffectiveArea
        c_const = 299792458.0
        1.0, (gamma_input.n2 .* grid.W) ./ (c_const .* gamma_input.Aeff_function.(grid.W))
    else
        throw(ArgumentError("Unsupported nonlinearity type: $(typeof(gamma_input))"))
    end

    gamma_W = ifftshift(gamma_W_mon)

    if params.raman_model !== nothing
        @warn "Raman scattering is currently not supported in the Birefringent Vectorial solver; falling back to Coupled Kerr terms only."
    end

    nonlinear_function = _vectorial_spm_fwm

    # Pre-allocate buffers
    buf_t1 = zeros(ComplexF64, N, 2)
    buf_t2 = zeros(ComplexF64, N, 2)
    buf_f1 = zeros(ComplexF64, N, 2)

    return VectorialPhysicsModel(
        to_freq,
        to_time,
        Dx,
        Dy,
        gamma_z_model,
        grid.omega0,
        gamma_W,
        Float64(medium.deltabeta0),
        N,
        nonlinear_function,
        buf_t1,
        buf_t2,
        buf_f1,
    )
end
