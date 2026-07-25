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
struct PhysicsModel{TF, TT, NL, TG, TA <: AbstractArray{ComplexF64}, TVR <: AbstractVector{Float64}, TRW}
    to_freq::TF
    to_time::TT
    D::TA
    gamma::TG
    omega0::Float64
    gamma_W::TVR
    W::TVR
    dt::Float64
    N::Int
    fr::Float64
    RW::TRW
    nonlinear_function::NL
    # Pre-allocated buffers
    buf_t1::TA
    buf_t2::TA
    buf_f1::TA
    aux_data::NamedTuple
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

# Helper to transfer arrays to the same device/type as a template array
function _to_device(template::AbstractArray, host_array::AbstractArray)
    dev_array = similar(template, eltype(host_array), size(host_array))
    copyto!(dev_array, host_array)
    return dev_array
end

"""
    build_physics_model(grid::Grid, params::SimParams, [template::AbstractArray])

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

[`dispersion_operator`](@ref), [`raman_response`](@ref)
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)) where {S, M <: Medium}
    medium = params.medium
    N = grid.N

    # Extract physics flags
    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

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
        c = 299792458.0
        1.0, (gamma_input.n2 .* grid.W) ./ (c .* gamma_input.Aeff_function.(grid.W))
    else
        throw(ArgumentError("Unsupported nonlinearity type: $(typeof(gamma_input))"))
    end

    # Put gamma_W in FFT-natural order and move to device
    gamma_W_host = ifftshift(gamma_W_mon)
    gamma_W = _to_device(template, gamma_W_host)

    # Optional GPU-compatible arrays for W (used in shocks)
    W_host = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)
    W = _to_device(template, W_host)

    # Raman response in frequency domain (if enabled)
    fr = 0.0
    raman_freq_response = nothing
    if enable_raman
        # Compute the Raman response h_R(t) in the time domain
        fr, h_R = raman_response(grid, params.raman_model)
        # Frequency-domain Raman response
        raman_freq_response = _to_device(template, N .* ifft(ifftshift(h_R)))
    end

    # Compute dispersion operator. grid.V is in monotonic order, so the
    # operator must be fftshifted to FFT-natural order to align with AW.
    D_host = fftshift(dispersion_operator(grid, medium))
    D = _to_device(template, D_host)

    # Transform plans (FFTW.MEASURE for optimal performance). Standard optics
    # convention: time → frequency is ifft, frequency → time is fft.
    # By using `similar(template)` we allow for future GPU arrays.
    tmp = similar(template)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    # Select nonlinear function
    nonlinear_function = choose_nonlinear_term(enable_raman)

    # Pre-allocate working buffers for zero-allocation nonlinear operators
    # Using `similar(D)` links the buffer dimension to the field size, and
    # ensures they inherit any GPU array traits.
    buf_t1 = similar(D)
    buf_t2 = similar(D)
    buf_f1 = similar(D)

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
        NamedTuple()
    )
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:AmplifyingMedium}, [template])

Construct PhysicsModel for active amplifying fiber propagation with gain saturation & ASE.
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)) where {S, M <: AmplifyingMedium}
    medium = params.medium
    N = grid.N

    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

    gamma_input = medium.gamma
    gamma_z_model, gamma_W_mon = if gamma_input isa Number
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input / grid.omega0, W_factor
    else
        W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
        gamma_input, W_factor
    end

    gamma_W_host = ifftshift(gamma_W_mon)
    gamma_W = _to_device(template, gamma_W_host)

    W_host = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)
    W = _to_device(template, W_host)

    fr = 0.0
    raman_freq_response = nothing
    if enable_raman
        fr, h_R = raman_response(grid, params.raman_model)
        raman_freq_response = _to_device(template, N .* ifft(ifftshift(h_R)))
    end

    D_host = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion, medium.lambda0)))
    if medium.g0 isa Number
        D_host .+= medium.g0 / 2.0
    else
        D_host .+= fftshift(medium.g0.(grid.W) ./ 2.0)
    end
    D = _to_device(template, D_host)

    tmp = similar(template)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    nonlinear_function = choose_nonlinear_term(enable_raman)

    buf_t1 = similar(D)
    buf_t2 = similar(D)
    buf_f1 = similar(D)

    aux = (
        g0 = medium.g0,
        Esat = medium.Esat,
        noise_figure_db = medium.noise_figure_db,
        D_base = _to_device(template, D_host),
    )

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
        aux
    )
end

"""
    _semiconductor_spm(u, model::PhysicsModel, z)

Non-linear step for semiconductor waveguides (SOI, Ge, GaAs) with TPA, FCA, and FCR.
"""
function _semiconductor_spm(u, model::PhysicsModel, z)
    dt = model.dt
    aux = model.aux_data
    alpha2 = aux.alpha2
    Aeff = aux.Aeff
    sigma_fca = aux.sigma_fca
    k_fcr = aux.k_fcr
    tau_c = aux.tau_c
    omega0 = model.omega0
    hbar = 1.054571817e-34

    A = u # u is ALREADY in time domain!

    c_gen = alpha2 / (2.0 * hbar * omega0 * Aeff^2)
    Nc = zeros(Float64, length(A))
    N_curr = 0.0
    for i in 1:length(A)
        P_t = abs2(A[i])
        rate = c_gen * (P_t^2)
        N_curr = N_curr * exp(-dt / tau_c) + rate * dt
        Nc[i] = N_curr
    end

    gamma_val = (model.gamma isa Number) ? model.gamma * omega0 : model.gamma(z) * omega0
    tpa_loss = alpha2 / (2.0 * Aeff)
    fcr_phase = (omega0 / 2.99792458e8) * k_fcr

    @. model.buf_t2 = A * (
        im * (gamma_val * abs2(A) - fcr_phase * Nc) -
        (tpa_loss * abs2(A) + 0.5 * sigma_fca * Nc)
    )

    mul!(model.buf_f1, model.to_freq, model.buf_t2)
    @. model.buf_f1 = model.buf_f1 * (model.gamma_W / omega0)
    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:SemiconductorMedium}, [template])

Construct PhysicsModel for semiconductor waveguides (TPA & Free-Carrier Dynamics).
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)) where {S, M <: SemiconductorMedium}
    medium = params.medium
    N = grid.N

    enable_shock = params.self_steepening
    gamma_input = medium.gamma
    gamma_z_model = gamma_input isa Number ? (z -> gamma_input / grid.omega0) : (z -> gamma_input(z) / grid.omega0)
    W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
    gamma_W_host = ifftshift(W_factor)
    gamma_W = _to_device(template, gamma_W_host)

    W_host = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)
    W = _to_device(template, W_host)

    D_host = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion, medium.lambda0)))
    D = _to_device(template, D_host)

    tmp = similar(template)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    buf_t1 = similar(D)
    buf_t2 = similar(D)
    buf_f1 = similar(D)

    aux = (
        alpha2 = medium.alpha2,
        Aeff = medium.Aeff,
        sigma_fca = medium.sigma_fca,
        k_fcr = medium.k_fcr,
        tau_c = medium.tau_c,
    )

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
        0.0,
        nothing,
        _semiconductor_spm,
        buf_t1,
        buf_t2,
        buf_f1,
        aux
    )
end

# ===========================================================================
# Vectorial/Birefringent Coupled GNLSE Support
# ===========================================================================

"""
    _vectorial_spm_fwm(u, model::PhysicsModel, z)

Coupled Kerr (SPM, XPM, and FWM coherent coupling) nonlinear operator.
"""
function _vectorial_spm_fwm(u::AbstractMatrix{ComplexF64}, model::PhysicsModel, z::Real)
    ux = @view u[:, 1]
    uy = @view u[:, 2]

    deltabeta0 = model.aux_data.deltabeta0
    
    ph_x = exp(-2.0im * deltabeta0 * z)
    ph_y = conj(ph_x)

    # Compute SPM + XPM + FWM in time domain (zero allocations via @views)
    @views @. model.buf_t1[:, 1] = (abs2(ux) + (2.0 / 3.0) * abs2(uy)) * ux + (1.0 / 3.0) * (uy^2) * conj(ux) * ph_x
    @views @. model.buf_t1[:, 2] = (abs2(uy) + (2.0 / 3.0) * abs2(ux)) * uy + (1.0 / 3.0) * (ux^2) * conj(uy) * ph_y

    # Transform to frequency domain
    mul!(model.buf_f1, model.to_freq, model.buf_t1)

    # Get gamma at z
    gamma_z = eval_gamma(model.gamma, z, model.omega0)

    # Multiply by i * gamma_z * gamma_W (broadcasting vector over matrix works natively)
    @. model.buf_f1 = 1.0im * gamma_z * model.gamma_W * model.buf_f1

    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:BirefringentMedium}, [template::AbstractMatrix])

Construct PhysicsModel for Coupled GNLSE propagation.
"""
function build_physics_model(grid::Grid, params::SimParams{S, M}, template::AbstractMatrix=zeros(ComplexF64, grid.N, 2)) where {S, M <: BirefringentMedium}
    medium = params.medium
    N = grid.N
    enable_shock = params.self_steepening

    # Transform plans. By using `similar` and `template`, we prepare for GPU.
    tmp = similar(template)
    to_freq = plan_ifft(tmp, 1; flags=FFTW.MEASURE) # FFT along dimension 1
    to_time = plan_fft(tmp, 1; flags=FFTW.MEASURE)

    # Compute dispersion operators for x and y
    Dx = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion_x, medium.lambda0)))
    Dy = fftshift(dispersion_operator(grid, Medium(medium.length, medium.gamma, medium.loss, medium.dispersion_y, medium.lambda0)))
    
    # Combine into an N x 2 matrix
    D_host = hcat(Dx, Dy)
    D = _to_device(template, D_host)

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

    gamma_W_host = ifftshift(gamma_W_mon)
    gamma_W = _to_device(template, gamma_W_host)

    if params.raman_model !== nothing
        @warn "Raman scattering is currently not supported in the Birefringent Vectorial solver; falling back to Coupled Kerr terms only."
    end

    nonlinear_function = _vectorial_spm_fwm

    # Pre-allocate buffers
    buf_t1 = similar(template)
    buf_t2 = similar(template)
    buf_f1 = similar(template)
    
    W_host = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)
    W = _to_device(template, W_host)

    return PhysicsModel(
        to_freq,
        to_time,
        D,
        gamma_z_model,
        grid.omega0,
        gamma_W,
        W,
        grid.dt,
        N,
        0.0,
        nothing,
        nonlinear_function,
        buf_t1,
        buf_t2,
        buf_f1,
        (deltabeta0 = Float64(medium.deltabeta0),) # Auxiliary data
    )
end
