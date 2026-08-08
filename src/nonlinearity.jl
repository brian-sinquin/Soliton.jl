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
  - `buf_f1`, `buf_f2`: Pre-allocated frequency-domain buffers. Most nonlinear
    functions only use `buf_f1`; `buf_f2` exists so `_semiconductor_spm` can
    FFT the shock-weighted (Kerr/TPA/3PA) and non-shocked (FCA/FCR) nonlinear
    contributions separately before summing them in frequency domain (see
    [`_semiconductor_spm`](@ref)).
"""
struct PhysicsModel{
    TF, TT, NL, TG, TA <: AbstractArray{ComplexF64}, TVR <: AbstractVector{Float64}, TRW
}
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
    buf_f2::TA
    aux_data::NamedTuple
end

@inline function eval_gamma(gamma::Number, ::Real, ::Real)
    return gamma
end

@inline function eval_gamma(gamma::Function, z::Real, omega0::Float64)
    return gamma(z) / omega0
end

"""
    step_index_aeff(core_radius_m::Real, NA::Real, lambda_m::Real) -> Float64

Calculate the effective mode area A_eff(λ) [m²] for a step-index single-mode fiber
using the Marcuse empirical Gaussian mode-field radius formula:

    V(λ) = (2π a / λ) · NA
    w(λ) = a · (0.65 + 1.619 / V^1.5 + 2.879 / V^6)
    A_eff(λ) = π w(λ)²

Reference: D. Marcuse, "Loss analysis of single-mode fiber splices," Bell
Syst. Tech. J. 56, 703-718 (1977).
"""
function step_index_aeff(core_radius_m::Real, NA::Real, lambda_m::Real)
    core_radius_m > 0 || throw(ArgumentError("Core radius must be positive"))
    NA > 0 || throw(ArgumentError("Numerical aperture NA must be positive"))
    lambda_m > 0 || throw(ArgumentError("Wavelength must be positive"))

    V = (2π * core_radius_m / lambda_m) * NA
    V > 0.8 || throw(ArgumentError("V-number $V too small for single-mode guidance"))

    w = core_radius_m * (0.65 + 1.619 / (V^1.5) + 2.879 / (V^6))
    return π * w^2
end

"""
    MarcuseAeff(core_radius_m::Real, NA::Real)

Callable object representing a wavelength-dependent mode area A_eff(ω) [m²] based on Marcuse's model.
"""
struct MarcuseAeff
    core_radius_m::Float64
    NA::Float64

    function MarcuseAeff(core_radius_m::Real, NA::Real)
        new(Float64(core_radius_m), Float64(NA))
    end
end

function (m::MarcuseAeff)(omega::Real)
    lambda_m = 2π * c / abs(omega)
    return step_index_aeff(m.core_radius_m, m.NA, lambda_m)
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
    factor = 1.0im * gamma_z

    # Multiply by iγ_z * gamma_W
    @. model.buf_f1 = factor * model.gamma_W * model.buf_f1

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
    factor = 1.0im * gamma_z

    @. model.buf_f1 = factor * model.gamma_W * model.buf_f1

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
    _resolve_gamma(gamma_input, grid::Grid, enable_shock::Bool) -> (gamma_z_model, gamma_W_mon)

Resolve a medium's `gamma` field — a bare `Number`, a `Function` of z, or a
[`NonlinearityModel`](@ref) subtype (`ConstantNonlinearity`,
`FrequencyDependentNonlinearity`, `NonlinearityFromEffectiveArea`) — into the
two quantities every `build_physics_model` method needs:

  - `gamma_z_model`: passed to [`eval_gamma`](@ref) at each nonlinear-step
    evaluation (a `Number` or a `Function` of `z`).
  - `gamma_W_mon`: the frequency-domain `γ(ω)` weighting vector (monotonic
    frequency order, matching `grid.W`).

Dispatched by type instead of an `isa` chain, so every `AbstractMedium` gets
the same `NonlinearityModel` support for free — previously `AmplifyingMedium`
and `SemiconductorMedium` only handled `Number`/`Function` and would hit a
`MethodError` in `eval_gamma` if given a `ConstantNonlinearity`, etc.
"""
function _resolve_gamma(gamma_input::Number, grid::Grid, enable_shock::Bool)
    W_factor = enable_shock ? grid.W : fill(grid.omega0, grid.N)
    return gamma_input / grid.omega0, W_factor
end

function _resolve_gamma(gamma_input::Function, grid::Grid, enable_shock::Bool)
    W_factor = enable_shock ? grid.W : fill(grid.omega0, grid.N)
    return gamma_input, W_factor
end

function _resolve_gamma(gamma_input::ConstantNonlinearity, grid::Grid, enable_shock::Bool)
    W_factor = enable_shock ? grid.W : fill(grid.omega0, grid.N)
    return gamma_input.gamma / grid.omega0, W_factor
end

function _resolve_gamma(gamma_input::FrequencyDependentNonlinearity, grid::Grid, ::Bool)
    return 1.0, gamma_input.gamma_function.(grid.W)
end

function _resolve_gamma(gamma_input::NonlinearityFromEffectiveArea, grid::Grid, ::Bool)
    return 1.0, (gamma_input.n2 .* grid.W) ./ (c .* gamma_input.Aeff_function.(grid.W))
end

function _resolve_gamma(gamma_input, ::Grid, ::Bool)
    throw(ArgumentError("Unsupported nonlinearity type: $(typeof(gamma_input))"))
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
function build_physics_model(
    grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)
) where {S, M <: Medium}
    medium = params.medium
    N = grid.N

    # Extract physics flags
    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

    # Gamma (nonlinear coefficient) - normalize by ω₀ if constant
    gamma_z_model, gamma_W_mon = _resolve_gamma(medium.gamma, grid, enable_shock)

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
    buf_f2 = similar(D)

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
        buf_f2,
        NamedTuple(),
    )
end

function _amplifying_spm(u, model::PhysicsModel, z::Real)
    aux = model.aux_data
    g0_val = aux.g0 isa Real ? Float64(aux.g0) : 0.0
    Esat = aux.Esat
    dt = model.dt

    E_pulse = sum(abs2, u) * dt
    delta_g = -0.5 * g0_val * (E_pulse / (Esat + E_pulse))

    gamma_z = eval_gamma(model.gamma, z, model.omega0)
    ig = 1.0im * gamma_z

    @. model.buf_t1 = u * (ig * abs2(u) + delta_g)

    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    inv_w0 = 1.0 / model.omega0
    @. model.buf_f1 = model.buf_f1 * model.gamma_W * inv_w0

    return model.buf_f1
end

function _amplifying_spm_raman(u, model::PhysicsModel, z::Real)
    aux = model.aux_data
    g0_val = aux.g0 isa Real ? Float64(aux.g0) : 0.0
    Esat = aux.Esat
    dt = model.dt

    E_pulse = sum(abs2, u) * dt
    delta_g = -0.5 * g0_val * (E_pulse / (Esat + E_pulse))

    RW = model.RW::Vector{ComplexF64}

    @. model.buf_t1 = abs2(u)
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = model.buf_f1 * RW
    mul!(model.buf_t2, model.to_time, model.buf_f1)

    gamma_z = eval_gamma(model.gamma, z, model.omega0)
    ig = 1.0im * gamma_z

    @. model.buf_t1 =
        u * (ig * ((1.0 - model.fr) * abs2(u) + model.fr * dt * model.buf_t2) + delta_g)

    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    inv_w0 = 1.0 / model.omega0
    @. model.buf_f1 = model.buf_f1 * model.gamma_W * inv_w0

    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:AmplifyingMedium}, [template])

Construct PhysicsModel for active amplifying fiber propagation with gain saturation & ASE.
"""
function build_physics_model(
    grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)
) where {S, M <: AmplifyingMedium}
    medium = params.medium
    N = grid.N

    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

    gamma_z_model, gamma_W_mon = _resolve_gamma(medium.gamma, grid, enable_shock)

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

    # D operator carries the unsaturated (small-signal) gain g₀/2; the saturation
    # correction −g₀/2 · E/(Esat+E) is applied dynamically in `_amplifying_spm`.
    # Together they implement: g_net = g₀ · Esat/(Esat+E).
    D_host = fftshift(dispersion_operator(grid, medium))
    D = _to_device(template, D_host)

    tmp = similar(template)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    nonlinear_function = enable_raman ? _amplifying_spm_raman : _amplifying_spm

    buf_t1 = similar(D)
    buf_t2 = similar(D)
    buf_f1 = similar(D)
    buf_f2 = similar(D)

    aux = (
        g0=medium.g0,
        Esat=medium.Esat,
        noise_figure_db=medium.noise_figure_db,
        D_base=_to_device(template, D_host),
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
        buf_f2,
        aux,
    )
end

"""
    inject_ase_noise!(U, u_time, model::PhysicsModel, dz::Real, rng::AbstractRNG=default_rng())

Add one propagation step's worth of Amplified Spontaneous Emission (ASE)
quantum noise to the frequency-domain field `U` (FFT-natural order, in place).
A no-op for any medium other than `AmplifyingMedium`, detected by the absence
of `:noise_figure_db`/`:g0`/`:Esat` in `model.aux_data`.

`u_time` must hold the time-domain field at the *start* of this step — it is
used to evaluate the saturated gain, matching [`_amplifying_spm`](@ref)/
[`_amplifying_spm_raman`](@ref).

# Physics

```math
n_{sp} = \\frac{10^{F_{dB}/10}}{2}, \\qquad
n_{ase} = n_{sp}\\left(e^{g(z)\\Delta z} - 1\\right), \\qquad
S_{ASE} = n_{sp}\\,\\hbar\\,\\omega_0\\left(e^{g(z)\\Delta z}-1\\right)
```

matching the Active Amplifying Fibers section of `docs/src/physics.md`.
`n_ase` is injected as complex-Gaussian quadrature noise using the same
per-mode convention as [`add_noise`](@ref)'s quantum-noise seeding
(`N·dt·⟨|δU|²⟩ = n_ase·ħω₀`), except flat across all frequency modes (a
constant `ω₀` in place of the per-mode `ω`), consistent with the flat-ASE-PSD
approximation used in `docs/src/physics.md`.

Reference: E. Desurvire, "Erbium-Doped Fiber Amplifiers" (Wiley, 1994), Ch. 2.
"""
function inject_ase_noise!(
    U::AbstractVector{ComplexF64},
    u_time::AbstractVector{ComplexF64},
    model::PhysicsModel,
    dz::Real,
    rng::AbstractRNG=default_rng(),
)
    aux = model.aux_data
    haskey(aux, :noise_figure_db) || return U

    g0_val = aux.g0 isa Real ? Float64(aux.g0) : 0.0
    Esat = aux.Esat
    dt = model.dt
    N = model.N
    omega0 = model.omega0

    E_pulse = sum(abs2, u_time) * dt
    g_local = g0_val * (Esat / (Esat + E_pulse))  # net saturated gain [1/m]

    n_sp = 10.0^(aux.noise_figure_db / 10.0) / 2.0
    n_ase = n_sp * (exp(g_local * dz) - 1.0)
    n_ase > 0 || return U

    hbar_local = 1.054571817e-34
    scale = sqrt(n_ase * hbar_local * omega0 / (2 * N * dt))
    @inbounds for m in eachindex(U)
        U[m] += scale * complex(randn(rng), randn(rng))
    end

    return U
end

"""
    _semiconductor_gamma_at_z(gamma_input, z::Real) -> Float64

Resolve a `SemiconductorMedium`'s `gamma` field to a raw physical value
[1/(W·m)] at position `z`. Unlike [`eval_gamma`](@ref)/[`_resolve_gamma`](@ref)
(used by `_spm`/`_spm_raman`/`_amplifying_spm`/`_vectorial_spm_fwm`, which
store γ pre-divided by ω₀ and reconstruct it via `gamma_W`),
[`_semiconductor_spm`](@ref) applies the shock/self-steepening factor as a
separate post-multiplication (`model.gamma_W / omega0`, see below) and so
needs the un-normalized γ directly.
"""
@inline function _semiconductor_gamma_at_z(gamma_input::Number, ::Real)
    return Float64(gamma_input)
end

@inline function _semiconductor_gamma_at_z(gamma_input::Function, z::Real)
    return Float64(gamma_input(z))
end

@inline function _semiconductor_gamma_at_z(gamma_input::ConstantNonlinearity, ::Real)
    return Float64(gamma_input.gamma)
end

@inline function _semiconductor_gamma_at_z(gamma_input, ::Real)
    throw(ArgumentError(
        "SemiconductorMedium only supports Number, Function, or ConstantNonlinearity " *
        "gamma (got $(typeof(gamma_input))); FrequencyDependentNonlinearity and " *
        "NonlinearityFromEffectiveArea are not yet supported for TPA/FCA/FCR propagation."
    ))
end

"""
    _semiconductor_spm(u, model::PhysicsModel, z)

Non-linear step for semiconductor waveguides (SOI, Ge, GaAs) with TPA, 3PA, FCA, and FCR.

TPA and 3PA free-carrier generation add (two- and three-photon transitions are
independent absorption channels): `dN/dt = [α₂|A|⁴/(2Aeff²) + α₃|A|⁶/(3Aeff³)] / (ħω₀) − N/τc`.
Setting `alpha3 = 0` recovers pure-TPA behavior.

Self-steepening (the `gamma_W/ω₀` shock weighting) is applied only to the Kerr/TPA/3PA
term, not to FCA/FCR: shock arises from the frequency dependence of the χ⁽³⁾/χ⁽⁵⁾
polarization response near ω₀, whereas free-carrier plasma effects are a separate,
slowly-varying (ps–ns) process with no such frequency dependence near the optical
carrier. The two contributions are therefore FFT'd separately (`buf_t2`→`buf_f1` with
shock weighting, `buf_t1`→`buf_f2` without) and summed in frequency domain.
"""
function _semiconductor_spm(u, model::PhysicsModel, z)
    dt = model.dt
    aux = model.aux_data
    alpha2 = aux.alpha2
    Aeff = aux.Aeff
    sigma_fca = aux.sigma_fca
    k_fcr = aux.k_fcr
    tau_c = aux.tau_c
    alpha3 = aux.alpha3
    Nc = aux.Nc
    omega0 = model.omega0
    hbar = 1.054571817e-34

    A = u # u is ALREADY in time domain!

    c_gen2 = alpha2 / (2.0 * hbar * omega0 * Aeff^2)
    c_gen3 = alpha3 / (3.0 * hbar * omega0 * Aeff^3)
    N_curr = 0.0
    decay = exp(-dt / tau_c)
    gain_factor = tau_c * (1.0 - decay)
    @inbounds for i in 1:length(A)
        P_t = abs2(A[i])
        rate = c_gen2 * (P_t^2) + c_gen3 * (P_t^3)
        N_curr = N_curr * decay + rate * gain_factor
        Nc[i] = N_curr
    end

    gamma_val = _semiconductor_gamma_at_z(model.gamma, z)
    tpa_loss = alpha2 / (2.0 * Aeff)
    tpa3_loss = alpha3 / (2.0 * Aeff^2)
    fcr_phase = (omega0 / 2.99792458e8) * k_fcr

    # Kerr + TPA + 3PA: bound-electron polarization response near ω₀ -> gets the
    # shock/self-steepening weighting.
    @. model.buf_t2 =
        A * (
            im * gamma_val * abs2(A) - (tpa_loss * abs2(A) + tpa3_loss * abs2(A)^2)
        )
    mul!(model.buf_f1, model.to_freq, model.buf_t2)
    @. model.buf_f1 = model.buf_f1 * (model.gamma_W / omega0)

    # FCA + FCR: free-carrier plasma response, no shock weighting.
    @. model.buf_t1 = A * (-im * fcr_phase * Nc - 0.5 * sigma_fca * Nc)
    mul!(model.buf_f2, model.to_freq, model.buf_t1)

    @. model.buf_f1 = model.buf_f1 + model.buf_f2
    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:SemiconductorMedium}, [template])

Construct PhysicsModel for semiconductor waveguides (TPA, 3PA & Free-Carrier Dynamics).
"""
function build_physics_model(
    grid::Grid, params::SimParams{S, M}, template::AbstractVector=zeros(ComplexF64, grid.N)
) where {S, M <: SemiconductorMedium}
    medium = params.medium
    N = grid.N

    enable_shock = params.self_steepening
    # SemiconductorMedium needs the raw (un-normalized) gamma, resolved at
    # call time by `_semiconductor_gamma_at_z` — unlike `_resolve_gamma`, it is
    # stored on the model as-is rather than pre-divided by omega0.
    gamma_z_model = medium.gamma
    W_factor = enable_shock ? grid.W : fill(grid.omega0, N)
    gamma_W_host = ifftshift(W_factor)
    gamma_W = _to_device(template, gamma_W_host)

    W_host = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)
    W = _to_device(template, W_host)

    D_host = fftshift(dispersion_operator(grid, medium))
    D = _to_device(template, D_host)

    tmp = similar(template)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    buf_t1 = similar(D)
    buf_t2 = similar(D)
    buf_f1 = similar(D)
    buf_f2 = similar(D)

    aux = (
        alpha2=medium.alpha2,
        Aeff=medium.Aeff,
        sigma_fca=medium.sigma_fca,
        k_fcr=medium.k_fcr,
        tau_c=medium.tau_c,
        alpha3=medium.alpha3,
        Nc=zeros(Float64, N),
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
        buf_f2,
        aux,
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
    @views @. model.buf_t1[:, 1] =
        (abs2(ux) + (2.0 / 3.0) * abs2(uy)) * ux + (1.0 / 3.0) * (uy^2) * conj(ux) * ph_x
    @views @. model.buf_t1[:, 2] =
        (abs2(uy) + (2.0 / 3.0) * abs2(ux)) * uy + (1.0 / 3.0) * (ux^2) * conj(uy) * ph_y

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
function build_physics_model(
    grid::Grid,
    params::SimParams{S, M},
    template::AbstractMatrix=zeros(ComplexF64, grid.N, 2),
) where {S, M <: BirefringentMedium}
    medium = params.medium
    N = grid.N
    enable_shock = params.self_steepening

    # Transform plans. By using `similar` and `template`, we prepare for GPU.
    tmp = similar(template)
    to_freq = plan_ifft(tmp, 1; flags=FFTW.MEASURE) # FFT along dimension 1
    to_time = plan_fft(tmp, 1; flags=FFTW.MEASURE)

    # Compute dispersion operators for x and y
    Dx = fftshift(
        dispersion_operator(
            grid,
            Medium(
                medium.length,
                medium.gamma,
                medium.loss,
                medium.dispersion_x,
                medium.lambda0,
            ),
        ),
    )
    Dy = fftshift(
        dispersion_operator(
            grid,
            Medium(
                medium.length,
                medium.gamma,
                medium.loss,
                medium.dispersion_y,
                medium.lambda0,
            ),
        ),
    )

    # Combine into an N x 2 matrix
    D_host = hcat(Dx, Dy)
    D = _to_device(template, D_host)

    # Compute gamma and gamma_W
    gamma_z_model, gamma_W_mon = _resolve_gamma(medium.gamma, grid, enable_shock)

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
    buf_f2 = similar(template)

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
        buf_f2,
        (deltabeta0=Float64(medium.deltabeta0),), # Auxiliary data
    )
end

# ===========================================================================
# Second-Order (χ⁽²⁾) Nonlinearity Support
# ===========================================================================

"""
    _second_order_spm(u, model::PhysicsModel, z)

Coupled χ⁽²⁾ nonlinear operator for [`SecondOrderMedium`](@ref): SHG/SFG/DFG
coupling between the fundamental (`u[:,1]`) and second-harmonic (`u[:,2]`)
envelopes.

```math
\\left.\\frac{\\partial A_1}{\\partial z}\\right|_{NL} = i\\kappa(z)\\, A_2 A_1^{*}\\, e^{-i\\Delta k_0 z}
\\qquad
\\left.\\frac{\\partial A_2}{\\partial z}\\right|_{NL} = i\\kappa(z)\\, A_1^{2}\\, e^{+i\\Delta k_0 z}
```

Using the *same* real `κ(z)` in both equations makes this exactly power-conserving
(`d(P₁+P₂)/dz = 0`) in the lossless case — see the `SecondOrderMedium` docstring for
the derivation. `κ(z)` is a period-`poling_period` sign-flipping square wave when
quasi-phase-matching is enabled (`poling_period > 0`), else constant.
"""
function _second_order_spm(u::AbstractMatrix{ComplexF64}, model::PhysicsModel, z::Real)
    A1 = @view u[:, 1]
    A2 = @view u[:, 2]

    aux = model.aux_data
    kappa0 = aux.kappa
    deltak0 = aux.deltak0
    poling_period = aux.poling_period

    kappa_z = if poling_period > 0
        m = mod(z, poling_period)
        m < poling_period / 2 ? kappa0 : -kappa0
    else
        kappa0
    end

    ph = cis(-deltak0 * z)  # exp(-i*deltak0*z)

    @views @. model.buf_t1[:, 1] = 1.0im * kappa_z * A2 * conj(A1) * ph
    @views @. model.buf_t1[:, 2] = 1.0im * kappa_z * A1^2 * conj(ph)

    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    return model.buf_f1
end

"""
    build_physics_model(grid::Grid, params::SimParams{S, <:SecondOrderMedium}, [template::AbstractMatrix])

Construct PhysicsModel for coupled χ⁽²⁾ (fundamental + second-harmonic) propagation.

The fundamental branch dispersion is evaluated on `grid.V` (`= ω - ω₀`) directly;
the second-harmonic branch dispersion is evaluated on `grid.V .- grid.omega0`
(`= ω - 2ω₀`), since `dispersion_sh`'s Taylor coefficients are defined relative to
the second harmonic's own carrier `2ω₀`, not the grid's fundamental-referenced
`ω₀`. Both use the shared package convention that `TaylorDispersion` excludes β₀
(see [`SecondOrderMedium`](@ref)'s `deltak0` field for why that's supplied
separately).
"""
function build_physics_model(
    grid::Grid,
    params::SimParams{S, M},
    template::AbstractMatrix=zeros(ComplexF64, grid.N, 2),
) where {S, M <: SecondOrderMedium}
    medium = params.medium
    N = grid.N

    tmp = similar(template)
    to_freq = plan_ifft(tmp, 1; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp, 1; flags=FFTW.MEASURE)

    med_fund = Medium(medium.length, 0.0, medium.loss_fund, medium.dispersion_fund, medium.lambda0)
    med_sh = Medium(
        medium.length, 0.0, medium.loss_sh, medium.dispersion_sh, medium.lambda0 / 2
    )

    D_fund = fftshift(dispersion_operator(grid.V, med_fund, 0.0))
    D_sh = fftshift(dispersion_operator(grid.V .- grid.omega0, med_sh, 0.0))

    D_host = hcat(D_fund, D_sh)
    D = _to_device(template, D_host)

    buf_t1 = similar(template)
    buf_t2 = similar(template)
    buf_f1 = similar(template)
    buf_f2 = similar(template)

    W_host = fill(grid.omega0, N)
    W = _to_device(template, W_host)
    gamma_W = _to_device(template, W_host)

    return PhysicsModel(
        to_freq,
        to_time,
        D,
        0.0,           # gamma unused by _second_order_spm
        grid.omega0,
        gamma_W,
        W,
        grid.dt,
        N,
        0.0,
        nothing,
        _second_order_spm,
        buf_t1,
        buf_t2,
        buf_f1,
        buf_f2,
        (kappa=Float64(medium.kappa), deltak0=Float64(medium.deltak0),
            poling_period=Float64(medium.poling_period)),
    )
end
