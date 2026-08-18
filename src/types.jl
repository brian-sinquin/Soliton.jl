"""
Types and structures for GNLSE in natural SI units.

Adapted from: gnlse-python (https://github.com/WUST-FOG/gnlse-python)
"""

"""
    DispersionModel

Abstract base type for chromatic-dispersion models. A model maps the relative
angular-frequency grid `V = ω - ω₀` [rad/s] to the propagation-constant
deviation `B(V)` [1/m] used in the dispersion operator `D = iB - α/2`.

Concrete models: [`TaylorDispersion`](@ref), [`TabulatedDispersion`](@ref).
"""
abstract type DispersionModel end

"""
    TaylorDispersion(betas)

Dispersion from a Taylor expansion of the propagation constant about ω₀:

    B(V) = Σ βₙ / n! · Vⁿ ,   n ≥ 2

`betas[1] = β₂` [s²/m], `betas[2] = β₃` [s³/m], … (β₀ and β₁ are excluded).

Parametric in the coefficient element type `T<:Real` (default `Float64`) so
that `betas`/`beta1` may hold AD dual numbers (e.g. `ForwardDiff.Dual`) for
gradient-based dispersion-profile optimization, in addition to plain
`Float64`.
"""
struct TaylorDispersion{T <: Real} <: DispersionModel
    betas::Vector{T}
    beta1::T
end

# An empty `betas` vector means no dispersion (pure SPM).
function TaylorDispersion(betas::AbstractVector{<:Real}, beta1::Real=0.0)
    T = promote_type(eltype(betas), typeof(beta1), Float64)
    TaylorDispersion{T}(collect(T, betas), T(beta1))
end

"""
    TabulatedDispersion(detuning, beta)

Dispersion from a measured/tabulated curve. `detuning` is the relative angular
frequency ω - ω₀ [rad/s] (sorted ascending); `beta` is the corresponding
propagation-constant deviation `B` [1/m] in the co-moving frame. Values are
linearly interpolated onto the simulation grid; outside the tabulated range the
nearest endpoint is held (flat extrapolation).

Parametric in the sample element type `T<:Real` (default `Float64`), so
`beta` may hold AD dual numbers for gradient-based fitting of a tabulated
dispersion curve.
"""
struct TabulatedDispersion{T <: Real} <: DispersionModel
    detuning::Vector{T}
    beta::Vector{T}

    # Explicit inner constructor: defining one here suppresses Julia's
    # auto-generated default `TabulatedDispersion(::Vector{T}, ::Vector{T}) where T`,
    # which would otherwise be more specific than the validating outer
    # constructor below for same-typed vector arguments and silently skip
    # these checks.
    function TabulatedDispersion{T}(detuning::Vector{T}, beta::Vector{T}) where {T <: Real}
        length(detuning) == length(beta) ||
            throw(ArgumentError("detuning and beta must have equal length"))
        length(detuning) >= 2 || throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) || throw(ArgumentError("detuning must be sorted ascending"))
        new{T}(detuning, beta)
    end
end

function TabulatedDispersion(detuning::AbstractVector{<:Real}, beta::AbstractVector{<:Real})
    T = promote_type(eltype(detuning), eltype(beta), Float64)
    TabulatedDispersion{T}(collect(T, detuning), collect(T, beta))
end

"""
    SellmeierDispersion(B, C)

Dispersion computed directly from Sellmeier coefficients:

    n²(λ) = 1 + Σ Bᵢ · λ² / (λ² - Cᵢ)

`B` is a vector of dimensionless coefficients, and `C` is a vector of resonance wavelengths squared (usually in μm²).

# Constructors

```julia
SellmeierDispersion(B, C; microns=true)
```

If `microns` is `true` (default), the `C` coefficients are assumed to be in μm² and will be converted to m² for natural SI unit calculations.

Parametric in the coefficient element type `T<:Real` (default `Float64`), so
`B`/`C` may hold AD dual numbers for gradient-based Sellmeier-coefficient
fitting.
"""
struct SellmeierDispersion{T <: Real} <: DispersionModel
    B::Vector{T}
    C::Vector{T}

    # Explicit inner constructor: see the note on TabulatedDispersion above —
    # this suppresses the auto-generated default that would otherwise bypass
    # the length check for same-typed vector arguments.
    function SellmeierDispersion{T}(B::Vector{T}, C::Vector{T}) where {T <: Real}
        length(B) == length(C) || throw(ArgumentError("B and C must have equal length"))
        new{T}(B, C)
    end
end

function SellmeierDispersion(
    B::AbstractVector{<:Real}, C::AbstractVector{<:Real}; microns::Bool=true
)
    T = promote_type(eltype(B), eltype(C), Float64)
    C_val = microns ? collect(T, C) .* T(1e-12) : collect(T, C) # 1 μm² = 1e-12 m²
    SellmeierDispersion{T}(collect(T, B), C_val)
end

"""
    AbstractComponent

Abstract base type for all optical components (fibers, amplifiers, filters, etc.).
"""
abstract type AbstractComponent end

"""
    Medium{T<:Real}

Fiber medium parameters for GNLSE propagation.

# Fields

  - `length::T`: Propagation length [m]
  - `gamma::T`: Nonlinear coefficient [1/(W·m)]
  - `loss::T`: Loss factor [dB/m]
  - `dispersion::DispersionModel`: Chromatic-dispersion model
  - `lambda0::T`: Center wavelength [m]

# Constructors

```julia
Medium(length, gamma, loss, betas::AbstractVector, lambda0)   # Taylor βₙ
Medium(length, gamma, loss, dispersion::DispersionModel, lambda0)
Medium(; length, gamma, loss=0.0, betas=(…), dispersion=(…), lambda0)
```

# Notes

  - Taylor `betas` exclude β₀ and β₁; `betas[1] = β₂`
  - loss in dB/m converted to α = ln(10^(loss/10)) Np/m in the operator
"""
abstract type AbstractMedium <: AbstractComponent end

struct Medium{T <: Real, TG, TL, D <: DispersionModel} <: AbstractMedium
    length::T
    gamma::TG
    loss::TL
    dispersion::D
    lambda0::T

    function Medium{T, TG, TL, D}(
        length::T, gamma::TG, loss::TL, dispersion::D, lambda0::T
    ) where {T <: Real, TG, TL, D <: DispersionModel}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        if TG <: Real
            gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        end
        if TL <: Real
            loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        end
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))

        new{T, TG, TL, D}(length, gamma, loss, dispersion, lambda0)
    end
end

# Outer constructors
function Medium(
    length::Real, gamma::Any, loss::Any, dispersion::DispersionModel, lambda0::Real
)
    T = promote_type(typeof(length), typeof(lambda0), Float64)
    loss_val = loss isa Real ? T(loss) : loss
    return Medium{T, typeof(gamma), typeof(loss_val), typeof(dispersion)}(
        T(length), gamma, loss_val, dispersion, T(lambda0)
    )
end

function Medium(
    length::Real, gamma::Any, loss::Any, betas::AbstractVector{<:Real}, lambda0::Real
)
    return Medium(length, gamma, loss, TaylorDispersion(betas), lambda0)
end

"""
    Medium(; length, gamma, loss=0.0, betas=nothing, dispersion=nothing, lambda0)

Keyword constructor — avoids positional-argument mistakes. Supply the dispersion
either as Taylor `betas` or as an explicit `dispersion::DispersionModel` (exactly
one of the two).
"""
function Medium(;
    length::Real,
    gamma::Any,
    loss::Any=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(typeof(length), typeof(lambda0), Float64)
    loss_val = loss isa Real ? T(loss) : loss
    Medium{T, typeof(gamma), typeof(loss_val), typeof(disp)}(
        T(length), gamma, loss_val, disp, T(lambda0)
    )
end

"""
    AmplifyingMedium(length, gamma, g0, Esat, noise_figure_db, loss, dispersion, lambda0)

Active rare-earth-doped amplifying fiber medium (EDFA, YDFA, TDFA) with gain saturation and ASE noise.

# Fields

  - `length::T`: Fiber length [m]
  - `gamma::TG`: Nonlinear coefficient [1/(W·m)]
  - `g0::GD`: Small-signal gain [1/m] (dB/m converted to Np/m: `g_Np = ln(10^(g_dB/10))`, or vector/function)
  - `Esat::T`: Saturation energy [J]
  - `noise_figure_db::T`: Amplifier Noise Figure [dB] (default: 4.0 dB)
  - `loss::TL`: Background loss factor [dB/m] (scalar, vector, or function)
  - `dispersion::DispersionModel`: Chromatic dispersion model
  - `lambda0::T`: Center wavelength [m]
"""
struct AmplifyingMedium{T <: Real, TG, GD, TL} <: AbstractMedium
    length::T
    gamma::TG
    g0::GD
    Esat::T
    noise_figure_db::T
    loss::TL
    dispersion::DispersionModel
    lambda0::T

    function AmplifyingMedium(
        length::T,
        gamma::TG,
        g0::GD,
        Esat::T,
        noise_figure_db::T,
        loss::TL,
        dispersion::DispersionModel,
        lambda0::T,
    ) where {T <: Real, TG, GD, TL}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        Esat > 0 || throw(ArgumentError("Saturation energy Esat must be positive"))
        noise_figure_db >= 0 || throw(ArgumentError("Noise Figure must be non-negative"))
        if TL <: Real
            loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        end
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T, TG, GD, TL}(
            length, gamma, g0, Esat, noise_figure_db, loss, dispersion, lambda0
        )
    end
end

function AmplifyingMedium(;
    length::Real,
    gamma::Any,
    g0::Any=nothing,
    g0_db::Union{Real, AbstractVector{<:Real}, Nothing}=nothing,
    Esat::Real,
    noise_figure_db::Real=4.0,
    loss::Any=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    (g0 === nothing) ⊻ (g0_db === nothing) || throw(
        ArgumentError("provide concentration/gain via exactly one of `g0` or `g0_db`")
    )
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    gain_val = if g0_db !== nothing
        (
        if g0_db isa Real
            Float64(g0_db) * (log(10.0) / 10.0)
        else
            Float64.(g0_db) .* (log(10.0) / 10.0)
        end
    )
    else
        g0
    end
    T = promote_type(
        typeof(length), typeof(Esat), typeof(noise_figure_db), typeof(lambda0), Float64
    )
    loss_val = loss isa Real ? T(loss) : loss
    return AmplifyingMedium(
        T(length), gamma, gain_val, T(Esat), T(noise_figure_db), loss_val, disp, T(lambda0)
    )
end

"""
    SemiconductorMedium(; length, gamma, alpha2, Aeff, sigma_fca, k_fcr, tau_c, loss, dispersion, lambda0)

Semiconductor waveguide medium (Silicon-on-Insulator SOI, Germanium, GaAs) with Two-Photon Absorption (TPA) and Free-Carrier Dynamics (FCA, FCR, lifetime τ_c).

# Fields

  - `length::T`: Waveguide length [m]
  - `gamma::TG`: Kerr nonlinear coefficient [1/(W·m)]
  - `alpha2::T`: Two-photon absorption coefficient α₂ [m/W]
  - `Aeff::T`: Effective mode area A_eff [m²]
  - `sigma_fca::T`: Free-carrier absorption cross section σ_FCA [m²] (default: 1.45e-21 m² for Si at 1550 nm)
  - `k_fcr::T`: Free-carrier refraction coefficient k_FCR [m³] (default: 5.3e-27 m³ for Si at 1550 nm)
  - `tau_c::T`: Free-carrier recombination lifetime τ_c [s] (default: 1.0e-9 s)
  - `loss::T`: Linear background attenuation α₀ [dB/m]
  - `dispersion::DispersionModel`: Chromatic dispersion model
  - `lambda0::T`: Center wavelength [m]
"""
struct SemiconductorMedium{T <: Real, TG} <: AbstractMedium
    length::T
    gamma::TG
    alpha2::T
    Aeff::T
    sigma_fca::T
    k_fcr::T
    tau_c::T
    loss::T
    dispersion::DispersionModel
    lambda0::T

    function SemiconductorMedium(
        length::T,
        gamma::TG,
        alpha2::T,
        Aeff::T,
        sigma_fca::T,
        k_fcr::T,
        tau_c::T,
        loss::T,
        dispersion::DispersionModel,
        lambda0::T,
    ) where {T <: Real, TG}
        length > 0 || throw(ArgumentError("Waveguide length must be positive"))
        alpha2 >= 0 || throw(ArgumentError("TPA coefficient alpha2 must be non-negative"))
        Aeff > 0 || throw(ArgumentError("Effective area Aeff must be positive"))
        tau_c > 0 || throw(ArgumentError("Carrier lifetime tau_c must be positive"))
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T, TG}(
            length, gamma, alpha2, Aeff, sigma_fca, k_fcr, tau_c, loss, dispersion, lambda0
        )
    end
end

function SemiconductorMedium(;
    length::Real,
    gamma::Any,
    alpha2::Real,
    Aeff::Real,
    sigma_fca::Real=1.45e-21,
    k_fcr::Real=5.3e-27,
    tau_c::Real=1.0e-9,
    loss::Real=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(
        typeof(length),
        typeof(alpha2),
        typeof(Aeff),
        typeof(sigma_fca),
        typeof(k_fcr),
        typeof(tau_c),
        typeof(loss),
        typeof(lambda0),
        Float64,
    )
    return SemiconductorMedium(
        T(length),
        gamma,
        T(alpha2),
        T(Aeff),
        T(sigma_fca),
        T(k_fcr),
        T(tau_c),
        T(loss),
        disp,
        T(lambda0),
    )
end

"""
    RamanModel

Abstract base type for Raman response models.
"""
abstract type RamanModel end

"""
    BlowWood <: RamanModel

Single Lorentzian Raman response model from K. J. Blow & D. Wood.

Parameters (SI units):

  - fr = 0.18: Raman fraction
  - τ₁ = 12.2 fs
  - τ₂ = 32 fs

Reference: K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989)
"""
struct BlowWood <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64

    BlowWood(; fr::Float64=0.18, tau1::Float64=12.2e-15, tau2::Float64=32.0e-15) =
        new(fr, tau1, tau2)
end

"""
    LinAgrawal <: RamanModel

Three-component Raman model from Q. Lin & G. P. Agrawal.

Parameters (SI units):

  - fr = 0.245: Raman fraction
  - τ₁ = 12.2 fs
  - τ₂ = 32 fs
  - τb = 96 fs
  - fb = 0.21
  - fc = 0.04

Reference: Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006)
"""
struct LinAgrawal <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64
    taub::Float64
    fb::Float64
    fc::Float64

    LinAgrawal(;
        fr::Float64=0.245,
        tau1::Float64=12.2e-15,
        tau2::Float64=32.0e-15,
        taub::Float64=96.0e-15,
        fb::Float64=0.21,
        fc::Float64=0.04,
    ) = new(fr, tau1, tau2, taub, fb, fc)
end

"""
    Hollenbeck <: RamanModel

13-oscillator Raman model from D. Hollenbeck & C. D. Cantrell.

Parameters:

  - fr = 0.20: Raman fraction

Reference: D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002)
"""
struct Hollenbeck <: RamanModel
    fr::Float64

    Hollenbeck(; fr::Float64=0.20) = new(fr)
end

"""
    MolecularRamanGas(gas::Symbol)

Raman response model for molecular gases (H₂, N₂) in hollow-core fibers.

Supported gases:

  - `:H2_rotational`: Rotational S(1) Raman line (Δν_R = 17.6 THz, τ₁ = 9 fs, τ₂ = 100 ps, f_R = 0.12)
  - `:H2_vibrational`: Vibrational Q(1) Raman line (Δν_R = 124.6 THz, τ₁ = 1.28 fs, τ₂ = 100 ps, f_R = 0.08)
  - `:N2`: Molecular nitrogen Raman line (Δν_R = 2.2 THz, τ₁ = 72 fs, τ₂ = 50 ps, f_R = 0.10)
"""
struct MolecularRamanGas <: RamanModel
    gas::Symbol
    fr::Float64
    tau1::Float64
    tau2::Float64
end

function MolecularRamanGas(gas::Symbol)
    if gas === :H2_rotational
        return MolecularRamanGas(:H2_rotational, 0.12, 9.0e-15, 100.0e-12)
    elseif gas === :H2_vibrational
        return MolecularRamanGas(:H2_vibrational, 0.08, 1.28e-15, 100.0e-12)
    elseif gas === :N2
        return MolecularRamanGas(:N2, 0.10, 72.0e-15, 50.0e-12)
    else
        throw(
            ArgumentError(
                "Unsupported gas '$gas'. Supported: :H2_rotational, :H2_vibrational, :N2"
            ),
        )
    end
end

"""
    NonlinearityModel

Abstract base type for nonlinearity models.
"""
abstract type NonlinearityModel end

"""
    ConstantNonlinearity(gamma)

Constant nonlinear coefficient γ [1/(W·m)].
"""
struct ConstantNonlinearity <: NonlinearityModel
    gamma::Float64
end

"""
    FrequencyDependentNonlinearity(gamma_func)

Frequency-dependent nonlinear coefficient where `gamma_func(w)` takes absolute angular frequency w [rad/s]
and returns γ [1/(W·m)].
"""
struct FrequencyDependentNonlinearity{F} <: NonlinearityModel
    gamma_function::F
end

"""
    NonlinearityFromEffectiveArea(n2, Aeff_func)

Nonlinearity calculated from nonlinear index n₂ [m²/W] and a frequency-dependent effective mode area.
`Aeff_func` takes absolute frequency w [rad/s] and returns mode area A_eff [m²].
"""
struct NonlinearityFromEffectiveArea{F} <: NonlinearityModel
    n2::Float64
    Aeff_function::F
end

"""
    Grid{T<:Real}

Time and frequency grid for GNLSE propagation.

# Fields

  - `N::Int`: Number of grid points (resolution)
  - `t::Vector{T}`: Time grid [s], centered at zero
  - `V::Vector{T}`: Relative angular frequency grid ω - ω₀ [rad/s], monotonic
  - `W::Vector{T}`: Absolute optical angular frequency ω = ω₀ + V [rad/s], monotonic
  - `dt::T`: Time step [s]
  - `omega0::T`: Central angular frequency ω₀ [rad/s]
  - `lambda0::T`: Center wavelength λ₀ [m]

# Notes

  - V = ω - ω₀ is the physical detuning; W = ω₀ + V is the absolute frequency
  - ω₀ = 2πc/λ₀ (all frequencies in rad/s)
  - V and W are stored in monotonic order, not FFT-natural order
"""
struct Grid{T <: Real}
    N::Int
    t::Vector{T}
    V::Vector{T}          # Relative angular frequency ω - ω₀ [rad/s], monotonic
    W::Vector{T}          # Absolute angular frequency ω₀ + V [rad/s], monotonic
    dt::T
    omega0::T             # Central angular frequency [rad/s]
    lambda0::T            # Center wavelength [m]
end

"""
    AbstractPulse

Abstract base type for optical pulses.
"""
abstract type AbstractPulse end

"""
    Pulse{T<:Complex}

Optical pulse envelope in time and frequency domains.

# Fields

  - `At::Vector{T}`: Time domain envelope A(t) [√W]
  - `AW::Vector{T}`: Frequency domain envelope A(ω) [√W·s]
  - `grid::Grid`: Associated time-frequency grid

# Notes

Following gnlse-python convention:

  - AW = N * ifft(At) (note: inverted FFT convention)
  - At = fft(AW)
  - Power: P(t) = |A(t)|²
  - Energy: E = ∫|A(t)|²dt
"""
mutable struct Pulse{T <: Complex} <: AbstractPulse
    At::Vector{T}
    AW::Vector{T}
    grid::Grid
end

"""
    SimParams

Simulation parameters for GNLSE propagation.

# Fields

  - `medium::Medium`: Fiber medium parameters
  - `z_saves::Int`: Number of snapshots to save along fiber
  - `raman_model::Union{RamanModel,Nothing}`: Raman scattering model (Nothing to disable)
  - `self_steepening::Bool`: Enable self-steepening/shock effect
  - `solver::GNLSESolver`: Numerical solver (default: `ERK4IP()`). Tolerances are
    set on the solver, e.g. `ERK4IP(; rtol=1e-6, atol=1e-8)`.

# Notes

Following gnlse-python's GNLSESetup structure:

  - z_saves determines memory allocation for result arrays
  - Raman disabled by setting raman_model = nothing or fr = 0
  - Self-steepening modifies W frequency grid
"""
struct SimParams{S <: GNLSESolver, M <: AbstractMedium}
    medium::M
    z_saves::Int
    raman_model::Union{RamanModel, Nothing}
    self_steepening::Bool
    solver::S
    save_freq::Bool

    function SimParams(
        medium::M,
        z_saves::Int,
        raman_model::Union{RamanModel, Nothing},
        self_steepening::Bool,
        solver::S,
        save_freq::Bool=true,
    ) where {S <: GNLSESolver, M <: AbstractMedium}
        z_saves >= 2 || throw(
            ArgumentError(
                "z_saves must be >= 2 (need at least start and end points; " *
                "got z_saves=$z_saves)",
            ),
        )
        new{S, M}(medium, z_saves, raman_model, self_steepening, solver, save_freq)
    end
end

# Backward-compatible positional constructor
function SimParams(
    medium::AbstractMedium,
    z_saves::Int,
    raman_model::Union{RamanModel, Nothing},
    self_steepening::Bool,
    rtol::Float64,
    atol::Float64,
)
    return SimParams(
        medium, z_saves, raman_model, self_steepening, ERK4IP(; rtol=rtol, atol=atol), true
    )
end

# Keyword constructor supporting both generic solver and backward-compatible rtol/atol
function SimParams(;
    medium::AbstractMedium,
    z_saves::Int=200,
    raman_model::Union{RamanModel, Nothing}=BlowWood(),
    self_steepening::Bool=false,
    rtol::Union{Float64, Nothing}=nothing,
    atol::Union{Float64, Nothing}=nothing,
    solver::Union{GNLSESolver, Nothing}=nothing,
    save_freq::Bool=true,
)
    if solver === nothing
        rtol_val = rtol === nothing ? 1e-6 : rtol
        atol_val = atol === nothing ? 1e-8 : atol
        solv = ERK4IP(; rtol=rtol_val, atol=atol_val)
    else
        (rtol === nothing && atol === nothing) || throw(
            ArgumentError(
                "Cannot specify both `solver` and `rtol`/`atol` compatibility options"
            ),
        )
        solv = solver
    end
    return SimParams(medium, z_saves, raman_model, self_steepening, solv, save_freq)
end

"""
    Solution{T<:Complex}

Solution to GNLSE propagation.

# Fields

  - `t::Vector{Float64}`: Time domain grid [s]
  - `W::Vector{Float64}`: Absolute angular frequency grid [rad/s]
  - `omega0::Float64`: Central angular frequency [rad/s]
  - `Z::Vector{Float64}`: Propagation distances [m]
  - `At::Matrix{T}`: Time domain solution (N × z_saves)
  - `AW::Matrix{T}`: Frequency domain solution (N × z_saves)

# Notes

Following gnlse-python's Solution structure:

  - At[i, j] = pulse at time t[i], distance Z[j]
  - AW[i, j] = frequency component at W[i], distance Z[j]
"""
struct Solution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Matrix{T}
    AW::Matrix{T}
end

# ---------------------------------------------------------------------------
# Compact REPL display
# ---------------------------------------------------------------------------

"""
    show(io::IO, m::TaylorDispersion)

Display a TaylorDispersion model compactly in the REPL, showing the number of
Taylor coefficients (β₂, β₃, …) included in the dispersion expansion.
"""
Base.show(io::IO, m::TaylorDispersion) =
    print(io, "TaylorDispersion(", length(m.betas), " terms from β₂)")

"""
    show(io::IO, m::TabulatedDispersion)

Display a TabulatedDispersion model compactly in the REPL, showing the number
of frequency samples over which the propagation constant is tabulated.
"""
Base.show(io::IO, m::TabulatedDispersion) =
    print(io, "TabulatedDispersion(", length(m.detuning), " samples)")

"""
    show(io::IO, m::Medium)

Display a Medium (fiber) specification compactly in the REPL. Shows:

  - L: fiber length [m]
  - γ: nonlinear coefficient [1/W/m]
  - loss: attenuation [dB/m]
  - λ₀: center wavelength [nm]
  - dispersion model type
"""
function Base.show(io::IO, m::Medium)
    print(
        io,
        "Medium(L=",
        m.length,
        " m, γ=",
        m.gamma,
        " /W/m, loss=",
        m.loss,
        " dB/m, λ₀=",
        round(m.lambda0 * 1e9; digits=2),
        " nm, ",
        m.dispersion,
        ")",
    )
end

"""
    show(io::IO, g::Grid)

Display a simulation grid compactly in the REPL. Shows:

  - N: number of grid points
  - window: temporal duration [ps]
  - λ₀: center wavelength [nm]
  - dt: time resolution [fs]
"""
function Base.show(io::IO, g::Grid)
    print(
        io,
        "Grid(N=",
        g.N,
        ", window=",
        round((g.t[end] - g.t[1]) * 1e12; digits=3),
        " ps, λ₀=",
        round(g.lambda0 * 1e9; digits=2),
        " nm, dt=",
        round(g.dt * 1e15; digits=3),
        " fs)",
    )
end

"""
    show(io::IO, p::Pulse)

Display a Pulse compactly in the REPL. Shows:

  - N: grid resolution
  - peak: peak intensity [W]
"""
function Base.show(io::IO, p::Pulse)
    Pmax = maximum(abs2, p.At)
    print(io, "Pulse(N=", p.grid.N, ", peak=", round(Pmax; sigdigits=4), " W)")
end

"""
    show(io::IO, s::SimParams)

Display simulation parameters compactly in the REPL. Shows:

  - z_saves: number of snapshots along fiber
  - raman: Raman model type or "off"
  - self_steepening: whether shock term is active
  - solver: solver type with tolerances (if ERK4IP)
"""
function Base.show(io::IO, s::SimParams)
    solver_str = if s.solver isa ERK4IP
        "ERK4IP(rtol=$(s.solver.rtol), atol=$(s.solver.atol))"
    else
        string(nameof(typeof(s.solver)))
    end
    print(
        io,
        "SimParams(z_saves=",
        s.z_saves,
        ", raman=",
        s.raman_model === nothing ? "off" : nameof(typeof(s.raman_model)),
        ", self_steepening=",
        s.self_steepening,
        ", solver=",
        solver_str,
        ")",
    )
end

"""
    show(io::IO, s::Solution)

Display a Solution compactly in the REPL. Shows:

  - number of saved distances
  - propagation distance span [m]
  - grid resolution [points]
"""
function Base.show(io::IO, s::Solution)
    print(
        io,
        "Solution(",
        length(s.Z),
        " saves over ",
        round(s.Z[end]; sigdigits=4),
        " m, N=",
        size(s.At, 1),
        ")",
    )
end

"""
    BirefringentMedium(length, gamma, loss, dispersion_x, dispersion_y, deltabeta0, lambda0)

Birefringent fiber medium parameters for coupled vector GNLSE propagation.

# Fields

  - `length::T`: Propagation length [m]
  - `gamma::TG`: Nonlinear coefficient γ [1/(W·m)]
  - `loss::T`: Loss factor [dB/m]
  - `dispersion_x::DispersionModel`: Dispersion model for x-polarization
  - `dispersion_y::DispersionModel`: Dispersion model for y-polarization
  - `deltabeta0::T`: Modal birefringence Δβ₀ [1/m] (difference in propagation constants β₀,x - β₀,y)
  - `lambda0::T`: Center wavelength [m]
"""
struct BirefringentMedium{T <: Real, TG} <: AbstractMedium
    length::T
    gamma::TG
    loss::T
    dispersion_x::DispersionModel
    dispersion_y::DispersionModel
    deltabeta0::T
    lambda0::T

    function BirefringentMedium(
        length::T,
        gamma::TG,
        loss::T,
        dispersion_x::DispersionModel,
        dispersion_y::DispersionModel,
        deltabeta0::T,
        lambda0::T,
    ) where {T <: Real, TG}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        if TG <: Real
            gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        end
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T, TG}(length, gamma, loss, dispersion_x, dispersion_y, deltabeta0, lambda0)
    end
end

"""
    VectorialPulse(At, grid)
    VectorialPulse(At_x, At_y, grid)

Two-component (orthogonal polarization) optical pulse envelope in time and frequency domains.

# Fields

  - `At::Matrix{ComplexF64}`: Time domain envelope matrix of size `N × 2` [√W]
  - `AW::Matrix{ComplexF64}`: Frequency domain envelope matrix of size `N × 2` [√W·s]
  - `grid::Grid`: Associated time-frequency grid
"""
struct VectorialPulse <: AbstractPulse
    At::Matrix{ComplexF64} # N x 2
    AW::Matrix{ComplexF64} # N x 2
    grid::Grid

    function VectorialPulse(At::Matrix{<:Complex}, AW::Matrix{<:Complex}, grid::Grid)
        size(At, 1) == grid.N ||
            throw(ArgumentError("Pulse dimensions must match grid resolution"))
        size(At, 2) == 2 ||
            throw(ArgumentError("Vectorial pulse must have exactly 2 components (x and y)"))
        size(AW) == size(At) ||
            throw(ArgumentError("AW dimensions must match At dimensions"))
        new(Matrix{ComplexF64}(At), Matrix{ComplexF64}(AW), grid)
    end

    function VectorialPulse(At::Matrix{<:Complex}, grid::Grid)
        size(At, 1) == grid.N ||
            throw(ArgumentError("Pulse dimensions must match grid resolution"))
        size(At, 2) == 2 ||
            throw(ArgumentError("Vectorial pulse must have exactly 2 components (x and y)"))
        AW = similar(At)
        AW[:, 1] .= ifft(At[:, 1])
        AW[:, 2] .= ifft(At[:, 2])
        new(Matrix{ComplexF64}(At), AW, grid)
    end
end

function VectorialPulse(
    At_x::AbstractVector{<:Number}, At_y::AbstractVector{<:Number}, grid::Grid
)
    length(At_x) == grid.N ||
        throw(ArgumentError("x component length must match grid resolution"))
    length(At_y) == grid.N ||
        throw(ArgumentError("y component length must match grid resolution"))
    At = Matrix{ComplexF64}(undef, grid.N, 2)
    At[:, 1] .= At_x
    At[:, 2] .= At_y
    return VectorialPulse(At, grid)
end

"""
    VectorialSolution{T<:Complex}

Solution to coupled vector GNLSE propagation.

# Fields

  - `t::Vector{Float64}`: Time domain grid [s]
  - `W::Vector{Float64}`: Absolute angular frequency grid [rad/s]
  - `omega0::Float64`: Central angular frequency [rad/s]
  - `Z::Vector{Float64}`: Propagation distances [m]
  - `At::Array{T, 3}`: Time domain solution array of size `N × 2 × z_saves`
  - `AW::Array{T, 3}`: Frequency domain solution array of size `N × 2 × z_saves`
"""
struct VectorialSolution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Array{T, 3} # N x 2 x z_saves
    AW::Array{T, 3} # N x 2 x z_saves
end

# Base.show definitions for Vectorial types
function Base.show(io::IO, p::VectorialPulse)
    Pmax_x = maximum(abs2, p.At[:, 1])
    Pmax_y = maximum(abs2, p.At[:, 2])
    print(
        io,
        "VectorialPulse(N=",
        p.grid.N,
        ", peak_x=",
        round(Pmax_x; sigdigits=4),
        " W, peak_y=",
        round(Pmax_y; sigdigits=4),
        " W)",
    )
end

function Base.show(io::IO, s::VectorialSolution)
    print(
        io,
        "VectorialSolution(",
        length(s.Z),
        " saves over ",
        round(s.Z[end]; sigdigits=4),
        " m, N=",
        size(s.At, 1),
        ")",
    )
end
