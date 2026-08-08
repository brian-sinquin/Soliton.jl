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
"""
struct TaylorDispersion <: DispersionModel
    betas::Vector{Float64}
    beta1::Float64
end

# An empty `betas` vector means no dispersion (pure SPM).
TaylorDispersion(betas::AbstractVector{<:Real}, beta1::Real=0.0) =
    TaylorDispersion(collect(Float64, betas), Float64(beta1))

"""
    TabulatedDispersion(detuning, beta)

Dispersion from a measured/tabulated curve. `detuning` is the relative angular
frequency ω - ω₀ [rad/s] (sorted ascending); `beta` is the corresponding
propagation-constant deviation `B` [1/m] in the co-moving frame. Values are
linearly interpolated onto the simulation grid; outside the tabulated range the
nearest endpoint is held (flat extrapolation).
"""
struct TabulatedDispersion <: DispersionModel
    detuning::Vector{Float64}
    beta::Vector{Float64}

    function TabulatedDispersion(
        detuning::AbstractVector{<:Real}, beta::AbstractVector{<:Real}
    )
        length(detuning) == length(beta) ||
            throw(ArgumentError("detuning and beta must have equal length"))
        length(detuning) >= 2 || throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) || throw(ArgumentError("detuning must be sorted ascending"))
        new(collect(Float64, detuning), collect(Float64, beta))
    end
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
"""
struct SellmeierDispersion <: DispersionModel
    B::Vector{Float64}
    C::Vector{Float64}

    function SellmeierDispersion(
        B::AbstractVector{<:Real}, C::AbstractVector{<:Real}; microns::Bool=true
    )
        length(B) == length(C) || throw(ArgumentError("B and C must have equal length"))
        C_val = microns ? collect(Float64, C) .* 1e-12 : collect(Float64, C) # 1 μm² = 1e-12 m²
        new(collect(Float64, B), C_val)
    end
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
    SemiconductorMedium(; length, gamma, alpha2, Aeff, sigma_fca, k_fcr, tau_c, alpha3, loss, dispersion, lambda0)

Semiconductor waveguide medium (Silicon-on-Insulator SOI, Germanium, GaAs) with Two-Photon Absorption (TPA), Three-Photon Absorption (3PA), and Free-Carrier Dynamics (FCA, FCR, lifetime τ_c).

3PA becomes the dominant nonlinear-absorption channel in the mid-infrared, where the
photon energy falls below half the bandgap (so TPA is forbidden) but three-photon
transitions remain allowed (e.g. Si/Ge/SiGe beyond ≈2.2 µm). TPA and 3PA free-carrier
generation are additive, so both may be enabled simultaneously; set `alpha3=0` (default)
to recover pure-TPA behavior.

# Fields

  - `length::T`: Waveguide length [m]
  - `gamma::TG`: Kerr nonlinear coefficient [1/(W·m)]
  - `alpha2::T`: Two-photon absorption coefficient α₂ [m/W]
  - `Aeff::T`: Effective mode area A_eff [m²]
  - `sigma_fca::T`: Free-carrier absorption cross section σ_FCA [m²] (default: 1.45e-21 m² for Si at 1550 nm)
  - `k_fcr::T`: Free-carrier refraction coefficient k_FCR [m³] (default: 5.3e-27 m³ for Si at 1550 nm)
  - `tau_c::T`: Free-carrier recombination lifetime τ_c [s] (default: 1.0e-9 s)
  - `alpha3::T`: Three-photon absorption coefficient α₃ [m³/W²] (default: 0.0, i.e. disabled)
  - `loss::T`: Linear background attenuation α₀ [dB/m]
  - `dispersion::DispersionModel`: Chromatic dispersion model
  - `lambda0::T`: Center wavelength [m]

# Known Limitations

This model follows the standard 1D lumped-parameter formalism (Lin, Painter & Agrawal
2007) and its usual simplifications. These are not bugs — they are the same
approximations essentially every published GNLSE-TPA/3PA model makes — but should be
kept in mind when comparing quantitatively against an experiment:

  - **Shared `Aeff` across nonlinear orders**: the Kerr, TPA, and 3PA terms all reuse
    the same modal effective area, i.e. `∫f²dA`, `∫f³dA/Aeff`, and `∫f⁴dA/Aeff²` are all
    approximated as `≈ Aeff`, rather than using their true (generally different)
    mode-overlap integrals. This approximation degrades faster for the higher-order 3PA
    term than for TPA, especially in narrow or highly multimode mid-IR waveguides.
  - **Linear recombination only**: carrier decay is purely `-N_c/τ_c`. Auger
    recombination (`∝ N_c³`) is not modeled, and can become significant above
    ~10¹⁸ cm⁻³ carrier densities — a regime that high-intensity mid-IR 3PA pumping can
    reach. Carrier diffusion out of the mode is also not modeled.
  - **No 3PA-associated nonlinear refraction**: by Kramers-Kronig, a nonzero `alpha3`
    implies an accompanying intensity-dependent real refractive-index correction
    (a five-wave-mixing-type term), which is not included here — consistent with how
    the TPA-associated refractive correction is also omitted, i.e. both are dropped at
    the same order.
  - **Self-steepening applies to Kerr/TPA/3PA only, not FCA/FCR**: the shock/
    self-steepening correction is physically tied to the frequency dependence of the
    bound-electron (χ⁽³⁾/χ⁽⁵⁾) polarization response near ω₀. Free-carrier plasma
    effects (FCA, FCR) are a separate, slowly-varying process and are propagated
    without shock weighting (see `_semiconductor_spm` in `src/nonlinearity.jl`).
  - **Partial independent package cross-validation**: TPA *is* validated against an
    external package — `gnlse-python`'s Kerr-only nonlinear step accepts a complex
    `nonlinearity` value (`γ_r + i·α₂/(2·Aeff)`), which reproduces the TPA field equation
    exactly (the standard "complex γ = Kerr + i·TPA" convention), letting its own
    independent adaptive solver (`scipy.solve_ivp`, not Soliton's ERK4IP/SSFM) serve as
    ground truth (`test/test_adversarial.jl`, Scenario 7). 3PA and FCA/FCR have no such
    trick available — 3PA needs a quintic field term (`|A|⁴A`) that `gnlse-python`'s
    cubic-only Kerr/Raman step cannot express, and FCA/FCR need an auxiliary
    carrier-density ODE it has no concept of. No other open package implementing this
    physics was found (searched: PyNLO, MEEP — no complex-χ³ support, unresolved since
    [GitHub issue #1099](https://github.com/NanoComp/meep/issues/1099) — and Tidy3D FDTD,
    which ships a `TwoPhotonAbsorption` model but no ≥3-photon or free-carrier model and
    is a paid cloud service, unsuitable for a local test suite). 3PA and FCA/FCR remain
    validated only via the dedicated ODE ground-truth integrator
    (`test/test_semiconductor.jl`).

# References

  - Q. Lin, J. Zhang, G. Piestun, R. Boyraz & G. P. Agrawal, "Nonlinear optical
    phenomena in silicon waveguides: modeling and applications," Opt. Express
    **15**, 16604 (2007) — TPA/FCA/FCR formalism this type extends.
  - Multi-photon absorption and third-order nonlinearity in silicon at
    mid-infrared wavelengths, Opt. Express **21**, 32192 (2013) — α₃
    coefficient for Si (≈2×10⁻³ cm³/GW² near 2.6 µm).
  - Impact of third-order dispersion and three-photon absorption on
    mid-infrared time magnification via four-wave mixing in Si₀.₈Ge₀.₂
    waveguides, Appl. Opt. **59**, 1187 (2020).
"""
struct SemiconductorMedium{T <: Real, TG} <: AbstractMedium
    length::T
    gamma::TG
    alpha2::T
    Aeff::T
    sigma_fca::T
    k_fcr::T
    tau_c::T
    alpha3::T
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
        alpha3::T,
        loss::T,
        dispersion::DispersionModel,
        lambda0::T,
    ) where {T <: Real, TG}
        length > 0 || throw(ArgumentError("Waveguide length must be positive"))
        alpha2 >= 0 || throw(ArgumentError("TPA coefficient alpha2 must be non-negative"))
        Aeff > 0 || throw(ArgumentError("Effective area Aeff must be positive"))
        tau_c > 0 || throw(ArgumentError("Carrier lifetime tau_c must be positive"))
        alpha3 >= 0 || throw(ArgumentError("3PA coefficient alpha3 must be non-negative"))
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T, TG}(
            length,
            gamma,
            alpha2,
            Aeff,
            sigma_fca,
            k_fcr,
            tau_c,
            alpha3,
            loss,
            dispersion,
            lambda0,
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
    alpha3::Real=0.0,
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
        typeof(alpha3),
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
        T(alpha3),
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
    CoupledPulse

Abstract base type for pulses carrying two coupled envelope components as an
`N × 2` matrix (`At`/`AW`) on a shared time-frequency grid — e.g.
[`VectorialPulse`](@ref) (orthogonal polarizations) or [`SecondOrderPulse`](@ref)
(fundamental + second harmonic). Solvers that operate generically on any
two-field coupled system (currently `SSFM`; see `src/solvers/ssfm_vectorial.jl`)
dispatch on this type rather than the two concrete subtypes individually.
"""
abstract type CoupledPulse <: AbstractPulse end

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
struct VectorialPulse <: CoupledPulse
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

# ===========================================================================
# Second-Order (χ⁽²⁾) Nonlinearity Support
# ===========================================================================

"""
    SecondOrderMedium(; length, kappa, betas_fund, betas_sh, deltak0, poling_period=0.0,
                     loss_fund=0.0, loss_sh=0.0, lambda0)

Coupled-envelope χ⁽²⁾ (second-order) nonlinear medium for second-harmonic
generation (SHG), sum-/difference-frequency generation, and (degenerate) optical
parametric amplification. Propagates two envelopes on the same time grid: the
fundamental `A₁` at ω₀ and the second harmonic `A₂` at 2ω₀ (Phase 1 of the
second-order-nonlinearity roadmap — see `ROADMAP.md`; this is the classical
two-envelope coupled-mode model, not the unified broadband χ⁽²⁾+χ⁽³⁾ model of
T. Voumard et al., *APL Photonics* **8**, 036114 (2023), planned for Phase 2).

# Fields

  - `length::T`: Waveguide length [m]
  - `kappa::T`: Normalized χ⁽²⁾ coupling coefficient κ [1/(√W·m)]. This is the
    standard normalized SHG/SFG coupling constant (see e.g. Suhara & Fujimura,
    *Waveguide Nonlinear-Optic Devices*): `κ = (ω₀·d_eff)/(n·c) · √(2/(ε₀·c·n·Aeff))`
    up to the convention-dependent prefactor of the source formula — pass the
    value directly rather than re-deriving `d_eff`/`Aeff`/`n` internally, exactly
    as `gamma` is a direct lumped coefficient for Kerr media.
  - `dispersion_fund::DispersionModel`: Dispersion model for the fundamental
    branch, expanded around ω₀ (its own `β₁, β₂, …`, no `β₀` term — see `deltak0`).
  - `dispersion_sh::DispersionModel`: Dispersion model for the second-harmonic
    branch, expanded around 2ω₀.
  - `deltak0::T`: Phase mismatch Δk₀ = β₀(2ω₀) − 2β₀(ω₀) [1/m]. Not derivable from
    `dispersion_fund`/`dispersion_sh` (which intentionally omit β₀, following the
    rest of the package's interaction-picture convention) — supply it directly,
    exactly as `BirefringentMedium`'s `deltabeta0` is supplied directly.
  - `poling_period::T`: Periodic-poling (QPM) period [m]. `0.0` (default) disables
    poling (uniform `κ`, appropriate for birefringent- or naturally-phase-matched
    devices); a positive value applies a period-`poling_period` sign-flipping
    square-wave modulation to `κ(z)`.
  - `loss_fund::T`, `loss_sh::T`: Linear attenuation [dB/m] for each branch.
  - `lambda0::T`: Fundamental center wavelength [m] (the second harmonic is at
    `lambda0/2`).

# Physics

```math
\\frac{\\partial A_1}{\\partial z} = i\\hat D_1 A_1 + i\\kappa(z)\\, A_2 A_1^{*}\\, e^{-i\\Delta k_0 z}
```
```math
\\frac{\\partial A_2}{\\partial z} = i\\hat D_2 A_2 + i\\kappa(z)\\, A_1^{2}\\, e^{+i\\Delta k_0 z}
```

with `κ` real and identical in both equations, which by construction conserves
total power `P₁+P₂` exactly for `loss_fund=loss_sh=0` (Manley-Rowe / photon-number
conservation — 2 photons at ω₀ convert to 1 photon at 2ω₀, so no factor of 2 needed
between the two equations). At perfect phase matching (`Δk₀=0`, undepleted-pump
regime `|A₂| ≪ |A₁|`), this reduces to the classic `sinc²` SHG conversion curve;
in the fully depleted regime it reduces to the exact `tanh²(κ√P₁(0)·z)` SHG
conversion efficiency.

# Known Limitations (Phase 1)

  - χ⁽³⁾ (Kerr/Raman) is not modeled simultaneously on either branch — pure χ⁽²⁾
    only. Simultaneous χ⁽²⁾+χ⁽³⁾ (needed for cascaded-nonlinearity supercontinuum)
    is deferred to Phase 2.
  - Self-steepening is not applied to the χ⁽²⁾ coupling term.
  - QPM is modeled as an idealized square-wave `κ(z)`, not first-order-Fourier-
    reduced (i.e. no `2/π` QPM efficiency derating is applied — pass an
    already-derated `kappa` if that correction is desired).
"""
struct SecondOrderMedium{T <: Real} <: AbstractMedium
    length::T
    kappa::T
    dispersion_fund::DispersionModel
    dispersion_sh::DispersionModel
    deltak0::T
    poling_period::T
    loss_fund::T
    loss_sh::T
    lambda0::T

    function SecondOrderMedium(
        length::T,
        kappa::T,
        dispersion_fund::DispersionModel,
        dispersion_sh::DispersionModel,
        deltak0::T,
        poling_period::T,
        loss_fund::T,
        loss_sh::T,
        lambda0::T,
    ) where {T <: Real}
        length > 0 || throw(ArgumentError("Waveguide length must be positive"))
        poling_period >= 0 ||
            throw(ArgumentError("poling_period must be non-negative (0 disables QPM)"))
        loss_fund >= 0 || throw(ArgumentError("loss_fund must be non-negative"))
        loss_sh >= 0 || throw(ArgumentError("loss_sh must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T}(
            length,
            kappa,
            dispersion_fund,
            dispersion_sh,
            deltak0,
            poling_period,
            loss_fund,
            loss_sh,
            lambda0,
        )
    end
end

function SecondOrderMedium(;
    length::Real,
    kappa::Real,
    betas_fund::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion_fund::Union{DispersionModel, Nothing}=nothing,
    beta1_fund::Real=0.0,
    betas_sh::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion_sh::Union{DispersionModel, Nothing}=nothing,
    beta1_sh::Real=0.0,
    deltak0::Real,
    poling_period::Real=0.0,
    loss_fund::Real=0.0,
    loss_sh::Real=0.0,
    lambda0::Real,
)
    (betas_fund === nothing) ⊻ (dispersion_fund === nothing) ||
        throw(ArgumentError("provide exactly one of `betas_fund` or `dispersion_fund`"))
    (betas_sh === nothing) ⊻ (dispersion_sh === nothing) ||
        throw(ArgumentError("provide exactly one of `betas_sh` or `dispersion_sh`"))
    disp_fund =
        dispersion_fund === nothing ? TaylorDispersion(betas_fund, beta1_fund) :
        dispersion_fund
    disp_sh =
        dispersion_sh === nothing ? TaylorDispersion(betas_sh, beta1_sh) : dispersion_sh
    T = promote_type(
        typeof(length),
        typeof(kappa),
        typeof(deltak0),
        typeof(poling_period),
        typeof(loss_fund),
        typeof(loss_sh),
        typeof(lambda0),
        Float64,
    )
    return SecondOrderMedium(
        T(length),
        T(kappa),
        disp_fund,
        disp_sh,
        T(deltak0),
        T(poling_period),
        T(loss_fund),
        T(loss_sh),
        T(lambda0),
    )
end

"""
    SecondOrderPulse(At, grid)
    SecondOrderPulse(At_fund, At_sh, grid)

Two-component (fundamental + second-harmonic) optical pulse envelope in time and
frequency domains, for use with [`SecondOrderMedium`](@ref).

# Fields

  - `At::Matrix{ComplexF64}`: Time domain envelope matrix of size `N × 2` [√W]
    (`At[:,1]` = fundamental, `At[:,2]` = second harmonic)
  - `AW::Matrix{ComplexF64}`: Frequency domain envelope matrix of size `N × 2` [√W·s]
  - `grid::Grid`: Associated time-frequency grid (defined at the fundamental
    wavelength; `grid.omega0` must equal the fundamental's ω₀)
"""
struct SecondOrderPulse <: CoupledPulse
    At::Matrix{ComplexF64} # N x 2: [:,1]=fundamental, [:,2]=second harmonic
    AW::Matrix{ComplexF64}
    grid::Grid

    function SecondOrderPulse(At::Matrix{<:Complex}, AW::Matrix{<:Complex}, grid::Grid)
        size(At, 1) == grid.N ||
            throw(ArgumentError("Pulse dimensions must match grid resolution"))
        size(At, 2) == 2 ||
            throw(ArgumentError("SecondOrderPulse must have exactly 2 components (fundamental and SH)"))
        size(AW) == size(At) ||
            throw(ArgumentError("AW dimensions must match At dimensions"))
        new(Matrix{ComplexF64}(At), Matrix{ComplexF64}(AW), grid)
    end

    function SecondOrderPulse(At::Matrix{<:Complex}, grid::Grid)
        size(At, 1) == grid.N ||
            throw(ArgumentError("Pulse dimensions must match grid resolution"))
        size(At, 2) == 2 ||
            throw(ArgumentError("SecondOrderPulse must have exactly 2 components (fundamental and SH)"))
        AW = similar(At)
        AW[:, 1] .= ifft(At[:, 1])
        AW[:, 2] .= ifft(At[:, 2])
        new(Matrix{ComplexF64}(At), AW, grid)
    end
end

function SecondOrderPulse(
    At_fund::AbstractVector{<:Number}, At_sh::AbstractVector{<:Number}, grid::Grid
)
    length(At_fund) == grid.N ||
        throw(ArgumentError("fundamental component length must match grid resolution"))
    length(At_sh) == grid.N ||
        throw(ArgumentError("second-harmonic component length must match grid resolution"))
    At = Matrix{ComplexF64}(undef, grid.N, 2)
    At[:, 1] .= At_fund
    At[:, 2] .= At_sh
    return SecondOrderPulse(At, grid)
end

"""
    SecondOrderSolution{T<:Complex}

Solution to coupled χ⁽²⁾ (fundamental + second-harmonic) propagation.

# Fields

  - `t::Vector{Float64}`: Time domain grid [s]
  - `W::Vector{Float64}`: Absolute angular frequency grid [rad/s]
  - `omega0::Float64`: Central (fundamental) angular frequency [rad/s]
  - `Z::Vector{Float64}`: Propagation distances [m]
  - `At::Array{T, 3}`: Time domain solution array of size `N × 2 × z_saves`
  - `AW::Array{T, 3}`: Frequency domain solution array of size `N × 2 × z_saves`
"""
struct SecondOrderSolution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Array{T, 3} # N x 2 x z_saves
    AW::Array{T, 3} # N x 2 x z_saves
end

function Base.show(io::IO, p::SecondOrderPulse)
    Pmax_fund = maximum(abs2, p.At[:, 1])
    Pmax_sh = maximum(abs2, p.At[:, 2])
    print(
        io,
        "SecondOrderPulse(N=",
        p.grid.N,
        ", peak_fund=",
        round(Pmax_fund; sigdigits=4),
        " W, peak_sh=",
        round(Pmax_sh; sigdigits=4),
        " W)",
    )
end

function Base.show(io::IO, s::SecondOrderSolution)
    print(
        io,
        "SecondOrderSolution(",
        length(s.Z),
        " saves over ",
        round(s.Z[end]; sigdigits=4),
        " m, N=",
        size(s.At, 1),
        ")",
    )
end
