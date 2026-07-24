"""
Types and structures for JuGNLSE in natural SI units.

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
end

# An empty `betas` vector means no dispersion (pure SPM).
TaylorDispersion(betas::AbstractVector{<:Real}) = TaylorDispersion(collect(Float64, betas))

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
        length(detuning) >= 2 ||
            throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) ||
            throw(ArgumentError("detuning must be sorted ascending"))
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

    function SellmeierDispersion(B::AbstractVector{<:Real}, C::AbstractVector{<:Real}; microns::Bool=true)
        length(B) == length(C) || throw(ArgumentError("B and C must have equal length"))
        C_val = microns ? collect(Float64, C) .* 1e-12 : collect(Float64, C) # 1 μm² = 1e-12 m²
        new(collect(Float64, B), C_val)
    end
end

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
Medium(; length, gamma, loss=0.0, betas=…, dispersion=…, lambda0)
```

# Notes

  - Taylor `betas` exclude β₀ and β₁; `betas[1] = β₂`
  - loss in dB/m converted to α = ln(10^(loss/10)) Np/m in the operator
"""
struct Medium{T <: Real, TG}
    length::T
    gamma::TG
    loss::T
    dispersion::DispersionModel
    lambda0::T

    function Medium{T, TG}(
        length::T, gamma::TG, loss::T, dispersion::DispersionModel, lambda0::T
    ) where {T <: Real, TG}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        if TG <: Real
            gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        end
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))

        new{T, TG}(length, gamma, loss, dispersion, lambda0)
    end
end

# Outer constructors
function Medium(
    length::Real, gamma::Any, loss::Real, dispersion::DispersionModel, lambda0::Real
)
    T = promote_type(typeof(length), typeof(loss), typeof(lambda0), Float64)
    return Medium{T, typeof(gamma)}(T(length), gamma, T(loss), dispersion, T(lambda0))
end

function Medium(
    length::Real, gamma::Any, loss::Real, betas::AbstractVector{<:Real}, lambda0::Real
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
    loss::Real=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(
        typeof(length), typeof(loss), typeof(lambda0), Float64
    )
    Medium{T, typeof(gamma)}(T(length), gamma, T(loss), disp, T(lambda0))
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
    Pulse{T<:Complex}

Optical pulse envelope in time and frequency domains.

# Fields

  - `At::Vector{T}`: Time domain envelope A(t) [√W]
  - `AW::Vector{T}`: Frequency domain envelope A(ω) [√W·ps]
  - `grid::Grid`: Associated time-frequency grid

# Notes

Following gnlse-python convention:

  - AW = N * ifft(At) (note: inverted FFT convention)
  - At = fft(AW)
  - Power: P(t) = |A(t)|²
  - Energy: E = ∫|A(t)|²dt
"""
mutable struct Pulse{T <: Complex}
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
  - `rtol::Float64`: Relative tolerance for adaptive ODE solver
  - `atol::Float64`: Absolute tolerance for adaptive ODE solver

# Notes

Following gnlse-python's GNLSESetup structure:

  - z_saves determines memory allocation for result arrays
  - Raman disabled by setting raman_model = nothing or fr = 0
  - Self-steepening modifies W frequency grid
"""
struct SimParams{S <: GNLSESolver}
    medium::Medium
    z_saves::Int
    raman_model::Union{RamanModel, Nothing}
    self_steepening::Bool
    solver::S

    function SimParams(
        medium::Medium,
        z_saves::Int,
        raman_model::Union{RamanModel, Nothing},
        self_steepening::Bool,
        solver::S,
    ) where {S <: GNLSESolver}
        z_saves > 0 || throw(ArgumentError("z_saves must be positive"))
        new{S}(medium, z_saves, raman_model, self_steepening, solver)
    end
end

# Backward-compatible positional constructor
function SimParams(
    medium::Medium,
    z_saves::Int,
    raman_model::Union{RamanModel, Nothing},
    self_steepening::Bool,
    rtol::Float64,
    atol::Float64,
)
    return SimParams(medium, z_saves, raman_model, self_steepening, ERK4IP(; rtol=rtol, atol=atol))
end

# Keyword constructor supporting both generic solver and backward-compatible rtol/atol
function SimParams(;
    medium::Medium,
    z_saves::Int=200,
    raman_model::Union{RamanModel, Nothing}=BlowWood(),
    self_steepening::Bool=false,
    rtol::Union{Float64, Nothing}=nothing,
    atol::Union{Float64, Nothing}=nothing,
    solver::Union{GNLSESolver, Nothing}=nothing,
)
    if solver === nothing
        rtol_val = rtol === nothing ? 1e-6 : rtol
        atol_val = atol === nothing ? 1e-8 : atol
        solv = ERK4IP(; rtol=rtol_val, atol=atol_val)
    else
        (rtol === nothing && atol === nothing) ||
            throw(ArgumentError("Cannot specify both `solver` and `rtol`/`atol` compatibility options"))
        solv = solver
    end
    return SimParams(medium, z_saves, raman_model, self_steepening, solv)
end

"""
    Solution{T<:Complex}

Solution to GNLSE propagation.

# Fields

  - `t::Vector{Float64}`: Time domain grid [ps]
  - `W::Vector{Float64}`: Absolute angular frequency grid [rad/ps = THz]
  - `omega0::Float64`: Central angular frequency [rad/ps = THz]
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
    print(io, "Medium(L=", m.length, " m, γ=", m.gamma, " /W/m, loss=",
          m.loss, " dB/m, λ₀=", round(m.lambda0 * 1e9; digits=2), " nm, ",
          m.dispersion, ")")
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
    print(io, "Grid(N=", g.N, ", window=",
          round((g.t[end] - g.t[1]) * 1e12; digits=3), " ps, λ₀=",
          round(g.lambda0 * 1e9; digits=2), " nm, dt=",
          round(g.dt * 1e15; digits=3), " fs)")
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
  - rtol, atol: ODE solver tolerances
"""
function Base.show(io::IO, s::SimParams)
    print(io, "SimParams(z_saves=", s.z_saves, ", raman=",
          s.raman_model === nothing ? "off" : nameof(typeof(s.raman_model)),
          ", self_steepening=", s.self_steepening, ", rtol=", s.rtol,
          ", atol=", s.atol, ")")
end

"""
    show(io::IO, s::Solution)

Display a Solution compactly in the REPL. Shows:
  - number of saved distances
  - propagation distance span [m]
  - grid resolution [points]
"""
function Base.show(io::IO, s::Solution)
    print(io, "Solution(", length(s.Z), " saves over ",
          round(s.Z[end]; sigdigits=4), " m, N=", size(s.At, 1), ")")
end
