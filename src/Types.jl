module Types

export DispersionModel, TaylorDispersion, TabulatedDispersion
export Medium, Grid, RamanModel, BlowWood, LinAgrawal, Hollenbeck
export AbstractGNLSESolver, AbstractGammaCoefficient, ConstantGamma, ZDependentGamma, WavelengthDependentGamma
export Pulse, SimParams, GNLSEProblem, Solution

abstract type DispersionModel end
struct TaylorDispersion <: DispersionModel
    betas::Vector{Float64}
end
TaylorDispersion(betas::AbstractVector{<:Real}) = TaylorDispersion(collect(Float64, betas))

struct TabulatedDispersion <: DispersionModel
    detuning::Vector{Float64}
    beta::Vector{Float64}
    function TabulatedDispersion(detuning::AbstractVector{<:Real}, beta::AbstractVector{<:Real})
        length(detuning) == length(beta) || throw(ArgumentError("detuning and beta must have equal length"))
        length(detuning) >= 2 || throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) || throw(ArgumentError("detuning must be sorted ascending"))
        new(collect(Float64, detuning), collect(Float64, beta))
    end
end

struct Medium{T <: Real}
    length::T
    gamma::T
    loss::T
    dispersion::DispersionModel
    lambda0::T

    function Medium{T}(length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T) where {T <: Real}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T}(length, gamma, loss, dispersion, lambda0)
    end
end

# Positional constructor — Taylor betas (backward compatible)
Medium(length::T, gamma::T, loss::T, betas::AbstractVector{<:Real}, lambda0::T) where {T <: Real} =
    Medium{T}(length, gamma, loss, TaylorDispersion(betas), lambda0)

# Positional constructor — explicit dispersion model
Medium(length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T) where {T <: Real} =
    Medium{T}(length, gamma, loss, dispersion, lambda0)

function Medium(;
    length::Real,
    gamma::Real,
    loss::Real=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(typeof(length), typeof(gamma), typeof(loss), typeof(lambda0), Float64)
    Medium{T}(T(length), T(gamma), T(loss), disp, T(lambda0))
end

struct Grid{T <: Real}
    N::Int
    t::Vector{T}
    V::Vector{T}
    W::Vector{T}
    dt::T
    omega0::T
    lambda0::T
end

abstract type RamanModel end

struct BlowWood <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64
    BlowWood(; fr::Float64=0.18, tau1::Float64=12.2e-15, tau2::Float64=32.0e-15) =
        new(fr, tau1, tau2)
end

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

struct Hollenbeck <: RamanModel
    fr::Float64
    Hollenbeck(; fr::Float64=0.20) = new(fr)
end

abstract type AbstractGNLSESolver end

abstract type AbstractGammaCoefficient end

struct ConstantGamma <: AbstractGammaCoefficient
    gamma::Float64
end
ConstantGamma(gamma::Real) = ConstantGamma(Float64(gamma))

struct ZDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

struct WavelengthDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

mutable struct Pulse{T <: Complex}
    At::Vector{T}
    AW::Vector{T}
    grid::Grid
end

struct SimParams
    medium::Medium
    z_saves::Int
    raman_model::Union{RamanModel, Nothing}
    self_steepening::Bool
    rtol::Float64
    atol::Float64

    function SimParams(
        medium::Medium,
        z_saves::Int,
        raman_model::Union{RamanModel, Nothing},
        self_steepening::Bool,
        rtol::Float64,
        atol::Float64,
    )
        z_saves > 0 || throw(ArgumentError("z_saves must be positive"))
        rtol > 0 || throw(ArgumentError("rtol must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))
        new(medium, z_saves, raman_model, self_steepening, rtol, atol)
    end
end

function SimParams(;
    medium::Medium,
    z_saves::Int=200,
    raman_model::Union{RamanModel, Nothing}=BlowWood(),
    self_steepening::Bool=false,
    rtol::Float64=1e-6,
    atol::Float64=1e-8,
)
    SimParams(medium, z_saves, raman_model, self_steepening, rtol, atol)
end

struct GNLSEProblem{T <: Complex}
    medium::Medium
    grid::Grid
    initial_pulse::Pulse{T}
    sim_params::SimParams
    gamma_coefficient::AbstractGammaCoefficient
end

function GNLSEProblem(;medium::Medium, grid::Grid, initial_pulse::Pulse{T}, sim_params::SimParams, gamma_coefficient::AbstractGammaCoefficient=ConstantGamma(medium.gamma)) where T
    GNLSEProblem(medium, grid, initial_pulse, sim_params, gamma_coefficient)
end

function GNLSEProblem(pulse::Pulse{T}, medium::Medium, sim_params::SimParams, gamma_coeff::AbstractGammaCoefficient=ConstantGamma(medium.gamma)) where T
    GNLSEProblem(medium, pulse.grid, pulse, sim_params, gamma_coeff)
end

struct Solution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Matrix{T}
    AW::Matrix{T}
end

end # module
