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
end

# Constructors...
function Medium(; length, gamma, loss=0.0, betas=nothing, dispersion=nothing, lambda0)
    (betas === nothing) ⊻ (dispersion === nothing) || throw(ArgumentError("provide exactly one of betas or dispersion"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(typeof(length), typeof(gamma), typeof(loss), typeof(lambda0), Float64)
    Medium{T}(length, gamma, loss, disp, lambda0)
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
struct BlowWood <: RamanModel; fr::Float64; tau1::Float64; tau2::Float64; end
struct LinAgrawal <: RamanModel; fr::Float64; tau1::Float64; tau2::Float64; taub::Float64; fb::Float64; fc::Float64; end
struct Hollenbeck <: RamanModel; fr::Float64; end

abstract type AbstractGNLSESolver end
abstract type AbstractGammaCoefficient end

struct ConstantGamma <: AbstractGammaCoefficient; gamma::Float64; end
struct ZDependentGamma <: AbstractGammaCoefficient; gamma_func::Function; end
struct WavelengthDependentGamma <: AbstractGammaCoefficient; gamma_func::Function; end

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
end

struct GNLSEProblem{T <: Complex}
    medium::Medium
    grid::Grid
    initial_pulse::Pulse{T}
    sim_params::SimParams
    gamma_coefficient::AbstractGammaCoefficient
end

struct Solution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Matrix{T}
    AW::Matrix{T}
end

end # module Types
