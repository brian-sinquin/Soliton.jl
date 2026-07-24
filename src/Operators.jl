module Operators

using ..Types: DispersionModel, TaylorDispersion, TabulatedDispersion, Medium, Grid
using ..Types: AbstractGammaCoefficient, ConstantGamma, ZDependentGamma, WavelengthDependentGamma
using SpecialFunctions: factorial

export propagation_constant, dispersion_operator, compute_gamma

function propagation_constant(V::AbstractVector{Float64}, model::TaylorDispersion)
    B = zeros(Float64, length(V))
    for (i, beta) in enumerate(model.betas)
        n = i + 1
        B .+= beta ./ factorial(n) .* (V .^ n)
    end
    return B
end

function propagation_constant(V::AbstractVector{Float64}, model::TabulatedDispersion)
    xs, ys = model.detuning, model.beta
    B = similar(V)
    @inbounds for k in eachindex(V, B)
        x = V[k]
        if x <= xs[1]
            B[k] = ys[1]
        elseif x >= xs[end]
            B[k] = ys[end]
        else
            j = searchsortedlast(xs, x)
            t = (x - xs[j]) / (xs[j + 1] - xs[j])
            B[k] = ys[j] + t * (ys[j + 1] - ys[j])
        end
    end
    return B
end

function dispersion_operator(V::AbstractVector{Float64}, medium::Medium)
    alpha = log(10.0^(medium.loss / 10.0))
    B = propagation_constant(V, medium.dispersion)
    return @. 1im * B - alpha / 2
end

dispersion_operator(grid::Grid, medium::Medium) = dispersion_operator(grid.V, medium)

function compute_gamma(gamma_coeff::ConstantGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma
end
function compute_gamma(gamma_coeff::ZDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(z)
end
function compute_gamma(gamma_coeff::WavelengthDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(lambda)
end

end # module
