"""
Dispersion operator for GNLSE simulations in natural SI units.
"""

"""
    propagation_constant(V, model::DispersionModel)

Propagation-constant deviation `B(V)` [1/m] for the dispersion `model`, sampled
on the relative angular-frequency grid `V = ω - ω₀` [rad/s]. This is an
intermediate quantity (intermediate in the frequency domain) used internally to
construct the dispersion operator [`dispersion_operator`](@ref).

# Method Implementations

For **TaylorDispersion**, computes the power-series expansion:

    B(V) = Σ βₙ/n! · Vⁿ,  n ≥ 2

This representation is fast and suits analytical studies, but assumes dispersion
is smooth and well-approximated by the first few terms.

For **TabulatedDispersion**, linearly interpolates the measured/numerically-computed
dispersion curve onto the simulation grid, then uses constant extrapolation beyond
the tabulated frequency range. This is more accurate for complex materials (PCF,
highly dispersive windows) but requires tabulated data.
"""
function propagation_constant(V::AbstractVector{Float64}, model::TaylorDispersion)
    # Taylor series: B = β₁ · V + Σ βₙ/n! · Vⁿ, n ≥ 2
    B = model.beta1 .* V
    for (i, beta) in enumerate(model.betas)
        n = i + 1  # betas[1]=β₂ → n=2, betas[2]=β₃ → n=3, etc.
        B .+= beta ./ factorial(n) .* (V .^ n)
    end
    return B
end

function propagation_constant(V::AbstractVector{Float64}, model::TabulatedDispersion)
    # Linear interpolation onto V, flat extrapolation outside the tabulated range
    xs, ys = model.detuning, model.beta
    B = similar(V)
    @inbounds for k in eachindex(V, B)
        x = V[k]
        if x <= xs[1]
            B[k] = ys[1]
        elseif x >= xs[end]
            B[k] = ys[end]
        else
            j = searchsortedlast(xs, x)         # xs[j] ≤ x < xs[j+1]
            t = (x - xs[j]) / (xs[j + 1] - xs[j])
            B[k] = ys[j] + t * (ys[j + 1] - ys[j])
        end
    end
    return B
end

"""
    dispersion_operator(V::AbstractVector{Float64}, medium::Medium)

Construct the linear dispersion operator `D(V) = i·B(V) - α/2` [1/m], where:

  - `B(V)`: propagation-constant deviation from the dispersion model [1/m]
  - `α`: fiber loss in Neper/m, converted from dB/m via α = ln(10^(loss/10))
  - The factor i·B appears in the interaction-picture GNLSE; α/2 implements
    exponential decay

# Arguments

  - `V::Vector{Float64}`: relative angular frequency [rad/s]
  - `medium::Medium`: fiber with dispersion model and loss

# Returns

  - `D::Vector{ComplexF64}`: dispersion operator, one value per frequency bin

# Notes

The loss term `-α/2` in the frequency domain translates to multiplicative decay
`exp(-αz)` in the time domain (amplitude), which becomes `exp(-2αz)` in intensity.
See [`medium.loss`](@ref Medium) for units.
"""
function dispersion_operator(V::AbstractVector{Float64}, medium::Medium)
    alpha = log(10.0^(medium.loss / 10.0))
    omega0 = 2π * c / medium.lambda0
    B = propagation_constant(V, medium.dispersion, omega0)
    return @. 1im * B - alpha / 2
end

"""
    dispersion_operator(grid::Grid, medium::Medium)

Convenience wrapper that extracts `V` from `grid`.
"""
dispersion_operator(grid::Grid, medium::Medium) = dispersion_operator(grid.V, medium)

# 3-argument compatibility fallback
propagation_constant(V::AbstractVector{Float64}, model::DispersionModel, omega0::Float64) = propagation_constant(V, model)

"""
    propagation_constant(V::AbstractVector{Float64}, model::SellmeierDispersion, omega0::Float64)

Compute the propagation constant deviation B(V) using the Sellmeier dispersion equation.
"""
function propagation_constant(V::AbstractVector{Float64}, model::SellmeierDispersion, omega0::Float64)
    lambda0 = 2π * c / omega0

    # Calculate refractive index and derivative at central frequency
    n2_0 = 1.0
    sum_deriv_0 = 0.0
    for i in 1:length(model.B)
        Bi = model.B[i]
        Ci = model.C[i]
        n2_0 += Bi * lambda0^2 / (lambda0^2 - Ci)
        sum_deriv_0 += Bi * Ci / (lambda0^2 - Ci)^2
    end
    n_0 = sqrt(n2_0)
    beta0 = n_0 * omega0 / c
    beta1_0 = n_0 / c + (lambda0^2 / (c * n_0)) * sum_deriv_0

    # Calculate propagation constant deviation for each frequency bin
    B = similar(V)
    @inbounds for k in eachindex(V)
        omega = omega0 + V[k]
        if omega <= 0
            throw(ArgumentError("Absolute frequency must be positive. Check grid size and central frequency."))
        end
        lk = 2π * c / omega

        n2 = 1.0
        for i in 1:length(model.B)
            n2 += model.B[i] * lk^2 / (lk^2 - model.C[i])
        end
        nk = sqrt(n2)
        betak = nk * omega / c
        B[k] = betak - beta0 - beta1_0 * V[k]
    end

    return B
end
