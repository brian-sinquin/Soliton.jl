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
    loss_vector(V, medium, z=0.0) -> Vector{Float64}

Compute linear attenuation vector α [Np/m] across relative angular frequencies V at position z.
Supports scalar loss [dB/m], vector spectrum loss [dB/m], and function loss(z), loss(ω), or loss(ω, z).
"""
function loss_vector(V::AbstractVector{Float64}, medium::AbstractMedium, z::Float64=0.0)
    loss_val = hasproperty(medium, :loss) ? medium.loss : 0.0
    return _eval_loss_or_gain(V, medium, loss_val, z, true)
end

loss_vector(grid::Grid, medium::AbstractMedium, z::Float64=0.0) = loss_vector(grid.V, medium, z)

"""
    gain_vector(V, medium, z=0.0) -> Vector{Float64}

Compute small-signal gain vector g₀ [Np/m] across relative angular frequencies V at position z.
Supports scalar gain [Np/m], vector gain spectrum g₀(ω), and function g₀(z), g₀(ω), or g₀(ω, z).
"""
function gain_vector(V::AbstractVector{Float64}, medium::AbstractMedium, z::Float64=0.0)
    if hasproperty(medium, :g0)
        return _eval_loss_or_gain(V, medium, medium.g0, z, false)
    else
        return zeros(Float64, length(V))
    end
end

gain_vector(grid::Grid, medium::AbstractMedium, z::Float64=0.0) = gain_vector(grid.V, medium, z)

function _eval_loss_or_gain(V::AbstractVector{Float64}, medium::AbstractMedium, val::Any, z::Float64, is_loss::Bool)
    N = length(V)
    factor = log(10.0) / 10.0
    if val isa Real
        alpha_Np = is_loss ? Float64(val) * factor : Float64(val)
        return fill(alpha_Np, N)
    elseif val isa AbstractVector
        length(val) == N || throw(ArgumentError("Spectrum length $(length(val)) does not match grid size $N"))
        return is_loss ? Float64.(val) .* factor : collect(Float64, val)
    elseif val isa Function || applicable(val, 1.0) || applicable(val, 1.0, 0.0)
        omega0 = 2π * c / medium.lambda0
        omegas = V .+ omega0
        res = zeros(Float64, N)
        for i in 1:N
            out = try
                val(omegas[i], z)
            catch
                try
                    val(omegas[i])
                catch
                    val(z)
                end
            end
            res[i] = is_loss ? Float64(out) * factor : Float64(out)
        end
        return res
    else
        return zeros(Float64, N)
    end
end

"""
    SilicaLossSpectrum(; C_R=1.7, alpha_OH=50.0, lambda_OH=1.383e-6, sigma_OH=0.015e-6, A_IR=6.0e7, B_IR=48.0e-6)

Callable loss model for standard fused silica optical fiber attenuation [dB/m] as a function of angular frequency `omega` [rad/s].

Includes:
1. Rayleigh scattering: α_Rayleigh(λ) = C_R / (λ [μm])⁴ [dB/km] (default C_R = 1.7 dB·μm⁴/km)
2. OH absorption peak at 1383 nm (default peak 50 dB/km)
3. Infrared multiphonon absorption edge: α_IR(λ) = A_IR · exp(-B_IR / λ) [dB/km]
"""
struct SilicaLossSpectrum
    C_R::Float64        # dB*um^4/km
    alpha_OH::Float64   # dB/km peak height
    lambda_OH::Float64  # m
    sigma_OH::Float64   # m
    A_IR::Float64       # dB/km
    B_IR::Float64       # m

    function SilicaLossSpectrum(;
        C_R::Real=1.7,
        alpha_OH::Real=50.0,
        lambda_OH::Real=1.383e-6,
        sigma_OH::Real=0.015e-6,
        A_IR::Real=6.0e7,
        B_IR::Real=48.0e-6,
    )
        new(Float64(C_R), Float64(alpha_OH), Float64(lambda_OH), Float64(sigma_OH), Float64(A_IR), Float64(B_IR))
    end
end

function (s::SilicaLossSpectrum)(omega::Real, z::Real=0.0)
    lambda = 2π * c / abs(omega)
    lambda_um = lambda * 1e6

    alpha_rayleigh_db_km = s.C_R / (lambda_um^4)
    alpha_oh_db_km = s.alpha_OH * exp(-((lambda - s.lambda_OH) / s.sigma_OH)^2)
    alpha_ir_db_km = s.A_IR * exp(-s.B_IR / lambda)

    loss_db_km = alpha_rayleigh_db_km + alpha_oh_db_km + alpha_ir_db_km
    return loss_db_km / 1000.0 # Convert dB/km -> dB/m
end

"""
    dispersion_operator(V::AbstractVector{Float64}, medium::AbstractMedium, z::Float64=0.0)

Construct the linear dispersion & gain/loss operator `D(V, z) = i·B(V) + (g(V, z) - α(V, z))/2` [1/m].
"""
function dispersion_operator(V::AbstractVector{Float64}, medium::AbstractMedium, z::Float64=0.0)
    omega0 = 2π * c / medium.lambda0
    disp_model = hasproperty(medium, :dispersion) ? medium.dispersion : (hasproperty(medium, :dispersion_x) ? medium.dispersion_x : TaylorDispersion([0.0]))
    B = propagation_constant(V, disp_model, omega0)
    loss = loss_vector(V, medium, z)
    gain = gain_vector(V, medium, z)
    return @. 1im * B + (gain - loss) / 2
end

"""
    dispersion_operator(grid::Grid, medium::AbstractMedium, z::Float64=0.0)

Convenience wrapper that extracts `V` from `grid`.
"""
dispersion_operator(grid::Grid, medium::AbstractMedium, z::Float64=0.0) = dispersion_operator(grid.V, medium, z)

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
