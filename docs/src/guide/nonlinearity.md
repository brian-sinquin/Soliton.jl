```@meta
CurrentModule = GNLSE
```

# Wavelength-Dependent Nonlinearity

GNLSE.jl supports three levels of nonlinearity specification:

## 1. Constant Nonlinearity

The simplest model: a scalar ``\gamma`` [1/(W·m)].

```julia
# As a scalar in Medium
medium = Medium(0.1, 0.11, 0.0, [-1.2e-26], 835e-9)

# Or explicitly as ConstantNonlinearity
γ_model = ConstantNonlinearity(0.11)
medium = Medium(0.1, γ_model, 0.0, TaylorDispersion([-1.2e-26]), 835e-9)
```

## 2. Z-Dependent (Tapered Fiber)

For tapered or graded-index fibers where ``\gamma`` varies along the propagation distance:

```julia
# Exponential taper: γ(z) = γ₀ exp(-α_taper * z)
gamma_0 = 0.11
alpha_taper = 10.0   # 1/m
gamma_func = z -> gamma_0 * exp(-alpha_taper * z)

medium = Medium(0.01, gamma_func, 0.0, TaylorDispersion([-1.2e-26]), 835e-9)
```

The function signature must be `gamma_func(z::Float64) -> Float64`, returning ``\gamma(z)`` in [1/(W·m)].

## 3. Frequency-Dependent Nonlinearity

When the nonlinear coefficient varies with frequency (e.g., due to the frequency-dependent effective mode area):

```julia
# Direct frequency-domain function
γ_of_ω = ω -> 0.11 * (ω / (2π * 3e14))^0.5   # hypothetical dispersion

γ_model = FrequencyDependentNonlinearity(γ_of_ω)
medium = Medium(0.1, γ_model, 0.0, TaylorDispersion([-1.2e-26]), 835e-9)
```

## 4. Effective Mode Area Model

The most physically rigorous model: compute ``\gamma(\omega)`` from the material nonlinear index ``n_2`` and the frequency-dependent effective mode area ``A_{\rm eff}(\omega)``:

```math
\gamma(\omega) = \frac{n_2 \omega}{c \, A_{\rm eff}(\omega)}
```

```julia
n2 = 2.6e-20               # Nonlinear index [m²/W] (fused silica)
Aeff_func = ω -> 80e-12    # Constant Aeff = 80 µm² (standard SMF-28)

γ_model = NonlinearityFromEffectiveArea(n2, Aeff_func)
medium = Medium(1.0, γ_model, 0.0, TaylorDispersion([-21.5e-27]), 1550e-9)
```

A more realistic ``A_{\rm eff}(\omega)`` would be obtained from a mode solver:

```julia
# Hypothetical frequency-dependent Aeff from numerical mode solver data
using Interpolations
aeff_data  = [90e-12, 85e-12, 82e-12, 80e-12, 79e-12]   # [m²]
omega_data = range(1.1e15, 1.4e15; length=5)
Aeff_itp   = LinearInterpolation(omega_data, aeff_data; extrapolation_bc=Flat())

γ_model = NonlinearityFromEffectiveArea(n2, ω -> Aeff_itp(ω))
```

!!! tip
    Use `NonlinearityFromEffectiveArea` when modeling highly nonlinear fibers (HNLFs), photonic crystal fibers (PCFs), or integrated waveguides where the effective mode area varies significantly across the pulse bandwidth.

## Summary Table

| Model | When to use |
|:---|:---|
| Scalar `γ` | Standard single-mode fiber, rough estimates |
| `ConstantNonlinearity(γ)` | Explicit, equivalent to scalar |
| `z -> γ(z)` function | Tapered fibers, graded-index profiles |
| `FrequencyDependentNonlinearity` | Broadband models with known ``\gamma(\omega)`` |
| `NonlinearityFromEffectiveArea` | Mode-area data from FEM/mode solvers |
