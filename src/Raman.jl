module Raman

using ..Types: RamanModel, BlowWood, LinAgrawal, Hollenbeck, Grid
using SpecialFunctions: factorial # For factorial function in propagation_constant
using LinearAlgebra # For `mul!`
using FFTW # For ifft and fft
import ..JuGNLSE: c

export raman_response

"""
    raman_response(T::Vector{Float64}, model::BlowWood)

Compute Raman response following gnlse-python raman_blowwood.

# Arguments
- `T::Vector{Float64}`: Time vector [ps]
- `model::BlowWood`: Raman model with parameters

# Returns
- `(fr, RT)`: Raman fraction and response function

# Physics
Following gnlse-python raman_blowwood:

```python
tau1 = 0.0122  # ps
tau2 = 0.032   # ps
ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
RT = ha
RT[T < 0] = 0
fr = 0.18
```

Reference: K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989)
"""
function raman_response(T::Vector{Float64}, model::BlowWood)
    tau1 = model.tau1  # s
    tau2 = model.tau2  # s

    # gnlse-python: ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
    RT = (tau1^2 + tau2^2) / tau1 / (tau2^2) .* exp.(-T ./ tau2) .* sin.(T ./ tau1)

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    return model.fr, RT
end

"""
    raman_response(T::Vector{Float64}, model::LinAgrawal)

Compute Raman response following gnlse-python raman_linagrawal.

# Arguments
- `T::Vector{Float64}`: Time vector [ps]
- `model::LinAgrawal`: Raman model with parameters

# Returns
- `(fr, RT)`: Raman fraction and response function

# Physics
Following gnlse-python raman_linagrawal:
```python
tau1 = 0.0122  # ps
tau2 = 0.032   # ps
taub = 0.096   # ps
fb = 0.21
fc = 0.04
fa = 1 - fb - fc
# Anisotropic response
ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
# Isotropic response
hb = (2*taub - T) / (taub**2) * exp(-T/taub)
# Total response
RT = (fa + fc) * ha + fb * hb
RT[T < 0] = 0
fr = 0.245
```

Reference: Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006)
"""
function raman_response(T::Vector{Float64}, model::LinAgrawal)
    tau1 = model.tau1  # s
    tau2 = model.tau2  # s
    taub = model.taub  # s
    fb = model.fb
    fc = model.fc
    fa = 1 - fb - fc

    # gnlse-python: ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
    ha = (tau1^2 + tau2^2) / tau1 / (tau2^2) .* exp.(-T ./ tau2) .* sin.(T ./ tau1)

    # gnlse-python: hb = (2*taub - T) / (taub**2) * exp(-T/taub)
    hb = (2 .* taub .- T) ./ (taub^2) .* exp.(-T ./ taub)

    # gnlse-python: RT = (fa + fc) * ha + fb * hb
    RT = (fa + fc) .* ha .+ fb .* hb

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    return model.fr, RT
end

"""
    raman_response(T::Vector{Float64}, model::Hollenbeck)

Compute Raman response following D. Hollenbeck & C. D. Cantrell's 13-oscillator fit.

# Arguments
  - `T::Vector{Float64}`: Time vector [s]
  - `model::Hollenbeck`: Raman model with Raman fraction fr

# Returns
  - `(fr, RT)`: Raman fraction `fr` and impulse response `RT(t)` [1/s]

# Physics

The Hollenbeck model combines 13 Lorentzian resonances with Gaussian spectral
broadening to fit experimental Raman gain/loss data from silica fiber:

    h(ω) = Σ Aᵢ [Lorentzian(ω - ωᵢ, Γᵢ) ⊗ Gaussian(ΔGᵢ)]

Each resonance is parametrized by:
  - **CP**: center position [cm⁻¹]
  - **A**: peak amplitude (relative units)
  - **Gauss**: Gaussian FWHM [cm⁻¹]
  - **Lorentz**: Lorentzian FWHM [cm⁻¹]

The model is converted to the time domain and normalized by the Raman fraction
`fr` (set to 0.20 by default), which represents the fractional power transfer
into the Raman-shifted component.

Reference: D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002)
"""
function raman_response(T::Vector{Float64}, model::Hollenbeck)
    # Component positions [1/cm]
    CP = [56.25, 100.0, 231.25, 362.5, 463.0, 497.0, 611.5, 691.67, 793.67,
          835.5, 930.0, 1080.0, 1215.0]

    # Peak intensity (amplitude)
    A = [1.0, 11.40, 36.67, 67.67, 74.0, 4.5, 6.8, 4.6, 4.2, 4.5, 2.7, 3.1, 3.0]

    # Gaussian FWHM [1/cm]
    Gauss = [52.10, 110.42, 175.00, 162.50, 135.33, 24.5, 41.5, 155.00, 59.5, 64.3,
             150.0, 91.0, 160.0]

    # Lorentzian FWHM [1/cm]
    Lorentz = [17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67, 19.83, 21.43,
               50.00, 30.33, 53.33]

    # Convert wavenumbers [1/cm] to angular frequencies/rates [rad/s].
    # ω = 2π·c·(CP·100), with c in m/s; L and γ use π·c (FWHM convention).
    # c will be provided by the main JuGNLSE module
    w = 2π .* c .* 100.0 .* CP
    L = π .* c .* 100.0 .* Gauss
    gamma_lorentz = π .* c .* 100.0 .* Lorentz

    # Initialize RT
    RT = zeros(Float64, length(T))

    # gnlse-python: RT += A[i] * exp(-gamma[i]*T) * exp((-L[i]**2*T**2)/4) * sin(w[i]*T)
    for i in 1:length(A)
        @. RT += A[i] * exp(-gamma_lorentz[i] * T) * exp((-L[i]^2 * T^2) / 4) * sin(w[i] * T)
    end

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    # Normalize
    # gnlse-python: dt = T[1] - T[0]; RT = RT / (sum(RT) * dt)
    dt = T[2] - T[1]
    RT = RT ./ (sum(RT) * dt)

    return model.fr, RT
end

"""
    raman_response(grid::Grid, model::RamanModel)

Convenience wrapper that extracts time vector from grid.

# Arguments
- `grid::Grid`: Grid with time vector T
- `model::RamanModel`: Raman model

# Returns
- `(fr, RT)`: Raman fraction and response function
"""
function raman_response(grid::Grid, model::RamanModel)
    return raman_response(grid.t, model)
end

end # module
