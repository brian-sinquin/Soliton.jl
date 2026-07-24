module Pulses

using ..Types: Grid, Pulse, Solution
using FFTW: ifft # For ifft
using Random
using ..JuGNLSE: c

export create_grid, wavelength_grid, sech_pulse, gaussian_pulse, lorentzian_pulse, cw_pulse

# Conversion constant (c is now global in JuGNLSE.jl)
# const C = 299792458.0 # Speed of light in vacuum [m/s]

"""
    create_grid(resolution::Int, time_window::Real, wavelength::Real)

Create time-frequency grid for GNLSE simulations in natural SI units.

# Arguments

  - `resolution::Int`: Number of grid points (power of 2 recommended)
  - `time_window::Real`: Total time window [s]
  - `wavelength::Real`: Center wavelength [m]

# Returns

  - `Grid`: Grid structure with:

      + t: time grid [s] spanning [-time_window/2, time_window/2]
      + V: relative angular frequency ω - ω₀ [rad/s], monotonic
      + W: absolute angular frequency ω = ω₀ + V [rad/s], monotonic
      + dt: time step [s]
      + omega0: central angular frequency ω₀ [rad/s]
      + lambda0: center wavelength [m]

# Notes

  - ω₀ = 2πc/λ₀ where c = 299792458 m/s, gives ω₀ in rad/s
  - V = 2π · [-N/2, ..., N/2-1] / (N·dt) [rad/s] is the relative frequency
    (physical detuning ω - ω₀), monotonic ordering
  - W = ω₀ + V [rad/s] is the absolute optical frequency

`V` and `W` are stored in monotonic (not FFT-natural) order; operators that act
on FFT output apply `ifftshift` as needed.
"""
function create_grid(resolution::Int, time_window::Real, wavelength::Real)
    resolution > 0 || throw(ArgumentError("resolution must be positive"))
    ispow2(resolution) ||
        @warn "resolution should be a power of 2 for optimal FFT performance"
    time_window > 0 || throw(ArgumentError("time_window must be positive"))
    wavelength > 0 || throw(ArgumentError("wavelength must be positive"))

    N = resolution

    # Time domain grid [s]
    t = collect(range(-time_window / 2, time_window / 2; length=N))
    dt = t[2] - t[1]

    # Relative angular frequency grid [rad/s], monotonic.
    # This is the physical detuning ω - ω₀: the package uses the standard optics
    # FFT convention (envelope spectrum AW = ifft(At), field At = fft(AW)), so a
    # spectral component at V evolves in time as exp(-iVt).
    V = 2π .* ((-N ÷ 2):(N ÷ 2 - 1)) ./ (N * dt)

    # Central angular frequency [rad/s]: ω₀ = 2πc/λ₀
    omega0 = (2.0 * π * c) / wavelength

    # Absolute optical angular frequency grid ω = ω₀ + V [rad/s]
    W = omega0 .+ V

    Grid{Float64}(N, t, V, W, dt, omega0, wavelength)
end

"""
    wavelength_grid(grid::Grid)
    wavelength_grid(solution::Solution)

Wavelength grid [m] for the absolute frequency axis, λ = 2πc/ω. The result is
aligned element-for-element with `grid.W` (and with `solution.W` / the columns
of `solution.AW`), so it is monotonically decreasing in array order.
"""
wavelength_grid(grid::Grid) = (2π * c) ./ grid.W
wavelength_grid(solution::Solution) = (2π * c) ./ solution.W

"""
    sech_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate hyperbolic secant pulse in natural SI units.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python SechEnvelope:

```python
m = 2 * log(1 + sqrt(2))
A(T) = sqrt(Pmax) * 2 / (exp(m*T/FWHM) + exp(-m*T/FWHM))
     = sqrt(Pmax) * sech(m*T/FWHM)
```

Where m = 2*arcsinh(1) ≈ 1.763 is the factor relating FWHM to 1/e half-width.
"""
function sech_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 2 * np.log(1 + np.sqrt(2))
    m = 2 * log(1 + sqrt(2))

    # gnlse-python: A(T) = sqrt(Pmax) * 2 / (exp(m * grid.t / FWHM) + exp(-m * grid.t / FWHM))
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) * 2 / (exp(m * grid.t / FWHM) + exp(-m * grid.t / FWHM))

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    gaussian_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate Gaussian pulse following gnlse-python GaussianEnvelope.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python GaussianEnvelope, where `m = 4*log(2)` relates the 1/e²
half-width to the FWHM:

```python
A(T) = sqrt(Pmax) * exp(-m * 0.5 * T² / FWHM²)
```

This defines a pulse whose intensity drops to half-maximum at ±FWHM/2.
"""
function gaussian_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 4 * np.log(2)
    m = 4 * log(2)

    # gnlse-python: A(T) = sqrt(Pmax) * exp(-m * 0.5 * T**2 / FWHM**2)
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) * exp(-m * 0.5 * grid.t^2 / FWHM^2)

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    lorentzian_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate Lorentzian pulse following gnlse-python LorentzianEnvelope.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python LorentzianEnvelope:

```python
m = 2 * sqrt(sqrt(2) - 1)
A(T) = sqrt(Pmax) / (1 + (m*T/FWHM)^2)
```
"""
function lorentzian_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 2 * sqrt(sqrt(2) - 1)
    m = 2 * sqrt(sqrt(2) - 1)

    # gnlse-python: A(T) = sqrt(Pmax) / (1 + (m*T/FWHM)**2)
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) / (1 + (m * grid.t / FWHM)^2)

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    cw_pulse(grid::Grid, Pmax::Real; Pn::Real=0.0, rng=Random.default_rng())

Generate a continuous-wave (CW) field with optional broadband temporal noise.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: CW power [W]
  - `Pn::Real`: Power of the additive temporal noise floor [W] (default: 0.0)
  - `rng`: random source for the noise realization

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

A constant-amplitude field `√Pmax` with, if `Pn > 0`, an additive seed of
amplitude `√Pn` and an *independent* uniformly random phase in every time bin:

    A(t) = √Pmax + √Pn · exp(i·2π·U(t)),   U(t) ~ Uniform[0, 1)

For a physically grounded quantum (one-photon-per-mode) or RIN seed on top of a
clean field, use [`add_noise`](@ref) instead.
"""
function cw_pulse(
    grid::Grid, Pmax::Real; Pn::Real=0.0, rng::Random.AbstractRNG=Random.default_rng()
)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    Pn >= 0 || throw(ArgumentError("Noise power must be non-negative"))

    N = grid.N

    # Constant-amplitude CW field in the time domain
    At = fill(ComplexF64(sqrt(Pmax)), N)

    # Add noise if requested — a fresh, independent random phase per time bin
    if Pn > 0
        At .+= sqrt(Pn) .* cis.(2π .* rand(rng, N))
    end

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

end # module
