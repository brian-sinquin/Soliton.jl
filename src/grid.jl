"""
Grid generation for GNLSE simulations in natural SI units.

Units: Time in s, frequency in rad/s, wavelength in m
"""

"""
    create_grid(resolution::Int, time_window::Real, wavelength::Real)

Create time-frequency grid for GNLSE simulations in natural SI units.

# Arguments

  - `resolution::Int`: Even number of grid points, at least 2 (power of 2 recommended)
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
    resolution >= 2 || throw(ArgumentError("resolution must be >= 2"))
    iseven(resolution) || throw(ArgumentError("resolution must be even (odd N produces inconsistent frequency grids)"))
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
