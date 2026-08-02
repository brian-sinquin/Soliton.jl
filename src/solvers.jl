"""
    GNLSESolver

Abstract base type for all GNLSE propagation solvers.
"""
abstract type GNLSESolver end

function propagate end

"""
    ERK4IP(; rtol=1e-6, atol=1e-8, dz_init=nothing, dz_min=0.0)

Embedded Runge-Kutta 4(3) solver in the Interaction Picture.

# Fields

  - `rtol::Float64`: Relative error tolerance for adaptive stepping.
  - `atol::Float64`: Absolute error tolerance for adaptive stepping.
  - `dz_init::Union{Float64, Nothing}`: Optional initial step size [m].
  - `dz_min::Float64`: Minimum allowed step size [m]. If the adaptive controller
    shrinks below this floor (e.g. because the field diverged to NaN/Inf, or the
    local error never converges), propagation stops with a clear error instead
    of looping forever. `0.0` (default) auto-selects `max(1e-15, fiber_length * 1e-12)`.
"""
struct ERK4IP <: GNLSESolver
    rtol::Float64
    atol::Float64
    dz_init::Union{Float64, Nothing}
    dz_min::Float64

    function ERK4IP(;
        rtol::Real=1e-6,
        atol::Real=1e-8,
        dz_init::Union{Real, Nothing}=nothing,
        dz_min::Real=0.0,
    )
        rtol > 0 || throw(ArgumentError("rtol must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))
        dz_min >= 0 || throw(ArgumentError("dz_min must be non-negative"))
        dz_val = dz_init === nothing ? nothing : Float64(dz_init)
        new(Float64(rtol), Float64(atol), dz_val, Float64(dz_min))
    end
end

"""
    SSFM(dz)

Fixed-step Symmetric Split-Step Fourier Method (SSFM) solver configuration.
"""
struct SSFM <: GNLSESolver
    dz::Float64

    function SSFM(dz::Real)
        dz > 0 || throw(ArgumentError("Fixed step size dz must be positive"))
        return new(Float64(dz))
    end
end

"""
    AdaptiveSSFM(; phi_max=1e-3, dz_init=nothing, dz_min=1e-12, dz_max=1.0)

Phase-controlled adaptive Split-Step Fourier Method (SSFM) solver.
Controls step size via maximum nonlinear phase shift per step:
Δz_opt = phi_max / (γ_phys * P_max).
"""
struct AdaptiveSSFM <: GNLSESolver
    phi_max::Float64
    dz_init::Union{Float64, Nothing}
    dz_min::Float64
    dz_max::Float64

    function AdaptiveSSFM(;
        phi_max::Real=1e-3,
        dz_init::Union{Real, Nothing}=nothing,
        dz_min::Real=1e-12,
        dz_max::Real=1.0,
    )
        phi_max > 0 || throw(ArgumentError("phi_max must be positive"))
        dz_val = dz_init === nothing ? nothing : Float64(dz_init)
        new(Float64(phi_max), dz_val, Float64(dz_min), Float64(dz_max))
    end
end
