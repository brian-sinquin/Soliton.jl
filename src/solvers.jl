"""
    GNLSESolver

Abstract base type for all GNLSE propagation solvers.
"""
abstract type GNLSESolver end

function propagate end

"""
    ERK4IP(; rtol=1e-6, atol=1e-8, dz_init=nothing)

Embedded Runge-Kutta 4(3) solver in the Interaction Picture.

# Fields
- `rtol::Float64`: Relative error tolerance for adaptive stepping.
- `atol::Float64`: Absolute error tolerance for adaptive stepping.
- `dz_init::Union{Float64, Nothing}`: Optional initial step size [m].
"""
struct ERK4IP <: GNLSESolver
    rtol::Float64
    atol::Float64
    dz_init::Union{Float64, Nothing}

    function ERK4IP(; rtol::Real=1e-6, atol::Real=1e-8, dz_init::Union{Real, Nothing}=nothing)
        rtol > 0 || throw(ArgumentError("rtol must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))
        dz_val = dz_init === nothing ? nothing : Float64(dz_init)
        new(Float64(rtol), Float64(atol), dz_val)
    end
end
