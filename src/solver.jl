"""
Main solver interface following gnlse-python conventions.

Reference: gnlse-python GNLSE.run()
"""

"""
    solve(pulse::Pulse, params::SimParams; progress::Bool=true)

Solve GNLSE following gnlse-python conventions using adaptive ERK4IP method.

# Arguments

  - `pulse::Pulse`: Initial pulse condition
  - `params::SimParams`: Simulation parameters
  - `progress::Bool`: Show progress bar (default: true)

# Returns

  - `Solution`: Solution structure with t, W, omega0, Z, At, AW

# Example

```julia
# Create grid (natural SI units: s, m, W)
grid = create_grid(2^13, 12.5e-12, 835e-9)  # resolution, time_window [s], λ [m]

# Create medium: Medium(L[m], γ[1/W/m], loss[dB/m], betas[sⁿ/m], λ[m])
medium = Medium(0.15, 0.11, 0.0, [-11.83e-27], 835e-9)

# Create pulse
pulse = sech_pulse(grid, 10000.0, 50e-15)  # Pmax [W], FWHM [s]

# Setup simulation parameters
params = SimParams(;
    medium=medium,
    z_saves=200,
    raman_model=BlowWood(),
    self_steepening=false,
    rtol=1e-6,
    atol=1e-8,
)

# Solve
solution = solve(pulse, params)
```

# Notes

Integrates the GNLSE with the adaptive ERK4IP solver. All quantities are in
natural SI units; the envelope spectrum follows the standard optics convention
`AW = ifft(At)`.
"""
function propagate(pulse::Pulse, params::SimParams, solver::GNLSESolver; progress::Bool=true)
    model = build_physics_model(pulse.grid, params)
    return propagate(model, pulse, params, solver, progress)
end

function solve(pulse::Pulse, params::SimParams; progress::Bool=true)
    z, At, AW = propagate(pulse, params, params.solver; progress=progress)

    # Build solution
    grid = pulse.grid
    solution = Solution(
        grid.t,          # Time grid [s]
        grid.W,          # Absolute frequency [rad/s]
        grid.omega0,     # Central frequency [rad/s]
        z,               # Propagation distances [m]
        At,              # Time domain fields (N × z_saves)
        AW,              # Frequency domain fields (N × z_saves)
    )

    # Photon number is conserved by the GNLSE for a lossless fiber; a drift
    # indicates the step-size tolerance is too loose.
    if params.medium.loss == 0
        n = photon_number(solution)
        drift = abs(n[end] - n[1]) / n[1]
        drift > 1e-2 && @warn "Photon number drifted by " *
            "$(round(100 * drift; digits=2))% — consider a tighter `rtol`/`atol`."
    end

    return solution
end

"""
    solve(pulse::Pulse, stages::AbstractVector; progress::Bool=true)

Propagate an optical pulse through a sequence of stages (fibers, amplifiers, filters, etc.).

Returns a vector of results corresponding to each stage (either a `Solution` for propagation stages,
or a new `Pulse` for lumped elements / functional stages).
"""
function solve(pulse::Pulse, stages::AbstractVector; progress::Bool=true)
    current_pulse = pulse
    results = Any[]

    for (i, stage) in enumerate(stages)
        if stage isa SimParams
            # Fiber propagation stage (returns a Solution)
            sol = solve(current_pulse, stage; progress=progress)
            push!(results, sol)
            # Use final state of the fiber propagation for the next stage
            current_pulse = Pulse(sol.At[:, end], sol.AW[:, end], pulse.grid)
        elseif stage isa LumpedElement
            # Lumped element stage (returns a Pulse)
            current_pulse = apply(current_pulse, stage)
            push!(results, current_pulse)
        elseif stage isa Function
            # Custom functional stage (returns a Pulse)
            current_pulse = stage(current_pulse)
            push!(results, current_pulse)
        else
            throw(ArgumentError("Unsupported stage type: $(typeof(stage))"))
        end
    end

    return results
end
