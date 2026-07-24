using FFTW
using FFTW: fftshift!
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

import ..build_physics_model, ..PhysicsModel
import ..SSFM, ..propagate

"""
    propagate(model::PhysicsModel, pulse::Pulse, params::SimParams, solver::SSFM, progress::Bool)

Propagate the pulse using the fixed-step Symmetric Split-Step Fourier Method (SSFM).
"""
function propagate(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    solver::SSFM,
    progress::Bool,
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    z_end::Float64 = params.medium.length

    # Initial condition in frequency domain
    U = copy(pulse.AW)

    # Storage for output. Contiguous columns for efficient writes.
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= fftshift(U)

    # Progress bar setup
    prog = progress ? Progress(n_saves - 1; desc="SSFM Propagation... ", color=:green) : nothing

    # Save target points
    save_points = range(0.0, z_end; length=n_saves)

    # Pre-allocate SSFM working arrays
    U_mid = similar(U)
    U_nl = similar(U)
    u_mid = similar(pulse.At)
    u_temp = similar(pulse.At)
    
    # Calculate step count per save interval
    dz_save = z_end / (n_saves - 1)
    n_steps = max(1, round(Int, dz_save / solver.dz))
    dz_eff = dz_save / n_steps

    # Pre-compute exp(D * dz_eff / 2) and exp(D * dz_eff) operators
    exp_half_dz_D = exp.(model.D .* (dz_eff / 2.0))
    exp_full_dz_D = exp.(model.D .* dz_eff)

    z = 0.0

    for save_idx in 2:n_saves
        # Propagate from save_points[save_idx-1] to save_points[save_idx]
        for step in 1:n_steps
            z_mid = z + (step - 0.5) * dz_eff

            if step == 1
                # First step: linear half step from z to z_mid
                @. U_mid = U * exp_half_dz_D
            else
                # Subsequent steps: linear full step from midpoint of prev step to midpoint of current step
                @. U_mid = U_nl * exp_full_dz_D
            end

            # Transform to time domain (lab frame)
            mul!(u_mid, model.to_time, U_mid)

            # Evaluate nonlinear operator at midpoint
            Nu = model.nonlinear_function(u_mid, model, z_mid)

            # Apply nonlinear step: Euler step at the midpoint
            @. U_nl = U_mid + dz_eff * Nu
        end

        # After n_steps, apply final linear half step to land exactly on the save point
        @. U = U_nl * exp_half_dz_D
        z = save_points[save_idx]

        # Save state
        z_out[save_idx] = z
        copyto!(model.buf_f1, U)
        fftshift!(@view(Aw_out[:, save_idx]), model.buf_f1)
        mul!(u_temp, model.to_time, model.buf_f1)
        copyto!(@view(At_out[:, save_idx]), u_temp)

        if !isnothing(prog)
            update!(prog, save_idx - 1)
        end
    end

    return z_out, At_out, Aw_out
end
