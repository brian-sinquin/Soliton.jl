using FFTW
using FFTW: fftshift!
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

import ..build_physics_model, ..PhysicsModel
import ..SSFM, ..propagate, ..VectorialPulse, ..SimParams

"""
    propagate(model::PhysicsModel, pulse::VectorialPulse, params::SimParams, solver::SSFM, progress::Bool)

Propagate a `VectorialPulse` using the fixed-step Symmetric Split-Step Fourier Method (SSFM) for Coupled GNLSE.
"""
function propagate(
    model::PhysicsModel,
    pulse::VectorialPulse,
    params::SimParams,
    solver::SSFM,
    progress::Bool,
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    z_end::Float64 = params.medium.length

    # Initial condition in frequency domain: size N x 2
    U = copy(pulse.AW)

    # Storage for output
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, 2, n_saves)
    Aw_out = params.save_freq ? zeros(ComplexF64, N, 2, n_saves) : zeros(ComplexF64, 0, 0, 0)

    z_out[1] = 0.0
    At_out[:, :, 1] .= pulse.At
    if params.save_freq
        Aw_out[:, 1, 1] .= fftshift(U[:, 1])
        Aw_out[:, 2, 1] .= fftshift(U[:, 2])
    end

    # Progress bar setup
    prog = progress ? Progress(n_saves - 1; desc="Vectorial SSFM... ", color=:blue) : nothing

    # Target save points
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

    # Initialize first step midpoint: linear half-step from z=0
    @. U_mid = U * exp_half_dz_D

    for save_idx in 2:n_saves
        # Propagate from save_points[save_idx-1] to save_points[save_idx]
        for step in 1:n_steps
            step_global = (save_idx - 2) * n_steps + step
            z_mid = (step_global - 0.5) * dz_eff

            # Transform to time domain (lab frame)
            mul!(u_mid, model.to_time, U_mid)

            # Evaluate Coupled Kerr nonlinearity at midpoint
            Nu = model.nonlinear_function(u_mid, model, z_mid)

            # Apply nonlinear step: Euler step at midpoint
            @. U_nl = U_mid + dz_eff * Nu

            if step < n_steps || save_idx < n_saves
                # Advance midpoint to next step via linear full step
                @. U_mid = U_nl * exp_full_dz_D
            end
        end

        # Land exactly on save point with final linear half-step for output
        @. U = U_nl * exp_half_dz_D
        z = save_points[save_idx]

        # Save state
        z_out[save_idx] = z
        
        # Save U components
        copyto!(model.buf_f1, U)
        if params.save_freq
            fftshift!(@view(Aw_out[:, 1, save_idx]), @view(model.buf_f1[:, 1]))
            fftshift!(@view(Aw_out[:, 2, save_idx]), @view(model.buf_f1[:, 2]))
        end

        # Transform back to time domain
        mul!(u_temp, model.to_time, model.buf_f1)
        copyto!(@view(At_out[:, 1, save_idx]), @view(u_temp[:, 1]))
        copyto!(@view(At_out[:, 2, save_idx]), @view(u_temp[:, 2]))

        if !isnothing(prog)
            update!(prog, save_idx - 1)
        end
    end

    return z_out, At_out, Aw_out
end
