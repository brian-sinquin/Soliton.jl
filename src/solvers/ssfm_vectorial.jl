using FFTW
using FFTW: fftshift!
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

import ..build_physics_model, ..VectorialPhysicsModel
import ..SSFM, ..propagate, ..VectorialPulse, ..SimParams

"""
    propagate(model::VectorialPhysicsModel, pulse::VectorialPulse, params::SimParams, solver::SSFM, progress::Bool)

Propagate a `VectorialPulse` using the fixed-step Symmetric Split-Step Fourier Method (SSFM) for Coupled GNLSE.
"""
function propagate(
    model::VectorialPhysicsModel,
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
    Aw_out = zeros(ComplexF64, N, 2, n_saves)

    z_out[1] = 0.0
    At_out[:, :, 1] .= pulse.At
    Aw_out[:, 1, 1] .= fftshift(U[:, 1])
    Aw_out[:, 2, 1] .= fftshift(U[:, 2])

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
    exp_half_dz_Dx = exp.(model.Dx .* (dz_eff / 2.0))
    exp_half_dz_Dy = exp.(model.Dy .* (dz_eff / 2.0))
    exp_full_dz_Dx = exp.(model.Dx .* dz_eff)
    exp_full_dz_Dy = exp.(model.Dy .* dz_eff)

    z = 0.0

    for save_idx in 2:n_saves
        # Propagate from save_points[save_idx-1] to save_points[save_idx]
        for step in 1:n_steps
            z_mid = z + (step - 0.5) * dz_eff

            if step == 1
                # First step: linear half step from z to z_mid
                @. U_mid[:, 1] = U[:, 1] * exp_half_dz_Dx
                @. U_mid[:, 2] = U[:, 2] * exp_half_dz_Dy
            else
                # Subsequent steps: linear full step
                @. U_mid[:, 1] = U_nl[:, 1] * exp_full_dz_Dx
                @. U_mid[:, 2] = U_nl[:, 2] * exp_full_dz_Dy
            end

            # Transform to time domain (lab frame)
            mul!(@view(u_mid[:, 1]), model.to_time, @view(U_mid[:, 1]))
            mul!(@view(u_mid[:, 2]), model.to_time, @view(U_mid[:, 2]))

            # Evaluate Coupled Kerr nonlinearity at midpoint
            Nu = model.nonlinear_function(u_mid, model, z_mid)

            # Apply nonlinear step: Euler step at midpoint
            @. U_nl = U_mid + dz_eff * Nu
        end

        # After n_steps, apply final linear half step to land exactly on the save point
        @. U[:, 1] = U_nl[:, 1] * exp_half_dz_Dx
        @. U[:, 2] = U_nl[:, 2] * exp_half_dz_Dy
        z = save_points[save_idx]

        # Save state
        z_out[save_idx] = z
        
        # Save x component
        copyto!(@view(model.buf_f1[:, 1]), @view(U[:, 1]))
        fftshift!(@view(Aw_out[:, 1, save_idx]), @view(model.buf_f1[:, 1]))
        mul!(@view(u_temp[:, 1]), model.to_time, @view(model.buf_f1[:, 1]))
        copyto!(@view(At_out[:, 1, save_idx]), @view(u_temp[:, 1]))

        # Save y component
        copyto!(@view(model.buf_f1[:, 2]), @view(U[:, 2]))
        fftshift!(@view(Aw_out[:, 2, save_idx]), @view(model.buf_f1[:, 2]))
        mul!(@view(u_temp[:, 2]), model.to_time, @view(model.buf_f1[:, 2]))
        copyto!(@view(At_out[:, 2, save_idx]), @view(u_temp[:, 2]))

        if !isnothing(prog)
            update!(prog, save_idx - 1)
        end
    end

    return z_out, At_out, Aw_out
end
