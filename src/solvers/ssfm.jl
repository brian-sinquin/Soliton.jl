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
    rng::AbstractRNG=default_rng(),
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    z_end::Float64 = params.medium.length

    # Initial condition in frequency domain
    U = copy(pulse.AW)

    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = params.save_freq ? zeros(ComplexF64, N, n_saves) : zeros(ComplexF64, 0, 0)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    if params.save_freq
        Aw_out[:, 1] .= fftshift(U)
    end

    # Progress bar setup
    prog = if progress
        Progress(n_saves - 1; desc="SSFM Propagation... ", color=:green)
    else
        nothing
    end

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

    # Initialize first step midpoint: linear half-step from z=0
    @. U_mid = U * exp_half_dz_D

    for save_idx in 2:n_saves
        # Propagate from save_points[save_idx-1] to save_points[save_idx]
        for step in 1:n_steps
            step_global = (save_idx - 2) * n_steps + step
            z_mid = (step_global - 0.5) * dz_eff

            # Transform to time domain (lab frame)
            mul!(u_mid, model.to_time, U_mid)

            # Evaluate nonlinear operator at midpoint
            Nu = model.nonlinear_function(u_mid, model, z_mid)

            # Apply nonlinear step: Euler step at midpoint
            @. U_nl = U_mid + dz_eff * Nu

            # ASE noise (AmplifyingMedium only; no-op otherwise)
            inject_ase_noise!(U_nl, u_mid, model, dz_eff, rng)

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
        # `fftshift!` and `mul!` only read their source, so `U` is used
        # directly rather than staged through `model.buf_f1` — one less
        # N-element copy per save point, and one less place where active
        # pulse data flows through a model-owned scratch buffer (which is
        # what forces `Duplicated(model, ...)` under Enzyme; see
        # `docs/src/dev/adjoint_ad.md`).
        if params.save_freq
            fftshift!(@view(Aw_out[:, save_idx]), U)
        end
        mul!(u_temp, model.to_time, U)
        copyto!(@view(At_out[:, save_idx]), u_temp)

        if !isnothing(prog)
            update!(prog, save_idx - 1)
        end
    end

    return z_out, At_out, Aw_out
end
