using FFTW
using FFTW: fftshift!
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

import ..build_physics_model, ..PhysicsModel
import ..AdaptiveSSFM, ..propagate, ..Pulse, ..SimParams, ..Solution

"""
    propagate(model::PhysicsModel, pulse::Pulse, params::SimParams, solver::AdaptiveSSFM, progress::Bool)

Propagate pulse using Phase-Controlled Adaptive Split-Step Fourier Method (AdaptiveSSFM).
Step size is controlled via local nonlinear phase shift:
    dz_opt = clamp(solver.phi_max / (gamma_phys * P_max + 1e-15), solver.dz_min, solver.dz_max)
"""
function propagate(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    solver::AdaptiveSSFM,
    progress::Bool,
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    z_end::Float64 = params.medium.length

    U = copy(pulse.AW)

    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = zeros(ComplexF64, N, n_saves)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    Aw_out[:, 1] .= fftshift(U)

    prog = progress ? Progress(n_saves - 1; desc="Adaptive SSFM... ", color=:cyan) : nothing

    save_points = range(0.0, z_end; length=n_saves)

    U_mid = similar(U)
    U_nl = similar(U)
    u_mid = similar(pulse.At)
    u_temp = similar(pulse.At)

    z = 0.0
    save_idx = 2

    # Initial peak power and step size
    Pmax = maximum(abs2, pulse.At)
    gamma_phys = (model.gamma isa Number) ? model.gamma * model.omega0 : model.gamma(0.0) * model.omega0
    dz = solver.dz_init === nothing ? clamp(solver.phi_max / (gamma_phys * Pmax + 1e-12), solver.dz_min, solver.dz_max) : solver.dz_init

    while z < z_end && save_idx <= n_saves
        z_target = save_points[save_idx]

        while z < z_target - 1e-14
            dz_step = min(dz, z_target - z)

            exp_half_dz_D = exp.(model.D .* (dz_step / 2.0))
            @. U_mid = U * exp_half_dz_D

            mul!(u_mid, model.to_time, U_mid)
            Nu = model.nonlinear_function(u_mid, model, z + 0.5 * dz_step)

            @. U_nl = U_mid + dz_step * Nu
            @. U = U_nl * exp_half_dz_D
            z += dz_step

            # Adapt step size based on peak power
            mul!(u_temp, model.to_time, U)
            Pmax_curr = maximum(abs2, u_temp)
            gamma_curr = (model.gamma isa Number) ? model.gamma * model.omega0 : model.gamma(z) * model.omega0
            dz_suggest = solver.phi_max / (gamma_curr * Pmax_curr + 1e-15)
            dz = clamp(min(1.5 * dz_step, dz_suggest), solver.dz_min, solver.dz_max)
        end

        # Save state
        z_out[save_idx] = z
        copyto!(model.buf_f1, U)
        fftshift!(@view(Aw_out[:, save_idx]), model.buf_f1)
        mul!(u_temp, model.to_time, model.buf_f1)
        copyto!(@view(At_out[:, save_idx]), u_temp)

        if !isnothing(prog)
            update!(prog, save_idx - 1)
        end

        save_idx += 1
    end

    return z_out, At_out, Aw_out
end
