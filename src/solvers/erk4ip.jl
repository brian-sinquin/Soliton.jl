"""
Embedded Runge-Kutta 4(3) solver in interaction picture with adaptive stepping.

Implements 4th-order propagation with embedded 3rd-order error estimation for
automatic step size control. Uses the interaction picture formulation to reduce
stiffness from dispersion.

The key insight: Work in the interaction picture where Û = exp(-D̂z)U, which
removes fast oscillations from dispersion, allowing larger time steps.

# Algorithm (Balac & Mahé, 2013)

For ∂U/∂z = D̂U + N̂(U), transform to interaction picture:
Û(z) = exp(-D̂z)U(z)

Then: ∂Û/∂z = exp(-D̂z)N̂(exp(D̂z)Û)

This is integrated with RK4 using only 3 FFT pairs per step (not 5).

# Reference

S. Balac & A. Mahé, "Embedded Runge-Kutta scheme for step-size control in the
interaction picture method", Comput. Phys. Commun. 184, 1211-1219 (2013)
"""

using FFTW
using FFTW: fftshift!
using LinearAlgebra: mul!
using ProgressMeter: Progress, update!

# Import nonlinearity module functions
import ..build_physics_model, ..PhysicsModel

"""
    propagate(model::PhysicsModel, pulse::Pulse, params::SimParams, solver::ERK4IP, progress::Bool)

Propagate the pulse using the adaptive Runge-Kutta 4(3) interaction picture solver.
"""
function propagate(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    solver::ERK4IP,
    progress::Bool,
    rng::AbstractRNG=default_rng(),
)
    return _propagate_erk4ip!(
        model,
        pulse,
        params,
        progress,
        solver.rtol,
        solver.atol,
        solver.dz_init,
        solver.dz_min,
        rng,
    )
end

"""
    propagate_erk4ip(pulse::Pulse, params::SimParams; rtol=1e-6, atol=1e-8, dz=nothing)

Embedded Runge-Kutta 4(3) method in interaction picture with adaptive stepping.

Solves ∂U/∂z = D̂U + N̂(U) using interaction picture transformation to handle
stiff dispersion operator efficiently.

# Parameters

  - `rtol`, `atol`: Error tolerances for adaptive stepping
  - `dz`: Initial step size [m] (auto-selected if `nothing`)
  - `progress`: Show progress bar

# Returns

Tuple of (`z`, `At`, `Aw`): propagation distances, time and frequency domain fields

# Algorithm Efficiency

Uses only 3 FFT pairs per accepted step (vs 5 in naive implementations) by
reusing computations and working in interaction picture throughout.

# Reference

S. Balac & A. Mahé, Comput. Phys. Commun. 184, 1211 (2013);
A. M. Heidt, J. Lightwave Technol. 27, 3984 (2009)
"""
function propagate_erk4ip(
    pulse::Pulse,
    params::SimParams;
    progress::Bool=true,
    rtol::Float64=1e-6,
    atol::Float64=1e-8,
    dz::Union{Float64, Nothing}=nothing,
)
    solver = ERK4IP(; rtol=rtol, atol=atol, dz_init=dz)
    return propagate(pulse, params, solver; progress=progress)
end

function _propagate_erk4ip!(
    model::PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    progress::Bool,
    rtol::Float64,
    atol::Float64,
    dz_init::Union{Float64, Nothing},
    dz_min_arg::Float64,
    rng::AbstractRNG=default_rng(),
)
    grid = pulse.grid
    N = grid.N
    n_saves = params.z_saves
    # `SimParams.medium` has the abstract field type `Medium`, so annotate the
    # extracted length to keep the step variables type-stable in the hot loop.
    z_end::Float64 = params.medium.length
    dz_min::Float64 = dz_min_arg > 0 ? dz_min_arg : max(1e-15, z_end * 1e-12)

    # Initial condition in frequency domain
    U = copy(pulse.AW)

    # Storage for output. Layout is (N, n_saves): each saved field is a
    # contiguous column, so per-save writes are contiguous.
    z_out = zeros(n_saves)
    At_out = zeros(ComplexF64, N, n_saves)
    Aw_out = params.save_freq ? zeros(ComplexF64, N, n_saves) : zeros(ComplexF64, 0, 0)

    z_out[1] = 0.0
    At_out[:, 1] .= pulse.At
    if params.save_freq
        Aw_out[:, 1] .= fftshift(U)
    end

    # Adaptive stepping setup. `dz` is declared Float64 so the boxed
    # Union{Float64,Nothing} argument cannot leak into the hot loop.
    z = 0.0
    dz::Float64 = dz_init === nothing ? z_end / 1000 : dz_init
    save_idx = 2
    z_saves = range(0, z_end; length=n_saves)

    # Pre-allocate workspace for ERK4IP with minimal buffers
    # Key: Work in FREQUENCY domain (interaction picture), nonlinear function
    # takes time-domain input and returns frequency-domain output (like C code)
    k1 = similar(U)          # RK stage 1 (frequency domain)
    k2 = similar(U)          # RK stage 2 (frequency domain)
    k3 = similar(U)          # RK stage 3 (frequency domain)
    k4 = similar(U)          # RK stage 4 (frequency domain)
    k5 = similar(U)          # RK stage 5 (frequency domain, for error)
    Nu = similar(U)          # Nonlinear term N(u) in frequency domain
    U_temp = similar(U)      # Temporary frequency-domain buffer
    u_temp = similar(pulse.At)  # Time-domain buffer for IFFT(U_temp)
    exp_half_dz_D = similar(U)   # exp(D̂·h/2) operator

    # For error estimation
    r = similar(U)      # Intermediate for error calculation
    U4_fft = similar(U)  # 4th order solution (frequency domain)
    U5_fft = similar(U)  # 3rd order solution for error (frequency domain)
    u4 = similar(pulse.At)  # 4th order solution (time domain)

    # Initialize progress bar
    prog = progress ? Progress(n_saves - 1; desc="ERK4IP: ", showspeed=true) : nothing

    # Main propagation loop
    step_count = 0
    rejected_steps = 0

    # Initialize Nu = N(U(0)) for FSAL property
    mul!(u_temp, model.to_time, U)  # frequency → time
    copyto!(Nu, model.nonlinear_function(u_temp, model, 0.0))

    cached_dz = -1.0

    # AmplifyingMedium perturbs U with ASE noise after each accepted step,
    # which invalidates the FSAL shortcut (k5 was evaluated on the
    # pre-noise field) — recompute Nu in that case only, so every other
    # medium keeps the cheap "3 FFT pairs per step" FSAL path.
    is_amplifying = haskey(model.aux_data, :noise_figure_db)

    while z < z_end && save_idx <= n_saves
        # Target for next save
        z_target = z_saves[save_idx]
        dz = min(dz, z_target - z)

        # Dispersion half-step operator for this step size (cached when dz is unchanged)
        if dz != cached_dz
            @. exp_half_dz_D = exp(0.5 * dz * model.D)
            cached_dz = dz
        end

        # ============================================================
        # ERK4(3) in Interaction Picture - Following SPIP C code exactly
        # ============================================================
        # Transform to interaction picture: û = exp(-D̂z)U
        # Initial condition for RK: û_ip = exp(D̂h/2)·FFT(U)

        # Stage 1: k₁ = exp(D̂h/2)·N(U(z))  [FSAL: reuse from previous step]
        @. k1 = exp_half_dz_D * Nu

        # u₂ = exp(D̂h/2)·U + h/2·k₁, compute N(IFFT(u₂))
        @. U_temp = exp_half_dz_D * U + 0.5 * dz * k1
        mul!(u_temp, model.to_time, U_temp)  # frequency → time
        copyto!(k2, model.nonlinear_function(u_temp, model, z + 0.5 * dz))

        # u₃ = exp(D̂h/2)·U + h/2·k₂
        @. U_temp = exp_half_dz_D * U + 0.5 * dz * k2
        mul!(u_temp, model.to_time, U_temp)  # frequency → time
        copyto!(k3, model.nonlinear_function(u_temp, model, z + 0.5 * dz))

        # u₄ = exp(D̂h/2)·(exp(D̂h/2)·U + h·k₃) = exp(D̂h)·U + exp(D̂h/2)·h·k₃
        @. U_temp = exp_half_dz_D * (exp_half_dz_D * U + dz * k3)
        mul!(u_temp, model.to_time, U_temp)  # frequency → time
        copyto!(k4, model.nonlinear_function(u_temp, model, z + dz))

        # ============================================================
        # 4th Order Solution (following SPIP exactly)
        # ============================================================
        # r = exp(D̂h/2)·(exp(D̂h/2)·U + h·(k₁/6 + k₂/3 + k₃/3))
        @. r = exp_half_dz_D * (exp_half_dz_D * U + dz * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0))

        # U4_fft = r + h·k₄/6
        @. U4_fft = r + (dz / 6.0) * k4

        # Transform to time domain and compute k₅ = N(u₄)
        mul!(u4, model.to_time, U4_fft)  # frequency → time
        copyto!(k5, model.nonlinear_function(u4, model, z + dz))

        # ============================================================
        # 3rd Order Solution for Error Estimation (SPIP formula)
        # ============================================================
        # U5_fft = r + h·k₄/15 + h·k₅/10
        @. U5_fft = r + (dz / 15.0) * k4 + (dz / 10.0) * k5

        # Mixed absolute/relative error: per-mode error scaled by
        # (atol + rtol·|U|), then RMS over modes. A step is accepted when this
        # normalized error is ≤ 1.
        err2 = zero(Float64)
        @inbounds @simd for i in eachindex(U4_fft, U5_fft)
            sc = atol + rtol * abs(U4_fft[i])
            err2 += abs2(U4_fft[i] - U5_fft[i]) / (sc * sc)
        end
        local_error = sqrt(err2 / N)

        # ============================================================
        # Adaptive Step Size Control
        # ============================================================
        # dzₒₚₜ = clamp(0.9·(1/error)^(1/4), 0.5, 2.0) · dz
        safety = 0.9
        exponent = 0.25  # 1/(p+1) = 1/4 for the embedded 4(3) method

        if isfinite(local_error) && local_error <= 1.0
            # Accept step
            step_count += 1
            z += dz
            copyto!(U, U4_fft)

            if is_amplifying
                # u4 is the time-domain field at the end of this accepted step.
                inject_ase_noise!(U, u4, model, dz, rng)
                # Noise invalidated FSAL's k5 — recompute Nu for next step.
                mul!(u_temp, model.to_time, U)
                copyto!(Nu, model.nonlinear_function(u_temp, model, z))
            else
                # FSAL property: Nu for next step = k5 from this step
                copyto!(Nu, k5)
            end

            # Compute optimal step size for next iteration
            factor = safety * (1.0 / (local_error + 1e-300))^exponent
            factor = max(0.5, min(2.0, factor))  # Limit growth/shrink
            dz = factor * dz

            # Save output at target distance
            if z >= z_target - 1e-12 * z_end
                z_out[save_idx] = z

                # U is already the lab-frame field (RK4IP re-centers the
                # interaction picture each step), so no exp(D·z) is applied.
                copyto!(model.buf_f1, U)

                # Apply fftshift for monotonic frequency ordering in output if requested
                if params.save_freq
                    fftshift!(@view(Aw_out[:, save_idx]), model.buf_f1)
                end

                # Transform to time domain (lab frame)
                mul!(u_temp, model.to_time, model.buf_f1)
                copyto!(@view(At_out[:, save_idx]), u_temp)

                if !isnothing(prog)
                    update!(prog, save_idx - 1)
                end

                save_idx += 1
            end
        else
            # Reject step and shrink (same formula as the acceptance case).
            # A non-finite local_error means the field itself has already
            # diverged (NaN/Inf) — the adaptive formula would also evaluate to
            # NaN in that case, so fall back to a fixed hard shrink instead.
            rejected_steps += 1
            if isfinite(local_error)
                factor = safety * (1.0 / (local_error + 1e-300))^exponent
                factor = max(0.5, min(2.0, factor))
            else
                factor = 0.25
            end
            dz = factor * dz

            if dz < dz_min
                throw(
                    ErrorException(
                        "ERK4IP: step size collapsed to $dz m (below dz_min=$dz_min m) " *
                        "at z=$z m after $rejected_steps rejected steps. The field likely " *
                        "diverged (local_error=$local_error). Check medium parameters " *
                        "(gamma, loss/gain, dispersion) and pulse power for a numerically " *
                        "unstable configuration, or relax `rtol`/`atol`.",
                    ),
                )
            end
        end
    end

    if progress && !isnothing(prog)
        println("\n✓ Steps: $step_count accepted, $rejected_steps rejected")
    end

    return z_out, At_out, Aw_out
end
