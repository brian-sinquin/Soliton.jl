"""
Enzyme reverse-mode AD *through the actual nonlinear SSFM solver* — pulse
compression by dispersion design.

The three earlier `ad_*` examples all differentiate only the linear
dispersion math (`propagation_constant`/`dispersion_operator`) with
`ForwardDiff`, then separately run a full nonlinear simulation to check the
result. This example is different: the gradient itself is taken *through*
`Soliton.propagate` — FFTs, the nonlinear (Kerr) step, and the linear
half-steps all included — using `Enzyme.autodiff(Reverse, ...)`. This is
roadmap step 2 of `docs/src/dev/adjoint_ad.md`, validated in
`test/test_enzyme.jl`; this script is the first concrete design problem
built on top of it, following the exact "build `model`/`pulse` once outside
the differentiated closure, mutate `model.D` in place, pass everything else
as `Const`" idiom worked out there.

Problem: starting from a fiber with β₂ = -20 ps²/km (anomalous dispersion,
typical for pulse compression), adjust β₂ so that a 500 W, 100 fs sech pulse
comes out of a 5 cm fiber as compressed (narrow) as possible, judged by the
propagated pulse's own temporal second moment ∫|A(z=L,t)|²t²dt — a smooth,
`sum`-based reduction (matching the only reduction pattern validated for
Enzyme in this codebase so far) that shrinks as the pulse narrows around
t=0, for a *fixed* energy (energy is conserved under lossless, Raman-free
propagation, so it cannot itself be used as an objective — established
earlier in the same debugging session that produced `test/test_enzyme.jl`).

Only β₂ is optimized here (a single scalar): `medium.gamma`'s gradient is
not yet available (see `docs/src/dev/adjoint_ad.md`, "Still not done" —
`PhysicsModel.gamma` is an immutable `Float64` field with no in-place buffer
to mutate the way `.D` is mutated for `betas`).

Run with:
    julia --project=. -e 'import Pkg; Pkg.add(["Enzyme", "Plots"])'
    julia --project=. examples/ad_ssfm_enzyme_compression.jl
"""

using Soliton
using Enzyme
using FFTW: fftshift
using Plots
using Printf

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^72)
println("Enzyme AD through the full nonlinear SSFM solver: dispersion design")
println("for pulse compression (roadmap step 2 demonstration)")
println("="^72)

# --- Fixed physical setup ---
N = 2^12
time_window = 8e-12
lambda0 = 1550e-9
gamma0 = 0.11
L = 0.05
P0 = 500.0
FWHM = 100e-15
# CI run 32236352351 showed ~70% photon-number drift with n_steps=6: the
# nonlinear phase per step (gamma*P0*dz ~ 0.46 rad) was too large for the
# split-step scheme at this power/length. n_steps=60 keeps it ~0.046 rad.
n_steps = 60
z_saves = 2

grid = create_grid(N, time_window, lambda0)
pulse0 = sech_pulse(grid, P0, FWHM)
At0 = copy(pulse0.At)
AW0 = copy(pulse0.AW)
t2 = grid.t .^ 2

# β₂ in "ps²/km"-like O(1) units for well-conditioned Adam steps (same
# scaling convention as the ForwardDiff examples).
scale2 = 1e-27
theta0 = [-20.0]  # -20 ps^2/km, anomalous dispersion

dz = L / n_steps
medium0 = Medium(L, gamma0, 0.0, theta0 .* scale2, lambda0)
params = SimParams(;
    medium=medium0,
    z_saves=z_saves,
    raman_model=nothing,
    self_steepening=false,
    solver=SSFM(dz),
    save_freq=false,
)

# Built once, outside the differentiated closure — see test/test_enzyme.jl
# for why rebuilding `PhysicsModel`/`Pulse` *inside* the closure hits
# `EnzymeRuntimeActivityError` (a "conditionally active memory" mismatch
# Enzyme's static analysis can't resolve).
model = Soliton.build_physics_model(grid, params, At0)
pulse_const = Pulse(copy(At0), copy(AW0), grid)

function compression_loss(
    theta::AbstractVector{<:Real},
    model::Soliton.PhysicsModel,
    pulse::Pulse,
    params::SimParams,
    t2::AbstractVector{<:Real},
    scale::Real,
)
    m_disp = Medium(L, gamma0, 0.0, theta .* scale2, lambda0)
    model.D .= fftshift(dispersion_operator(pulse.grid, m_disp))
    _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
    return sum(abs2.(At[:, end]) .* t2) * scale
end

# --- Sanity check: Enzyme gradient vs. independent finite differences ---
function loss_of_theta_fd(theta1::Real)
    m = Soliton.build_physics_model(grid, params, At0)
    m_disp = Medium(L, gamma0, 0.0, [theta1] .* scale2, lambda0)
    m.D .= fftshift(dispersion_operator(grid, m_disp))
    _, At, _ = Soliton.propagate(m, pulse_const, params, params.solver, false)
    return sum(abs2.(At[:, end]) .* t2)
end

h = 1e-4 * abs(theta0[1])
g_fd = (loss_of_theta_fd(theta0[1] + h) - loss_of_theta_fd(theta0[1] - h)) / (2h)

dtheta0 = zero(theta0)
shadow0 = Enzyme.make_zero(model)
Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    compression_loss,
    Enzyme.Active,
    Enzyme.Duplicated(theta0, dtheta0),
    Enzyme.Duplicated(model, shadow0),
    Enzyme.Const(pulse_const),
    Enzyme.Const(params),
    Enzyme.Const(t2),
    Enzyme.Const(1.0),
)
@printf("Gradient check at theta0 = %.1f ps^2/km:\n", theta0[1])
@printf("  Enzyme (reverse-mode AD):  %.6e\n", dtheta0[1])
@printf("  Finite differences:        %.6e\n", g_fd)
@printf("  Relative difference:       %.3e\n", abs(dtheta0[1] - g_fd) / abs(g_fd))
println()

# CI run 32236352351 showed Adam making literally zero progress over 150
# iterations: the raw gradient (~5.4e-20) is many orders of magnitude
# smaller than Adam's eps=1e-8, so eps dominates m_hat/(sqrt(v_hat)+eps) and
# every step underflows to nothing. Rescale the objective (a constant
# multiplicative factor baked into `compression_loss` itself, so Enzyme
# differentiates the exact rescaled function used for optimization) so the
# gradient magnitude is O(1) and well clear of eps. Calibrated from the
# gradient check above rather than a hardcoded magic number, so it adapts
# if the physical parameters above change.
obj_scale = 0.1 / abs(dtheta0[1])

# --- Adam optimization loop, gradient supplied entirely by Enzyme ---
theta = copy(theta0)
mvec = zeros(1)
vvec = zeros(1)
adam_b1, adam_b2, eps_adam, lr = 0.9, 0.999, 1e-8, 0.05
n_iters = 150
loss_history = zeros(n_iters)

println("Optimizing beta2 via Enzyme-computed gradients through the full SSFM solver...")
for k in 1:n_iters
    dtheta = zero(theta)
    shadow = Enzyme.make_zero(model)
    # `ReverseWithPrimal` (rather than plain `Reverse`, used for the
    # gradient-only check above) also returns the primal loss value
    # alongside the gradient, avoiding a second, redundant forward pass
    # through the solver just to log the loss at each iteration.
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        compression_loss,
        Enzyme.Active,
        Enzyme.Duplicated(theta, dtheta),
        Enzyme.Duplicated(model, shadow),
        Enzyme.Const(pulse_const),
        Enzyme.Const(params),
        Enzyme.Const(t2),
        Enzyme.Const(obj_scale),
    )
    loss_history[k] = loss_val / obj_scale

    mvec .= adam_b1 .* mvec .+ (1 - adam_b1) .* dtheta
    vvec .= adam_b2 .* vvec .+ (1 - adam_b2) .* dtheta .^ 2
    m_hat = mvec ./ (1 - adam_b1^k)
    v_hat = vvec ./ (1 - adam_b2^k)
    theta .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

    if k % 25 == 0 || k == 1
        @printf("  iter %4d: loss = %.6e, beta2 = %.4f ps^2/km\n", k, loss_val, theta[1])
    end
end

@printf("\nbeta2: %.4f -> %.4f ps^2/km (Delta = %.1f%%)\n", theta0[1], theta[1], 100 * (theta[1] - theta0[1]) / theta0[1])

# --- Verification: run the real (non-differentiated) simulation before/after ---
println("\nVerifying via full nonlinear propagation (finer z-sampling for plotting)...")
verify_params(beta2_val) = SimParams(;
    medium=Medium(L, gamma0, 0.0, [beta2_val] .* scale2, lambda0),
    z_saves=50,
    raman_model=nothing,
    self_steepening=false,
    solver=SSFM(dz),
    save_freq=true,  # Pulse(::Solution) needs sol.AW; save_freq=false left it 0x0
)

pulse_in = sech_pulse(grid, P0, FWHM)
sol_before = solve(pulse_in, verify_params(theta0[1]); progress=false)
sol_after = solve(pulse_in, verify_params(theta[1]); progress=false)
pulse_before = Pulse(sol_before)
pulse_after = Pulse(sol_after)

energy(At) = sum(abs2, At)
rms_width(At) = sqrt(sum(abs2.(At) .* t2) / energy(At))

w_in = rms_width(pulse_in.At)
w_before = rms_width(pulse_before.At)
w_after = rms_width(pulse_after.At)
@printf("RMS temporal width: input = %.2f fs, before opt. output = %.2f fs, after opt. output = %.2f fs\n",
    w_in * 1e15, w_before * 1e15, w_after * 1e15)
@printf("Peak power: input = %.1f W, before opt. output = %.1f W, after opt. output = %.1f W\n",
    maximum(abs2.(pulse_in.At)), maximum(abs2.(pulse_before.At)), maximum(abs2.(pulse_after.At)))

# --- Plot 1: convergence ---
plt1 = plot(
    1:n_iters,
    loss_history;
    xlabel="Adam iteration",
    ylabel="Temporal second moment [m^2 * s^2... a.u.]",
    title="Enzyme-through-SSFM gradient descent",
    label="loss",
    linewidth=2,
    yscale=:log10,
)
savefig(plt1, joinpath(output_dir, "enzyme_ssfm_convergence.png"))
println("Saved: ", joinpath(output_dir, "enzyme_ssfm_convergence.png"))

# --- Plot 2: time-domain intensity, input vs. output before/after ---
t_fs = grid.t .* 1e15
window = abs.(t_fs) .< 500
plt2 = plot(
    t_fs[window],
    abs2.(pulse_in.At)[window];
    xlabel="Time [fs]",
    ylabel="Power [W]",
    title="Pulse compression via Enzyme-AD dispersion design",
    label="input",
    linewidth=2,
)
plot!(plt2, t_fs[window], abs2.(pulse_before.At)[window]; label=@sprintf("output, beta2=%.1f ps^2/km (before)", theta0[1]), linewidth=2)
plot!(plt2, t_fs[window], abs2.(pulse_after.At)[window]; label=@sprintf("output, beta2=%.4f ps^2/km (after)", theta[1]), linewidth=2)
savefig(plt2, joinpath(output_dir, "enzyme_ssfm_compression.png"))
println("Saved: ", joinpath(output_dir, "enzyme_ssfm_compression.png"))
