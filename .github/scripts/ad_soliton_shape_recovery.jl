"""
Enzyme reverse-mode AD *with respect to the pulse shape itself* — recovering
the fundamental soliton profile by constraining propagation invariance.

Every earlier `ad_*` example (including `ad_ssfm_enzyme_compression.jl`)
differentiates a *loss* through the SSFM solver with respect to a handful of
*fiber* parameters (Sellmeier coefficients, β₂, ...) while the input pulse
shape stays fixed. This example instead treats the pulse's own temporal
amplitude profile as the optimization variable — a `Enzyme.Duplicated`
`Pulse` (grid.N-dimensional) rather than `Duplicated` fiber parameters. The
fiber (`PhysicsModel`) is never itself optimized, but — as this script's own
gradient check first discovered the hard way — it still needs to be marked
`Duplicated` (with a throwaway zero shadow) rather than `Const`, since
`propagate()` reuses `model`'s scratch buffers for genuinely Active,
pulse-derived data; see the comment at its construction below. This is a
different differentiation surface than roadmap step 2 exercised, so it
re-validates against finite differences from scratch rather than assuming
the earlier result carries over.

The defining property of the fundamental (N=1) soliton is that it does not
change shape as it propagates (only a spatially-uniform phase accumulates).
So instead of hand-deriving the sech profile, this script *discovers* it:
starting from a deliberately wrong initial guess (a Gaussian, not a sech)
at a fixed total energy, gradient descent shrinks

    L(theta) = sum( (|propagate(theta)| - |theta|)^2 )

with the gradient supplied entirely by Enzyme through the real nonlinear
SSFM solver. Since energy is exactly conserved by lossless, Raman-free
propagation regardless of shape (a lesson already learned the hard way in
this codebase's AD examples — see `ad_ssfm_enzyme_compression.jl`'s and
`docs/src/dev/adjoint_ad.md`'s notes on degenerate objectives), matching
`L=0` would be trivially satisfied by shrinking the pulse to zero unless
energy is *also* held fixed. So after every Adam step, `theta` is projected
back onto the fixed-energy hypersphere (a simple rescale, no AD needed for
that part) — turning optimization into a search over *shape only*. For a
fixed fiber (β₂, γ) and a fixed energy matching the N=1 soliton condition,
the unique shape-invariant solution (up to translation) is the analytic
fundamental sech soliton, so a successful run should converge theta to that
shape without it ever being hard-coded as the target.

Run with:
    julia --project=. -e 'import Pkg; Pkg.add(["Enzyme", "Plots"])'
    julia --project=. examples/ad_soliton_shape_recovery.jl
"""

using Soliton
using Enzyme
using LinearAlgebra: mul!
using Plots
using Printf

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^72)
println("Enzyme AD w.r.t. pulse shape: recovering the fundamental soliton")
println("by constraining propagation invariance")
println("="^72)

# --- Fixed physics: fiber + target energy ---
N = 2^11
time_window = 4e-12
lambda0 = 1550e-9
gamma0 = 0.11
beta2 = -20e-27           # s^2/m, anomalous
T0 = 100e-15               # target soliton time-width (not FWHM)
P0 = abs(beta2) / (gamma0 * T0^2)   # N=1 fundamental soliton peak power
E0 = 2 * P0 * T0                     # sech energy = P0 * 2T0: the fixed energy budget
L_D = T0^2 / abs(beta2)
# CI run 32351102419 found the shape-mismatch signal at L=L_D was nearly
# degenerate: max|At_out|-|theta| ~ 3.5e-8, loss ~2e-13. A Gaussian barely
# reshapes over just one dispersion length at these parameters, so both the
# finite-difference check (self-inconsistent by 19-475% between two step
# sizes -- clear FD noise, not a real signal) and the Adam loop (loss never
# moved past floating-point-scale noise) had nothing real to work with.
# Propagating several dispersion lengths gives a genuine, O(1) mismatch.
L = 3 * L_D
n_steps = 90                         # keep the per-step resolution from before (3x steps for 3x L)
z_saves = 2

grid = create_grid(N, time_window, lambda0)

@printf(
    "Target fundamental soliton: T0=%.1f fs, P0=%.3f W, E0=%.4f pJ, L_D=%.3f m\n",
    T0 * 1e15, P0, E0 * 1e12, L_D
)

dz = L / n_steps
medium = Medium(L, gamma0, 0.0, [beta2], lambda0)
params = SimParams(;
    medium=medium,
    z_saves=z_saves,
    raman_model=nothing,
    self_steepening=false,
    solver=SSFM(dz),
    save_freq=false,
)

# Nothing in `model` is optimized here, but it still needs a *shadow*:
# `Soliton.propagate` reuses `model.buf_f1` as scratch space for values
# that are genuinely derived from the (Duplicated) pulse -- e.g.
# `copyto!(model.buf_f1, U)` followed by reading it back via `mul!`. If
# `model` is marked `Const`, Enzyme allocates no shadow for that buffer, so
# writing Active data through it silently drops the gradient (confirmed by
# a bisecting test in test/test_enzyme.jl: Enzyme returned exactly 0.0
# against a finite-difference gradient of 1.99 with `Const(model)`, and
# matched to 1.5e-9 once `model` was `Duplicated` with a throwaway zero
# shadow instead). We don't need model's own gradient, just real shadow
# memory for the buffers it lends out. `pulse` is built once and mutated in
# place each call -- the same "build once outside the closure" idiom used
# throughout this file's sibling examples, just applied to the
# *optimization variable* now instead of a `Const` input.
At0 = zeros(ComplexF64, grid.N)
model = Soliton.build_physics_model(grid, params, At0)
pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

function shape_invariance_loss(
    theta::AbstractVector{<:Real},
    model::Soliton.PhysicsModel,
    pulse::Pulse,
    params::SimParams,
)
    @. pulse.At = complex(theta, 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
    return sum(abs2, abs.(At[:, end]) .- abs.(theta))
end

# --- Wrong initial guess: a Gaussian (not sech), scaled to the target energy ---
theta0 = sqrt(P0) .* exp.(-grid.t .^ 2 ./ (2 * T0^2))
theta0 .*= sqrt(E0 / sum(abs2, theta0))

# --- Sanity check: Enzyme gradient vs. independent finite differences ---
# This is a *different* differentiation surface than the betas-gradient
# example (gradient w.r.t. the field itself, `Duplicated(pulse, ...)`
# instead of `Duplicated(model, ...)`), so it is re-verified from scratch.
function loss_of_theta_fd(theta_vec::AbstractVector{<:Real})
    p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
    @. p.At = complex(theta_vec, 0.0)
    mul!(p.AW, model.to_freq, p.At)
    _, At, _ = Soliton.propagate(model, p, params, params.solver, false)
    return sum(abs2, abs.(At[:, end]) .- abs.(theta_vec))
end

dtheta_check = zero(theta0)
shadow_check = Enzyme.make_zero(pulse)
model_shadow_check = Enzyme.make_zero(model)
Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    shape_invariance_loss,
    Enzyme.Active,
    Enzyme.Duplicated(theta0, dtheta_check),
    Enzyme.Duplicated(model, model_shadow_check),
    Enzyme.Duplicated(pulse, shadow_check),
    Enzyme.Const(params),
)

loss0 = loss_of_theta_fd(theta0)
At_end0 = let
    p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
    @. p.At = complex(theta0, 0.0)
    mul!(p.AW, model.to_freq, p.At)
    _, At, _ = Soliton.propagate(model, p, params, params.solver, false)
    At[:, end]
end
diff0 = abs.(At_end0) .- abs.(theta0)
@printf(
    "Initial loss = %.6e; max|diff| = %.6e, mean|diff| = %.6e (sanity check on signal scale)\n",
    loss0, maximum(abs.(diff0)), sum(abs.(diff0)) / length(diff0)
)

check_idx = unique([argmax(theta0), N ÷ 4, N ÷ 2 + N ÷ 8, N ÷ 2 - N ÷ 8])
println("Gradient check (Enzyme vs finite differences) at a few grid points:")
println("(two FD step sizes shown -- if the estimate is unstable between them, that's")
println("finite-difference floating-point noise, not necessarily an Enzyme error)")
for i in check_idx
    h1 = 1e-6 * sqrt(P0)
    h2 = 1e-4 * sqrt(P0)
    g_fd(h) = begin
        thp = copy(theta0)
        thp[i] += h
        thm = copy(theta0)
        thm[i] -= h
        (loss_of_theta_fd(thp) - loss_of_theta_fd(thm)) / (2h)
    end
    g_fd1 = g_fd(h1)
    g_fd2 = g_fd(h2)
    reldiff1 = abs(dtheta_check[i] - g_fd1) / max(abs(g_fd1), 1e-300)
    fd_selfdiff = abs(g_fd1 - g_fd2) / max(abs(g_fd2), 1e-300)
    @printf(
        "  idx %4d: theta0=%.3e  Enzyme=%.6e  FD(h=%.1e)=%.6e  FD(h=%.1e)=%.6e  rel.diff(Enzyme,FD1)=%.3e  rel.diff(FD1,FD2)=%.3e\n",
        i, theta0[i], dtheta_check[i], h1, g_fd1, h2, g_fd2, reldiff1, fd_selfdiff
    )
end
println()

# --- Adam optimization loop, gradient supplied entirely by Enzyme ---
theta = copy(theta0)
mvec = zeros(N)
vvec = zeros(N)
adam_b1, adam_b2, eps_adam, lr = 0.9, 0.999, 1e-8, 0.05
n_iters = 300
loss_history = zeros(n_iters)

println("Recovering the soliton shape via Enzyme-computed gradients through the full SSFM solver...")
for k in 1:n_iters
    dtheta = zero(theta)
    shadow = Enzyme.make_zero(pulse)
    model_shadow = Enzyme.make_zero(model)
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        shape_invariance_loss,
        Enzyme.Active,
        Enzyme.Duplicated(theta, dtheta),
        Enzyme.Duplicated(model, model_shadow),
        Enzyme.Duplicated(pulse, shadow),
        Enzyme.Const(params),
    )
    loss_history[k] = loss_val

    mvec .= adam_b1 .* mvec .+ (1 - adam_b1) .* dtheta
    vvec .= adam_b2 .* vvec .+ (1 - adam_b2) .* dtheta .^ 2
    m_hat = mvec ./ (1 - adam_b1^k)
    v_hat = vvec ./ (1 - adam_b2^k)
    theta .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)

    # Project back onto the fixed-energy hypersphere: excludes the trivial
    # zero solution and makes this a search over *shape only*.
    theta .*= sqrt(E0 / sum(abs2, theta))

    if k % 25 == 0 || k == 1
        @printf("  iter %4d: loss = %.6e\n", k, loss_val)
    end
end

# --- Compare the recovered shape to the analytic fundamental soliton ---
sech_target = sqrt(P0) .* (1 ./ cosh.(grid.t ./ T0))
rel_err_final = sqrt(sum(abs2, abs.(theta) .- sech_target)) / sqrt(sum(abs2, sech_target))
rel_err_initial = sqrt(sum(abs2, abs.(theta0) .- sech_target)) / sqrt(sum(abs2, sech_target))
@printf(
    "\nRelative L2 error vs. analytic sech soliton: initial guess = %.4f, recovered shape = %.4f\n",
    rel_err_initial, rel_err_final
)

loss_initial = shape_invariance_loss(theta0, model, pulse, params)
loss_final = shape_invariance_loss(theta, model, pulse, params)
@printf(
    "Shape-invariance loss: initial guess = %.6e, recovered shape = %.6e\n",
    loss_initial, loss_final
)

# --- Plot 1: convergence ---
plt1 = plot(
    1:n_iters,
    loss_history;
    xlabel="Adam iteration",
    ylabel="Shape-invariance loss [W^2, a.u.]",
    title="Recovering the fundamental soliton via Enzyme-AD invariance",
    label="loss",
    linewidth=2,
    yscale=:log10,
)
savefig(plt1, joinpath(output_dir, "soliton_recovery_convergence.png"))
println("Saved: ", joinpath(output_dir, "soliton_recovery_convergence.png"))

# --- Plot 2: recovered shape vs. initial guess vs. analytic sech ---
t_fs = grid.t .* 1e15
window = abs.(t_fs) .< 500
plt2 = plot(
    t_fs[window],
    theta0[window];
    xlabel="Time [fs]",
    ylabel="Amplitude [sqrt(W)]",
    title="Recovered pulse shape vs. analytic fundamental soliton",
    label="initial guess (Gaussian)",
    linewidth=2,
    linestyle=:dash,
)
plot!(plt2, t_fs[window], abs.(theta)[window]; label="recovered shape (Enzyme-AD)", linewidth=2)
plot!(plt2, t_fs[window], sech_target[window]; label="analytic sech soliton", linewidth=2, linestyle=:dot)
savefig(plt2, joinpath(output_dir, "soliton_recovery_shape.png"))
println("Saved: ", joinpath(output_dir, "soliton_recovery_shape.png"))
