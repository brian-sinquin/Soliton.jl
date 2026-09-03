"""
Finding the optimal operating point of a soliton-effect pulse compressor with
Enzyme **forward-mode** AD, and checking it against the published optimum.

# The experiment

Soliton-effect compression is a standard way to shorten a pulse: launch a
higher-order soliton (N > 1) into a fixed length of anomalous-dispersion fiber
and take the output where the pulse is narrowest. The design question a
practitioner actually faces is *how hard to pump*: for a given fiber (β₂, γ,
length L) and a given input duration, which input peak power compresses best?
Too little and the pulse barely reshapes; too much and it compresses before the
fiber end and has begun to break up again by the time it exits.

That optimum is known. Writing the soliton order

    N² = L_D / L_NL = γ P₀ T₀² / |β₂|,        z₀ = (π/2) L_D,  L_D = T₀²/|β₂|

the classic empirical relations for soliton-effect compression are

    z_opt / z₀ ≈ 0.32/N + 1.1/N²             (optimum fiber length)
    F_c        ≈ 4.1 N                        (compression factor)

(Agrawal, *Nonlinear Fiber Optics*, soliton-effect compression; also quoted in
the PCF compression literature, e.g. Foroni et al., `arXiv:physics/0610252`.)
Inverting the first relation for a **fixed** L gives the soliton order the fiber
is optimal for, and hence the ideal input peak power P₀ = N²|β₂|/(γT₀²).

The script never uses the formulas to *drive* the optimization: the gradient
comes entirely from Enzyme differentiating `Soliton.propagate`. They are
consulted only to score the answer.

# Where these relations are, and are not, valid

They are large-N asymptotics, and it is worth being explicit about that because
it dominates how well any simulation can be expected to agree. The N = 1 limit
settles it: a fundamental soliton propagates without reshaping at all, so its
true compression factor is exactly 1, while 4.1N predicts 4.1. The relation
cannot hold near N ≈ 1, and an earlier version of this script run at N ≈ 2.5
duly measured F_c = 7.07 against a predicted 10.22 — the optimum's *location*
reproduced to ~4 %, its *depth* did not.

So the experiment is set here at a higher soliton order, and two diagnostics are
printed rather than asserted. Both are confirmed by the run:

  1. a **convergence check** — the same operating point re-simulated with twice
     the steps and twice the temporal resolution. If the answer moved, the gap
     would be this script's numerics and nothing could be concluded about the
     formula. It does not: 13.74 fs baseline, 13.90 at 2x steps, 13.73 at 2x
     resolution, 13.90 at both — a 1.2 % spread, far too small to explain a
     20 %+ discrepancy in F_c.
  2. a **trend across N** — measured versus predicted compression factor at each
     order's own optimal length. If the relations really are large-N
     asymptotics, the measured/predicted ratio should climb toward 1 with N.
     It does, monotonically:

         N        1.5     2.0     3.0     4.0     5.0     8.0
         F_c/4.1N 0.381   0.508   0.685   0.781   0.835   0.918

At N ≈ 4 the AD optimum and the formula agree on the soliton order to **0.13 %**
(4.0051 vs 4.0000), against 4.19 % at N ≈ 2.5 — a thirtyfold improvement from
nothing but moving into the regime where the relation is meant to apply. The
compression factor agrees there too (12.83 measured vs 12.81 at the predicted
order), while `4.1N = 16.4` remains the optimistic asymptote it is.

# Choosing an objective that means "compressed"

The first version maximized output peak power, and CI showed why that is wrong:
dP/dN came back strongly positive and growing over the whole range. The free
parameter *is* the input power (P₀ ∝ N²), so "maximize output peak power" mostly
rewards pumping harder rather than compressing better.

The objective is instead the **effective duration**

    τ_eff = (∫I dt)² / ∫I² dt,       I(t) = |A(z=L, t)|²

invariant under I → αI, so it scores pulse *shape* only and cannot be gamed with
energy. It is smooth — two `sum` reductions, no `maximum` and no threshold
search, unlike a literal FWHM — which matters because it gets differentiated.
For a sech² intensity τ_eff = 3T₀ exactly.

τ_eff is a *proxy* for the FWHM, not a substitute: it integrates the whole trace
and so is dragged upward by the pedestal that soliton-effect compression always
leaves behind, which makes the τ_eff-based compression factor read well below the
FWHM-based one. What matters is that its *argmin* is in the right place. Both
factors are printed so the gap stays visible.

# Why forward mode

There is exactly **one** free parameter. Reverse mode would tape the whole
propagation to produce a single derivative; forward mode carries one tangent at
O(1) memory in the step count. This is the case
`docs/src/dev/ad_ecosystem_review.md` (finding 6) identifies as strictly better
served by forward mode, and it is the first example to use the forward rule in
`ext/SolitonEnzymeExt.jl`.

An exact derivative also changes *how* to optimize: instead of gradient descent
with a hand-tuned learning rate — which has already produced two real bugs in
this repository's examples — the optimum is found by bisecting the derivative to
its zero crossing. No step size, monotone convergence, bracket verified up front.

Run with:
    julia --project=. -e 'import Pkg; Pkg.add(["Enzyme", "Plots"])'
    julia --project=. examples/ad_soliton_compression_optimum.jl
"""

using Soliton
using Enzyme
using LinearAlgebra: mul!
using Plots
using Printf

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=" ^ 72)
println("Soliton-effect compression: AD-located optimum vs. the published one")
println("=" ^ 72)

# --- Fixed physics ---
lambda0 = 1550e-9            # m
gamma0 = 0.11                # 1/(W·m)
beta2 = -20e-27              # s²/m, anomalous
T0 = 100e-15                 # s, input sech half-width
m_sech = 2 * log(1 + sqrt(2))   # the FWHM↔T0 factor `sech_pulse` uses
FWHM_in = m_sech * T0
L_D = T0^2 / abs(beta2)
z0 = (pi / 2) * L_D

# Grid resolves the *compressed* pulse: at N ~ 8 the output is a few fs wide.
N_grid = 2^14
time_window = 4e-12          # s -> dt = 0.244 fs

peak_power_of_N(Nsol) = Nsol^2 * abs(beta2) / (gamma0 * T0^2)
length_for_order(Nsol) = z0 * (0.32 / Nsol + 1.1 / Nsol^2)
# (L/z0)·N² − 0.32·N − 1.1 = 0
function order_for_length(L)
    a = L / z0
    return (0.32 + sqrt(0.32^2 + 4 * a * 1.1)) / (2 * a)
end
# Nonlinear phase per step stays ~0.05 rad if the step count grows with N.
steps_for_order(Nsol) = max(800, round(Int, 400 * Nsol))

"""
Plain forward propagation of an N-th order soliton through length `L`. No AD:
grid, model and pulse are all rebuilt, which is only safe outside a
differentiated closure. Returns `(t, intensity)`.
"""
function simulate(Nsol, L; ngrid=N_grid, nsteps=steps_for_order(Nsol),
                  window=time_window)
    g = create_grid(ngrid, window, lambda0)
    med = Medium(L, gamma0, 0.0, [beta2], lambda0)
    pr = SimParams(; medium=med, z_saves=2, raman_model=nothing,
        self_steepening=false, solver=SSFM(L / nsteps), save_freq=false)
    mdl = Soliton.build_physics_model(g, pr, zeros(ComplexF64, ngrid))
    p = Pulse(zeros(ComplexF64, ngrid), zeros(ComplexF64, ngrid), g)
    P0 = peak_power_of_N(Nsol)
    @. p.At = complex(sqrt(P0) * sech(g.t / T0), 0.0)
    mul!(p.AW, mdl.to_freq, p.At)
    _, At, _ = Soliton.propagate(mdl, p, pr, pr.solver, false)
    return g.t, abs2.(At[:, end])
end

# --- The experiment: a fiber cut for soliton order ~4 ---
N_target = 4.0
L = length_for_order(N_target)
n_steps = steps_for_order(N_target)
N_lit = order_for_length(L)      # round-trips back to N_target
P0_lit = peak_power_of_N(N_lit)
Fc_lit = 4.1 * N_lit
grid = create_grid(N_grid, time_window, lambda0)

@printf("Fiber: L = %.4f m, beta2 = %.1f ps^2/km, gamma = %.3f /W/m\n",
    L, beta2 * 1e27, gamma0)
@printf("Input: sech, T0 = %.1f fs (FWHM = %.1f fs)\n", T0 * 1e15, FWHM_in * 1e15)
@printf("L_D = %.3f m, z0 = %.3f m, L/z0 = %.4f, grid dt = %.3f fs, %d steps\n",
    L_D, z0, L / z0, (time_window / N_grid) * 1e15, n_steps)
@printf("\nPublished optimum for this fiber: N = %.4f  ->  P0 = %.2f W\n", N_lit, P0_lit)
@printf("  predicted F_c = 4.1*N = %.2f  (output FWHM ~ %.1f fs)\n",
    Fc_lit, FWHM_in / Fc_lit * 1e15)

# --- Build model and pulse once, outside anything differentiated ---
medium = Medium(L, gamma0, 0.0, [beta2], lambda0)
params = SimParams(;
    medium=medium, z_saves=2, raman_model=nothing, self_steepening=false,
    solver=SSFM(L / n_steps), save_freq=false,
)
model = Soliton.build_physics_model(grid, params, zeros(ComplexF64, grid.N))
pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

"""
Effective duration τ_eff = (∫I dt)²/∫I² dt of the output, for an input soliton
of order `theta[1]`. The `grid.dt` factor turns the discrete sums into the
integrals — the same factor whose omission silently rescaled an earlier example.
"""
function output_duration(theta, model, pulse, params)
    P0 = theta[1]^2 * abs(beta2) / (gamma0 * T0^2)
    @. pulse.At = complex(sqrt(P0) * sech(grid.t / T0), 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
    I = abs2.(At[:, end])
    return grid.dt * sum(I)^2 / sum(abs2, I)
end

"""
Forward-mode dτ_eff/dN through the solver. `model`'s tangent is zero (the fiber
does not depend on N) but it must still be `Duplicated`, never `Const`, or its
scratch buffers get no shadow and the derivative silently comes back 0.0.
"""
function dtau_dN(Nsol)
    (deriv,) = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Forward),
        output_duration,
        Enzyme.Duplicated,
        Enzyme.Duplicated([Nsol], [1.0]),
        Enzyme.Duplicated(model, Enzyme.make_zero(model)),
        Enzyme.Duplicated(pulse, Enzyme.make_zero(pulse)),
        Enzyme.Const(params),
    )
    return deriv
end

tau_at(Nsol) = output_duration([Nsol], model, pulse, params)

# --- Sanity check the forward-mode gradient against finite differences ---
println("\nForward-mode gradient check (Enzyme vs. central differences):")
for Ncheck in (3.0, 4.0, 5.5)
    h = 1e-5
    g_ad = dtau_dN(Ncheck)
    g_fd = (tau_at(Ncheck + h) - tau_at(Ncheck - h)) / (2h)
    @printf("  N = %.2f: Enzyme = %+.6e  FD = %+.6e  rel.diff = %.3e\n",
        Ncheck, g_ad, g_fd, abs(g_ad - g_fd) / max(abs(g_fd), 1e-300))
end

# --- Scan and bisection ---
println("\nScanning soliton order (effective duration and its exact derivative)...")
N_scan = collect(range(2.4, 6.4; length=15))
tau_scan = similar(N_scan)
dtau_scan = similar(N_scan)
for (k, Nv) in enumerate(N_scan)
    tau_scan[k] = tau_at(Nv)
    dtau_scan[k] = dtau_dN(Nv)
end

"""
Minimum of τ_eff(N) by bisecting its derivative on `[lo, hi]`, which must
bracket a sign change. Kept in a function on purpose: assigning to `lo`/`hi`
inside a top-level `for` hits Julia's soft-scope rule and silently creates new
locals, which is how the first version of this script failed in CI.
"""
function bisect_derivative(deriv, lo, hi; iters=28)
    d_lo, d_hi = deriv(lo), deriv(hi)
    if d_lo >= 0 || d_hi <= 0
        @printf("\nNo sign change of dtau/dN on [%.2f, %.2f]", lo, hi)
        @printf(" (d_lo = %+.3e, d_hi = %+.3e).\n", d_lo, d_hi)
        return NaN, false
    end
    println("\nBisecting dtau/dN = 0 ...")
    for _ in 1:iters
        mid = 0.5 * (lo + hi)
        if deriv(mid) < 0
            lo = mid
        else
            hi = mid
        end
    end
    @printf("  converged: N = %.6f  (bracket width %.2e)\n", 0.5 * (lo + hi), hi - lo)
    return 0.5 * (lo + hi), true
end

N_ad, bracketed = bisect_derivative(dtau_dN, 2.5, 6.5)
if !bracketed
    println("The optimum is not bracketed — reporting the scan minimum instead.")
    N_ad = N_scan[argmin(tau_scan)]
end

# --- Result, measured by plain simulation ---
P0_ad = peak_power_of_N(N_ad)
t_ad, I_ad = simulate(N_ad, L; nsteps=n_steps)
_, I_lit = simulate(N_lit, L; nsteps=n_steps)
fwhm_ad = Soliton._fwhm(I_ad, t_ad)
fwhm_lit = Soliton._fwhm(I_lit, t_ad)

@printf("\n%-36s %12s %12s\n", "", "AD", "literature")
@printf("%-36s %12.4f %12.4f\n", "soliton order N", N_ad, N_lit)
@printf("%-36s %12.2f %12.2f\n", "input peak power P0 [W]", P0_ad, P0_lit)
@printf("%-36s %12.1f %12.1f\n", "output FWHM [fs]", fwhm_ad * 1e15, fwhm_lit * 1e15)
@printf("%-36s %12.2f %12.2f\n", "compression factor (FWHM ratio)",
    FWHM_in / fwhm_ad, FWHM_in / fwhm_lit)
@printf("\nDisagreement in N between AD and the published formula: %.2f %%\n",
    100 * abs(N_ad - N_lit) / N_lit)
@printf("tau_eff-based factor at the AD optimum: %.2f (proxy; pedestal-inflated)\n",
    3 * T0 / tau_at(N_ad))

# --- Diagnostic 1: is this script's own numerics converged? ---
# If refining steps or the time grid moves the answer, the mismatch below is
# ours and says nothing about the empirical relation.
println("\nConvergence of the simulation at the AD optimum:")
@printf("  %-30s %10s\n", "configuration", "FWHM [fs]")
for (label, kw) in (
    ("baseline", (nsteps=n_steps, ngrid=N_grid)),
    ("2x steps", (nsteps=2 * n_steps, ngrid=N_grid)),
    ("2x time resolution", (nsteps=n_steps, ngrid=2 * N_grid)),
    ("2x steps + 2x resolution", (nsteps=2 * n_steps, ngrid=2 * N_grid)),
)
    tt, II = simulate(N_ad, L; nsteps=kw.nsteps, ngrid=kw.ngrid)
    @printf("  %-30s %10.2f\n", label, Soliton._fwhm(II, tt) * 1e15)
end

# --- Diagnostic 2: does agreement improve with soliton order? ---
# Each order is simulated at *its own* optimal length, so this isolates the
# validity of the empirical relations from anything about our particular fiber.
println("\nMeasured vs. predicted compression factor, each at its own z_opt:")
@printf("  %6s %10s %12s %12s %10s\n", "N", "L [m]", "F_c measured", "4.1*N", "ratio")
N_trend = [1.5, 2.0, 3.0, 4.0, 5.0, 8.0]
ratio_trend = similar(N_trend)
for (k, Nv) in enumerate(N_trend)
    Lv = length_for_order(Nv)
    tt, II = simulate(Nv, Lv)
    fc = FWHM_in / Soliton._fwhm(II, tt)
    ratio_trend[k] = fc / (4.1 * Nv)
    @printf("  %6.1f %10.4f %12.2f %12.2f %10.3f\n", Nv, Lv, fc, 4.1 * Nv, ratio_trend[k])
end

# --- Plot 1: the objective and its exact derivative ---
plt1 = plot(
    N_scan, tau_scan .* 1e15;
    xlabel="soliton order N", ylabel="effective output duration [fs]",
    label="tau_eff", linewidth=2, legend=:topleft,
    title="Compressor optimum: objective and its Enzyme derivative",
)
vline!(plt1, [N_ad];
    label=@sprintf("AD optimum N = %.3f", N_ad), linestyle=:dash, linewidth=2)
vline!(plt1, [N_lit];
    label=@sprintf("published N = %.3f", N_lit), linestyle=:dot, linewidth=2)
plt1b = twinx(plt1)
plot!(
    plt1b, N_scan, dtau_scan .* 1e15;
    ylabel="d(tau_eff)/dN [fs]", label="derivative (forward-mode AD)",
    linewidth=2, color=:darkred, legend=:topright,
)
hline!(plt1b, [0.0]; label="", color=:gray, linestyle=:dash)
savefig(plt1, joinpath(output_dir, "compression_optimum_scan.png"))
println("Saved: ", joinpath(output_dir, "compression_optimum_scan.png"))

# --- Plot 2: the compressed pulse at the AD-found operating point ---
t_fs = t_ad .* 1e15
win = abs.(t_fs) .< 300
pulse_in = sech_pulse(grid, P0_ad, FWHM_in)
plt2 = plot(
    grid.t .* 1e15, abs2.(pulse_in.At);
    xlims=(-300, 300), xlabel="time [fs]", ylabel="power [W]",
    label=@sprintf("input, %.0f fs FWHM", FWHM_in * 1e15), linewidth=2,
    title="Soliton-effect compression at the AD-found optimum",
)
plot!(plt2, t_fs[win], I_ad[win];
    label=@sprintf("output, %.1f fs FWHM (F_c = %.1f)",
        fwhm_ad * 1e15, FWHM_in / fwhm_ad),
    linewidth=2)
savefig(plt2, joinpath(output_dir, "compression_optimum_pulse.png"))
println("Saved: ", joinpath(output_dir, "compression_optimum_pulse.png"))

# --- Plot 3: where the empirical relation becomes valid ---
plt3 = plot(
    N_trend, ratio_trend;
    xlabel="soliton order N", ylabel="measured F_c / (4.1 N)",
    label="ratio", linewidth=2, marker=:circle,
    title="Validity of F_c = 4.1N against the NLS",
)
hline!(plt3, [1.0]; label="perfect agreement", linestyle=:dash, color=:gray)
savefig(plt3, joinpath(output_dir, "compression_formula_validity.png"))
println("Saved: ", joinpath(output_dir, "compression_formula_validity.png"))
