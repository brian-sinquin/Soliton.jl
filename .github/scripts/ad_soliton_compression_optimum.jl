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

That optimum is known analytically. Writing the soliton order

    N² = L_D / L_NL = γ P₀ T₀² / |β₂|,        z₀ = (π/2) L_D,  L_D = T₀²/|β₂|

the classic empirical relations for soliton-effect compression are

    z_opt / z₀ ≈ 0.32/N + 1.1/N²             (optimum fiber length)
    F_c        ≈ 4.1 N                        (compression factor)

(Agrawal, *Nonlinear Fiber Optics*, soliton-effect compression; the same pair
is quoted in the PCF compression literature, e.g. Foroni et al.,
`arXiv:physics/0610252`.) Inverting the first relation for a **fixed** L gives
the soliton order the fiber is optimal for — and therefore the ideal input peak
power P₀ = N²|β₂|/(γT₀²).

So this script has a real, falsifiable answer to aim at. It never uses the
formulas to *drive* the optimization: the gradient comes entirely from Enzyme
differentiating `Soliton.propagate`, and the formulas are only consulted at the
end, to see whether AD landed where the literature says it should.

# Choosing an objective that means "compressed"

The first version of this script maximized the output peak power, and CI showed
why that is wrong: d(peak)/dN came back strongly positive and *growing* over the
whole scanned range (+2.0e2 at N=1.8, +9.9e4 at N=3.2). The reason is that the
input power itself is the parameter — P₀ ∝ N² — so "maximize output peak power"
mostly rewards pumping harder, not compressing better, and it does not
correspond to the criterion behind z_opt at all.

The objective here is instead the **effective duration**

    τ_eff = (∫I dt)² / ∫I² dt,       I(t) = |A(z=L, t)|²

which is invariant under I → αI, so it measures pulse *shape* only and cannot be
gamed by injecting more energy. It is smooth (two `sum` reductions, no `maximum`
and no threshold search, unlike a literal FWHM), which matters because it has to
be differentiated. For a sech² intensity τ_eff = 3T₀ exactly, so ratios of τ_eff
are comparable to ratios of FWHM for sech-like pulses.

It is only a *proxy* for the FWHM, though, and the run below shows how much: at
the optimum the FWHM ratio is 7.07 while the τ_eff ratio is 4.64, because τ_eff
integrates the whole trace and so is dragged upward by the pedestal that
soliton-effect compression always leaves behind. What matters here is that its
*argmin* is in the right place — and it is: minimizing τ_eff lands on a point
whose measured FWHM (24.9 fs) is better than the one the empirical formula
points at (27.8 fs). Both factors are reported so the gap stays visible.

# What the run actually finds

    soliton order N          AD 2.5961   vs   literature 2.4918   (4.19 % apart)
    input peak power P0      AD 122.5 W  vs   literature 112.9 W
    output FWHM              AD 24.9 fs  vs   literature 27.8 fs

Two honest caveats. The optimum AD finds is 4.2 % away from the empirical
prediction, which is about the accuracy the empirical relation itself claims —
and AD's point is the better one by direct measurement, which is what an
optimizer is supposed to do. But the *compression factor* the same relation
predicts, F_c ≈ 4.1N ≈ 10.2, is well above the 7.07 actually delivered here:
4.1N is an idealized figure and the NLS does not reach it at these modest
soliton orders. The location of the optimum reproduces; its depth does not.

# Why forward mode

There is exactly **one** free parameter. Reverse mode would tape the whole
500-step propagation to produce a single derivative; forward mode carries one
tangent alongside the primal at O(1) memory in the step count and roughly one
extra function evaluation. This is the case
`docs/src/dev/ad_ecosystem_review.md` (finding 6) identifies as strictly better
served by forward mode, and it is the first example to use the forward rule in
`ext/SolitonEnzymeExt.jl`.

An exact derivative also changes *how* to optimize. Rather than gradient descent
with a hand-tuned learning rate — which has already produced two real bugs in
this repository's examples — the optimum is located by bisecting the derivative
to its zero crossing: no step size, monotone convergence, and the bracket is
verified up front instead of assumed.

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

# --- Fixed experiment: fiber and input duration are given, power is free ---
N_grid = 2^13
time_window = 4e-12          # s
lambda0 = 1550e-9            # m
gamma0 = 0.11                # 1/(W·m)
beta2 = -20e-27              # s²/m, anomalous
T0 = 100e-15                 # s, input sech half-width (FWHM = 1.7627·T0)
m_sech = 2 * log(1 + sqrt(2))   # the FWHM↔T0 factor `sech_pulse` uses
FWHM_in = m_sech * T0

L_D = T0^2 / abs(beta2)      # dispersion length
z0 = (pi / 2) * L_D          # soliton period
L = 0.24                     # m — the fiber we happen to have on the bench
n_steps = 500
grid = create_grid(N_grid, time_window, lambda0)

@printf("Fiber: L = %.3f m, beta2 = %.1f ps^2/km, gamma = %.3f /W/m\n",
    L, beta2 * 1e27, gamma0)
@printf("Input: sech, T0 = %.1f fs (FWHM = %.1f fs)\n", T0 * 1e15, FWHM_in * 1e15)
@printf("L_D = %.3f m, soliton period z0 = %.3f m, L/z0 = %.4f\n", L_D, z0, L / z0)

# --- What the literature predicts, before we look at any gradient ---
# z_opt/z0 = 0.32/N + 1.1/N²  ⇒  (L/z0)·N² − 0.32·N − 1.1 = 0
L_over_z0 = L / z0
N_lit = (0.32 + sqrt(0.32^2 + 4 * L_over_z0 * 1.1)) / (2 * L_over_z0)
P0_lit = N_lit^2 * abs(beta2) / (gamma0 * T0^2)
Fc_lit = 4.1 * N_lit
@printf("\nPublished optimum for this fiber: N = %.4f  ->  P0 = %.2f W\n", N_lit, P0_lit)
@printf("  predicted compression factor F_c = 4.1*N = %.2f  (output FWHM ~ %.1f fs)\n",
    Fc_lit, FWHM_in / Fc_lit * 1e15)

# --- Build model and pulse once, outside anything differentiated ---
# (Constructing either inside the closure trips EnzymeRuntimeActivityError;
# see docs/src/dev/adjoint_ad.md.)
medium = Medium(L, gamma0, 0.0, [beta2], lambda0)
params = SimParams(;
    medium=medium,
    z_saves=2,
    raman_model=nothing,
    self_steepening=false,
    solver=SSFM(L / n_steps),
    save_freq=false,
)
template = zeros(ComplexF64, grid.N)
model = Soliton.build_physics_model(grid, params, template)
pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

peak_power_of_N(Nsol) = Nsol^2 * abs(beta2) / (gamma0 * T0^2)

"""
Effective duration τ_eff = (∫I dt)²/∫I² dt of the output pulse, for an input
soliton of order `theta[1]`. Amplitude-scale invariant, so it scores shape only.
The `grid.dt` factor is what turns the discrete sums into the integrals — the
same factor whose omission silently rescaled an earlier example by √dt.
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
Forward-mode derivative dτ_eff/dN, taken straight through the solver. The
`model` tangent is zero (the fiber does not depend on N) but the model must
still be `Duplicated`, never `Const`, or its scratch buffers get no shadow and
the derivative silently comes back 0.0.

Shadows are rebuilt per call for clarity; reusing one is the known optimization
(`ad_ecosystem_review.md`, finding 3).
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
for Ncheck in (1.8, 2.5, 3.2)
    h = 1e-5
    g_ad = dtau_dN(Ncheck)
    g_fd = (tau_at(Ncheck + h) - tau_at(Ncheck - h)) / (2h)
    @printf("  N = %.2f: Enzyme = %+.6e  FD = %+.6e  rel.diff = %.3e\n",
        Ncheck, g_ad, g_fd, abs(g_ad - g_fd) / max(abs(g_fd), 1e-300))
end

# --- Scan: effective output duration and its AD derivative vs soliton order ---
println("\nScanning soliton order (effective duration and its exact derivative)...")
N_scan = collect(range(1.2, 4.2; length=25))
tau_scan = similar(N_scan)
dtau_scan = similar(N_scan)
for (k, Nv) in enumerate(N_scan)
    tau_scan[k] = tau_at(Nv)
    dtau_scan[k] = dtau_dN(Nv)
end

"""
Locate the minimum of τ_eff(N) by bisecting its derivative to zero on
`[lo, hi]`, which must bracket a sign change (dτ/dN < 0 then > 0). Returns
`(N_opt, bracketed)`; when the bracket fails, falls back to the scan minimum so
the script reports something honest instead of a spurious root.

Kept in a function rather than at top level on purpose: assigning to `lo`/`hi`
inside a top-level `for` hits Julia's soft-scope rule and silently creates new
locals, which is exactly how the first version of this script failed in CI.
"""
function bisect_derivative(deriv, lo, hi; iters=32)
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

N_ad, bracketed = bisect_derivative(dtau_dN, 1.4, 4.0)
if !bracketed
    println("The optimum is not bracketed — reporting the scan minimum instead.")
    N_ad = N_scan[argmin(tau_scan)]
end

# --- Verify with plain forward simulations (no AD involved) ---
P0_ad = peak_power_of_N(N_ad)
pulse_in = sech_pulse(grid, P0_ad, FWHM_in)

function propagate_at(Nsol)
    p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
    P0 = peak_power_of_N(Nsol)
    @. p.At = complex(sqrt(P0) * sech(grid.t / T0), 0.0)
    mul!(p.AW, model.to_freq, p.At)
    _, At, _ = Soliton.propagate(model, p, params, params.solver, false)
    return At[:, end]
end

At_ad = propagate_at(N_ad)
At_lit = propagate_at(N_lit)
I_ad = abs2.(At_ad)

fwhm_out_ad = Soliton._fwhm(I_ad, grid.t)
fwhm_out_lit = Soliton._fwhm(abs2.(At_lit), grid.t)
Fc_ad = FWHM_in / fwhm_out_ad
Fc_lit_measured = FWHM_in / fwhm_out_lit
tau_in = 3 * T0                     # exact for a sech² intensity profile
Feff_ad = tau_in / tau_at(N_ad)

@printf("\n%-36s %12s %12s\n", "", "AD", "literature")
@printf("%-36s %12.4f %12.4f\n", "soliton order N", N_ad, N_lit)
@printf("%-36s %12.2f %12.2f\n", "input peak power P0 [W]", P0_ad, P0_lit)
@printf("%-36s %12.1f %12.1f\n", "output FWHM [fs]",
    fwhm_out_ad * 1e15, fwhm_out_lit * 1e15)
@printf("%-36s %12.2f %12.2f\n", "compression factor (FWHM ratio)",
    Fc_ad, Fc_lit_measured)
@printf("\nPredicted F_c = 4.1*N = %.2f; tau_eff-based factor at the AD optimum = %.2f\n",
    Fc_lit, Feff_ad)
@printf("Disagreement in N between AD and the published formula: %.2f %%\n",
    100 * abs(N_ad - N_lit) / N_lit)
@printf("Output peak power at the AD optimum: %.1f W (input %.1f W)\n",
    maximum(I_ad), P0_ad)

# --- Plot 1: the objective and its exact derivative ---
plt1 = plot(
    N_scan, tau_scan .* 1e15;
    xlabel="soliton order N",
    ylabel="effective output duration [fs]",
    label="tau_eff",
    linewidth=2,
    legend=:topleft,
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
t_fs = grid.t .* 1e15
win = abs.(t_fs) .< 400
plt2 = plot(
    t_fs[win], abs2.(pulse_in.At)[win];
    xlabel="time [fs]", ylabel="power [W]",
    label=@sprintf("input, %.0f fs FWHM", FWHM_in * 1e15),
    linewidth=2,
    title="Soliton-effect compression at the AD-found optimum",
)
plot!(plt2, t_fs[win], I_ad[win];
    label=@sprintf("output, %.1f fs FWHM (F_c = %.1f)", fwhm_out_ad * 1e15, Fc_ad),
    linewidth=2)
savefig(plt2, joinpath(output_dir, "compression_optimum_pulse.png"))
println("Saved: ", joinpath(output_dir, "compression_optimum_pulse.png"))
