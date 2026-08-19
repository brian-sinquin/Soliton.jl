"""
AD-driven dispersion inverse design + a high-resolution supercontinuum run.

Two independent, self-contained parts:

  Part 1 — An *original* differentiation problem: recover a glass's Sellmeier
           dispersion coefficients (B, C) from a handful of "measured"
           propagation-constant samples, using exact `ForwardDiff` gradients
           through `Soliton.propagation_constant` — not finite differences,
           not a black-box optimizer. This exercises the AD groundwork laid
           in `docs/src/dev/adjoint_ad.md` (Grid/DispersionModel/PhysicsModel
           now generic over `<:Real`, so `SellmeierDispersion` can carry
           `ForwardDiff.Dual` coefficients). The zero-dispersion wavelength
           (ZDW) of both the true and recovered glass is then located via a
           *second* AD pass (a nested `ForwardDiff.derivative` computing
           β₂(ω) = d²β/dω² exactly, then a bisection root-find on that), as
           an independent sanity check against the well-known fused-silica
           ZDW (~1.27-1.30 μm, Malitson 1965).

  Part 2 — A full nonlinear GNLSE supercontinuum simulation at high temporal
           and spectral resolution (2^14 points, 12.5 ps window), reproducing
           Dudley, Genty & Coen, Rev. Mod. Phys. 78, 1135 (2006) Fig. 3 —
           the exact validated parameters already used in
           `docs/src/examples/ex1_supercontinuum.md`, so this half carries no
           new numerical risk. Adds a spectrogram, soliton-fission tracking,
           and the predicted dispersive-wave wavelength on top of the
           package's built-in 4-panel `plot(sol)` dashboard.

Neither the AD extension used by earlier commits (Enzyme, for reverse-mode
through full propagation) nor this script's ForwardDiff usage are declared as
package dependencies — following the pattern already established for the
Enzyme weak-dependency extension (`ext/SolitonEnzymeExt.jl` / CI's ad-hoc
`Pkg.add("Enzyme")`), install what this script needs in your own environment
first:

    julia --project=. -e 'import Pkg; Pkg.add(["ForwardDiff", "Plots"])'
    julia --project=. examples/ad_dispersion_inverse_design.jl

This script was written and reasoned through carefully, but — like the
type-parameterization work in the AD compatibility branch before a working
Julia toolchain was available to validate it — it has NOT been executed. Run
it and expect to tune iteration count / learning rate / the initial guess in
Part 1 if convergence isn't clean.
"""

using Soliton
using ForwardDiff
using Plots
using Printf

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

# ============================================================================
# Part 1 — AD-driven Sellmeier dispersion inverse design
# ============================================================================

println("="^72)
println("Part 1: AD-driven Sellmeier dispersion inverse design")
println("="^72)

# Ground truth: fused silica (Malitson 1965), already in Soliton's fiber
# catalog. `SellmeierDispersion.C` is stored in m² (Soliton's natural SI
# units); converting back to μm² here just keeps the fit's numbers in a
# human-scale range.
true_model = FusedSilica()
B_true = true_model.B
C_true_um2 = true_model.C .* 1e12

# Reference wavelength for the propagation-constant frame, and 8 "measured"
# wavelengths spanning most of fused silica's valid Sellmeier range
# (0.21-3.71 μm) — a sparse, unevenly-spaced set, as a real vendor
# measurement sheet would give you.
lambda0 = 800e-9
omega0 = 2π * c / lambda0
sample_lambdas = [550e-9, 650e-9, 750e-9, 900e-9, 1050e-9, 1200e-9, 1400e-9, 1600e-9]
V_samples = [2π * c / lam - omega0 for lam in sample_lambdas]

B_measured = propagation_constant(V_samples, true_model, omega0)

# Deliberately wrong initial guess — same order of magnitude as the truth
# (a plausible "looked up the wrong glass" starting point), not derived from
# it.
B0 = [0.55, 0.50, 0.70]
C0_um2 = [0.05, 0.08, 80.0]

function loss(p::AbstractVector{T}) where {T}
    B = p[1:3]
    C_um2 = p[4:6]
    model = SellmeierDispersion(B, C_um2; microns=true)
    B_model = propagation_constant(V_samples, model, omega0)
    return sum(abs2, B_model .- B_measured)
end

p = vcat(B0, C0_um2)

# Hand-rolled Adam — avoids pulling in an optimization package for a 6
# parameter problem, and its per-parameter adaptive step size handles the
# very different natural scales of B (~O(1)) and C (~O(1e-3) to O(1e2) μm²)
# reasonably well without manual rescaling.
mvec = zeros(6)
vvec = zeros(6)
beta1, beta2, eps_adam, lr = 0.9, 0.999, 1e-8, 0.05
n_iters = 800
loss_history = zeros(n_iters)

for k in 1:n_iters
    g = ForwardDiff.gradient(loss, p)
    mvec .= beta1 .* mvec .+ (1 - beta1) .* g
    vvec .= beta2 .* vvec .+ (1 - beta2) .* g .^ 2
    m_hat = mvec ./ (1 - beta1^k)
    v_hat = vvec ./ (1 - beta2^k)
    p .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
    loss_history[k] = loss(p)
end

B_fit = p[1:3]
C_fit_um2 = p[4:6]

@printf("Final loss: %.3e (started at %.3e)\n", loss_history[end], loss_history[1])
println("Recovered B:       ", round.(B_fit; digits=4), "   (true: ", round.(B_true; digits=4), ")")
println(
    "Recovered C [μm²]: ",
    round.(C_fit_um2; digits=5),
    "   (true: ",
    round.(C_true_um2; digits=5),
    ")",
)

# --- Independent sanity check: locate the zero-dispersion wavelength (ZDW) ---
# via a *second*, nested AD pass rather than reusing the fit above. β₂(ω) =
# d²β/dω² is computed exactly with a forward-over-forward ForwardDiff call;
# its root (bisection) is the ZDW. This exercises AD at a different order
# than Part 1's gradient descent, and gives an external correctness check:
# the well-known fused-silica ZDW is ≈1.27-1.30 μm (Malitson 1965).
function beta_omega(model::SellmeierDispersion, omega::Real)
    lambda = 2π * c / omega
    n2 = one(eltype(model.B))
    for i in eachindex(model.B)
        n2 += model.B[i] * lambda^2 / (lambda^2 - model.C[i])
    end
    return sqrt(n2) * omega / c
end

beta2_omega(model::SellmeierDispersion, omega::Real) =
    ForwardDiff.derivative(w -> ForwardDiff.derivative(w2 -> beta_omega(model, w2), w), omega)

function find_zdw(model::SellmeierDispersion; lo_lambda=1.0e-6, hi_lambda=1.6e-6, iters=60)
    lo, hi = 2π * c / hi_lambda, 2π * c / lo_lambda  # omega bounds (lo_lambda -> higher omega)
    f_lo, f_hi = beta2_omega(model, lo), beta2_omega(model, hi)
    sign(f_lo) != sign(f_hi) || throw(
        ErrorException(
            "beta2_omega does not change sign between $(lo_lambda*1e9) nm and " *
            "$(hi_lambda*1e9) nm — widen the bracket to bound the true ZDW.",
        ),
    )
    for _ in 1:iters
        mid = (lo + hi) / 2
        f_mid = beta2_omega(model, mid)
        if sign(f_mid) == sign(f_lo)
            lo, f_lo = mid, f_mid
        else
            hi = mid
        end
    end
    return 2π * c / ((lo + hi) / 2)
end

fitted_model = SellmeierDispersion(B_fit, C_fit_um2; microns=true)
zdw_true = find_zdw(true_model)
zdw_fit = find_zdw(fitted_model)
@printf("ZDW of the TRUE model:      %.1f nm\n", zdw_true * 1e9)
@printf("ZDW of the RECOVERED model: %.1f nm\n", zdw_fit * 1e9)
println("(Expect both near 1270-1300 nm — the textbook fused-silica ZDW.)")

plt_loss = plot(
    1:n_iters,
    loss_history;
    yscale=:log10,
    xlabel="Adam iteration",
    ylabel="loss (Σ squared β residuals) [1/m²]",
    title="AD-driven Sellmeier fit convergence",
    legend=false,
    linewidth=2,
)
savefig(plt_loss, joinpath(output_dir, "part1_fit_convergence.png"))
println("Saved: ", joinpath(output_dir, "part1_fit_convergence.png"))

# ============================================================================
# Part 2 — High-resolution nonlinear GNLSE supercontinuum simulation
# ============================================================================

println()
println("="^72)
println("Part 2: High-resolution supercontinuum simulation")
println("(Dudley, Genty & Coen, Rev. Mod. Phys. 78, 1135 (2006), Fig. 3)")
println("="^72)

# Exact validated parameters from docs/src/examples/ex1_supercontinuum.md —
# unchanged, so this half carries no new numerical risk.
medium = commercial_fiber("NKT_NL_PM_750", length=0.15)  # 15 cm PCF

grid = create_grid(2^14, 12.5e-12, medium.lambda0)  # 16384 points, 12.5 ps window
pulse_in = sech_pulse(grid, 10000.0, 50e-15)         # 10 kW peak, 50 fs FWHM sech

params = SimParams(;
    medium=medium, z_saves=200, raman_model=BlowWood(), self_steepening=true
)

println("Grid: N=", grid.N, ", window=", round((grid.t[end] - grid.t[1]) * 1e12; digits=2),
    " ps, dt=", round(grid.dt * 1e15; digits=2), " fs, λ₀=", round(grid.lambda0 * 1e9; digits=1), " nm")

sol = solve(pulse_in, params; progress=true)
pulse_out = Pulse(sol)

# --- Headline numbers ---
z_fiss, peak_power_z, centroid_w_z = track_solitons(sol)
lambda_dw = dispersive_wave_wavelength(medium, pulse_in)
bw_in = spectral_bandwidth(pulse_in)
bw_out = spectral_bandwidth(pulse_out)

@printf("Soliton fission distance:        %.2f mm\n", z_fiss * 1e3)
@printf("Predicted dispersive-wave λ:     %.1f nm\n", lambda_dw * 1e9)
@printf("Input pulse energy:              %.2f nJ\n", pulse_energy(pulse_in) * 1e9)
@printf("Output pulse energy:             %.2f nJ\n", pulse_energy(pulse_out) * 1e9)
@printf("Photon number conservation:      %.3e -> %.3e\n", photon_number(sol)[1], photon_number(sol)[end])

# --- Visualization ---

# 1. Built-in 4-panel dashboard: temporal & spectral heatmaps + slices.
plt_dashboard = plot(sol)
savefig(plt_dashboard, joinpath(output_dir, "part2_dashboard.png"))
println("Saved: ", joinpath(output_dir, "part2_dashboard.png"))

# 2. Output spectrogram (STFT) — a direct view of the time-frequency
#    structure (soliton fission fan-out, dispersive wave) at the fiber exit.
t_delays, V_grid, S_matrix = spectrogram(pulse_out; n_delay=300)
lambda_grid_nm = (2π * c ./ (V_grid .+ grid.omega0)) .* 1e9
# heatmap needs ascending y — wavelength is inversely related to V_grid's
# (ascending) angular frequency, so it comes out descending; reorder both.
wl_order = sortperm(lambda_grid_nm)
lambda_grid_nm = lambda_grid_nm[wl_order]
S_matrix = S_matrix[wl_order, :]
# `ylims` alone only clips the *displayed* range — GR still has to rasterize
# all N=16384 rows (most of which, at extreme angular frequencies, map to
# physically meaningless near-zero or near-infinite wavelengths), which
# crashed the headless GR backend with "GKS: can't allocate memory" on the
# CI runner. Subset the data itself to the physically relevant window first.
wl_mask = 400.0 .<= lambda_grid_nm .<= 1600.0
lambda_grid_nm = lambda_grid_nm[wl_mask]
S_matrix = S_matrix[wl_mask, :]
plt_spectrogram = heatmap(
    t_delays .* 1e15,
    lambda_grid_nm,
    log10.(S_matrix .+ 1e-6 * maximum(S_matrix));
    xlabel="Time delay [fs]",
    ylabel="Wavelength [nm]",
    title="Output spectrogram (log scale)",
    color=:turbo,
    colorbar_title="log₁₀ intensity [a.u.]",
)
savefig(plt_spectrogram, joinpath(output_dir, "part2_spectrogram.png"))
println("Saved: ", joinpath(output_dir, "part2_spectrogram.png"))

# 3. Soliton-fission tracking: peak power and spectral centroid vs distance.
plt_track = plot(
    sol.Z .* 1e3,
    peak_power_z;
    xlabel="Distance [mm]",
    ylabel="Peak power [W]",
    label="peak power",
    title="Soliton fission tracking",
    linewidth=2,
)
vline!(plt_track, [z_fiss * 1e3]; label="fission point", linestyle=:dash)
savefig(plt_track, joinpath(output_dir, "part2_soliton_tracking.png"))
println("Saved: ", joinpath(output_dir, "part2_soliton_tracking.png"))

println()
println("Done. See ", output_dir, " for the 4 saved plots.")
