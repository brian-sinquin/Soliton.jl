"""
AD-driven dispersive-wave (Cherenkov radiation) wavelength targeting,
verified against full nonlinear propagation.

The two earlier examples stayed entirely inside the differentiable
propagation_constant/dispersion_operator layer. This one closes the loop:

  1. Design: starting from the real NKT_NL_PM_750 PCF (the exact fiber
     reproducing Dudley et al. 2006 Fig. 3 elsewhere in this repo), use
     `ForwardDiff.gradient` to adjust β₂ and β₃ — the two lowest-order
     Taylor dispersion coefficients — so the soliton-fission dispersive-wave
     phase-matching condition

         Δβ(Δω) = β(ω₀+Δω) − β(ω₀) − β₁·Δω − ½γP₀ = 0

     (the same condition `dispersive_wave_wavelength` already solves
     numerically) is satisfied at a *chosen target wavelength* instead of
     wherever the real fiber happens to put it (~652 nm). A small ridge
     regularization toward the original β₂/β₃ makes the otherwise
     underdetermined problem (2 free parameters, 1 target) well-posed: find
     the *smallest* redesign that hits the target, not just *a* redesign.
  2. Verify: build a `Medium` from the AD-optimized coefficients (keeping
     β₄..β₁₀ at the real fiber's values) and run the actual nonlinear
     GNLSE simulation (2^14 points, same pulse as the Dudley reproduction)
     to check that a real dispersive-wave feature actually shows up near the
     target in the emergent output spectrum — not just in the linearized
     phase-matching approximation used for the design step.

Run with:
    julia --project=. -e 'import Pkg; Pkg.add(["ForwardDiff", "Plots"])'
    julia --project=. examples/ad_dispersive_wave_design.jl

Written and reasoned through carefully but not executed locally (no Julia
toolchain in the environment that wrote it) — validated by the CI workflow
`.github/workflows/example-ad-dispersion.yml` instead.
"""

using Soliton
using ForwardDiff
using Plots
using Printf

output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^72)
println("AD-driven dispersive-wave wavelength targeting + verification")
println("="^72)

# The real, validated fiber (docs/src/examples/ex1_supercontinuum.md).
base_medium = commercial_fiber("NKT_NL_PM_750", length=0.15)
base_betas_full = base_medium.dispersion.betas  # β₂..β₁₀, Dudley et al. Table 1
gamma_val = base_medium.gamma
P0 = 10000.0  # same peak power as the Dudley reproduction's input pulse
omega0 = 2π * c / base_medium.lambda0

# Target: pull the dispersive wave from its natural ~652 nm down to 500 nm.
lambda_DW_target = 500e-9
V_target = 2π * c / lambda_DW_target - omega0

# β₂, β₃ in "ps^n/km"-like O(1) units (scale2[i] converts to SI), same
# reasoning as the flattening example's scale vector.
scale2 = [1e-27, 1e-39]
scale_pm = 0.5 * gamma_val * P0  # natural magnitude of the phase-mismatch residual

theta0 = [base_betas_full[1], base_betas_full[2]] ./ scale2

function phase_mismatch(full_betas::AbstractVector{T}, V::Real, gamma::Real, P0::Real) where {T}
    return propagation_constant([V], TaylorDispersion(full_betas))[1] - 0.5 * gamma * P0
end

# Loss = (normalized phase mismatch at the target)² + a small ridge pulling
# back toward the real fiber's β₂/β₃. Without the ridge term this is
# underdetermined (a 1-D family of (β₂,β₃) all zero the single target
# equation); with it, gradient descent converges to the minimal-perturbation
# redesign instead of drifting arbitrarily far along that family.
function loss(theta::AbstractVector{T}) where {T}
    betas2 = theta .* scale2
    full_betas = vcat(betas2, base_betas_full[3:end])
    pm = phase_mismatch(full_betas, V_target, gamma_val, P0) / scale_pm
    reg = 1e-4 * sum((theta .- theta0) .^ 2)
    return pm^2 + reg
end

pm_before = phase_mismatch(base_betas_full, V_target, gamma_val, P0)
@printf(
    "Phase mismatch at target (%.0f nm) before optimization: %.3e 1/m\n",
    lambda_DW_target * 1e9,
    pm_before,
)

theta = copy(theta0)
mvec = zeros(2)
vvec = zeros(2)
adam_b1, adam_b2, eps_adam, lr = 0.9, 0.999, 1e-8, 3e-2
n_iters = 800
loss_history = zeros(n_iters)

for k in 1:n_iters
    g = ForwardDiff.gradient(loss, theta)
    mvec .= adam_b1 .* mvec .+ (1 - adam_b1) .* g
    vvec .= adam_b2 .* vvec .+ (1 - adam_b2) .* g .^ 2
    m_hat = mvec ./ (1 - adam_b1^k)
    v_hat = vvec ./ (1 - adam_b2^k)
    theta .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps_adam)
    loss_history[k] = loss(theta)
end

betas2_opt = theta .* scale2
full_betas_opt = vcat(betas2_opt, base_betas_full[3:end])
pm_after = phase_mismatch(full_betas_opt, V_target, gamma_val, P0)

@printf("Phase mismatch at target after optimization:  %.3e 1/m\n", pm_after)
@printf("β₂: %.4f -> %.4f ps²/km  (Δ = %.1f%%)\n", theta0[1], theta[1], 100 * (theta[1] - theta0[1]) / theta0[1])
@printf("β₃: %.4f -> %.4f ps³/km  (Δ = %.1f%%)\n", theta0[2], theta[2], 100 * (theta[2] - theta0[2]) / theta0[2])

# --- Plot 1: phase-mismatch curve before/after, target marked ---
lambda_scan = collect(range(450e-9, 900e-9; length=200))
V_scan = [2π * c / lam - omega0 for lam in lambda_scan]
pm_curve_before = [phase_mismatch(base_betas_full, V, gamma_val, P0) for V in V_scan]
pm_curve_after = [phase_mismatch(full_betas_opt, V, gamma_val, P0) for V in V_scan]

plt1 = plot(
    lambda_scan .* 1e9,
    pm_curve_before ./ 1e3;
    label="before (real fiber)",
    xlabel="Wavelength [nm]",
    ylabel="Phase mismatch Δβ - γP₀/2 [1/mm]",
    title="Dispersive-wave phase matching",
    linewidth=2,
)
plot!(plt1, lambda_scan .* 1e9, pm_curve_after ./ 1e3; label="after (AD-redesigned)", linewidth=2)
hline!(plt1, [0.0]; label=nothing, linestyle=:dot, color=:black)
vline!(plt1, [lambda_DW_target * 1e9]; label="target", linestyle=:dash, color=:green)
savefig(plt1, joinpath(output_dir, "dw_phase_matching.png"))
println("Saved: ", joinpath(output_dir, "dw_phase_matching.png"))

# --- Verification: run the real nonlinear simulation on the redesigned fiber ---
println()
println("Verifying via full nonlinear propagation...")
verify_medium = Medium(
    base_medium.length, gamma_val, base_medium.loss, TaylorDispersion(full_betas_opt), base_medium.lambda0
)
grid = create_grid(2^14, 12.5e-12, verify_medium.lambda0)
pulse_in = sech_pulse(grid, P0, 50e-15)
params = SimParams(; medium=verify_medium, z_saves=200, raman_model=BlowWood(), self_steepening=true)
sol = solve(pulse_in, params; progress=false)
pulse_out = Pulse(sol)

lambda_dw_predicted = dispersive_wave_wavelength(verify_medium, pulse_in)
@printf("dispersive_wave_wavelength() prediction for redesigned fiber: %.1f nm\n", lambda_dw_predicted * 1e9)
@printf("(AD design target was %.1f nm)\n", lambda_DW_target * 1e9)

lambda_grid_nm = wavelength_grid(grid) .* 1e9
spec_out = abs2.(pulse_out.AW)
spec_out_db = 10 .* log10.(spec_out ./ maximum(spec_out) .+ 1e-8)
order = sortperm(lambda_grid_nm)

plt2 = plot(
    lambda_grid_nm[order],
    spec_out_db[order];
    xlabel="Wavelength [nm]",
    ylabel="Spectral power [dB]",
    title="Verification: output spectrum of the AD-redesigned fiber",
    ylims=(-60, 5),
    xlims=(400, 1600),
    linewidth=1.5,
    label="output spectrum",
)
vline!(plt2, [lambda_DW_target * 1e9]; label="AD target", linestyle=:dash, color=:green)
vline!(plt2, [lambda_dw_predicted * 1e9]; label="predicted (existing analysis fn)", linestyle=:dot, color=:orange)
savefig(plt2, joinpath(output_dir, "dw_verification_spectrum.png"))
println("Saved: ", joinpath(output_dir, "dw_verification_spectrum.png"))
