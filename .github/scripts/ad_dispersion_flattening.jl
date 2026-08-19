"""
AD-driven dispersion-flattening design.

The Sellmeier inverse-fit example (`ad_dispersion_inverse_design.jl`) turned
out to be a *parameter-recovery* problem, and a fundamentally non-identifiable
one: many (B, C) combinations reproduce nearly the same measured curve, so
"the loss went down" didn't mean "the true glass was recovered." This example
is deliberately a different *kind* of problem — a **design** problem with no
hidden ground truth to fail to recover:

  Fused silica (bulk material) has a genuinely curved GVD near a 1000 nm
  pump — it's not exactly flat, and its local curvature changes across even
  a modest ±5% band. Find a small Taylor correction (β₂..β₅, representing
  what an engineered waveguide/microstructure contribution could add) that
  FLATTENS the *local* group-velocity dispersion of (material + correction)
  across that band, by minimizing the variance of the local curvature with
  exact `ForwardDiff` gradients through `Soliton.propagation_constant`.

Any correction that reduces the variance is a strictly better design — there
is no "wrong" positive-loss local minimum the way there was for the Sellmeier
fit, which is exactly why this is a better-posed showcase of the AD
groundwork in `docs/src/dev/adjoint_ad.md`.

Run with:
    julia --project=. -e 'import Pkg; Pkg.add(["ForwardDiff", "Plots"])'
    julia --project=. examples/ad_dispersion_flattening.jl

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
println("AD-driven dispersion-flattening design")
println("="^72)

# Fixed, given material dispersion (fused silica) — NOT optimized. Represents
# the intrinsic bulk-glass GVD a real fiber core would have; it is genuinely
# curved (not a pure quadratic) because a 3-term Sellmeier model, unlike a
# bare TaylorDispersion, has real higher-derivative structure.
material = FusedSilica()

lambda0 = 1000e-9
omega0 = 2π * c / lambda0

# Band to flatten: ±5% around the pump.
band_lambdas = collect(range(950e-9, 1050e-9; length=21))
V_band = [2π * c / lam - omega0 for lam in band_lambdas]

dV = 1e11  # finite-difference step for the local-curvature proxy [rad/s]

# β₂..β₅ live at wildly different natural SI scales (β₂ ~ 1e-27 s²/m, β₅ ~
# 1e-63 s⁵/m), which would make a single Adam learning rate meaningless in
# raw SI units. `scale[i]` converts a natural-magnitude "ps^n/km"-like
# O(1) value into the SI units TaylorDispersion expects, the same role
# `microns=true` plays for SellmeierDispersion.C in the other example.
scale = [10.0^(-(12 * n + 3)) for n in 2:5]

# Local curvature (∝ β₂(λ)) of the TOTAL dispersion (material + correction),
# via a central finite difference on propagation_constant — deliberately
# single-order AD (no nested Duals): ForwardDiff only differentiates the
# *outer* loss below with respect to `theta`, through this ordinary
# floating-point finite-difference calculation.
function local_curvatures(theta::AbstractVector{T}) where {T}
    betas = theta .* scale
    correction = TaylorDispersion(betas)
    curv = Vector{T}(undef, length(V_band))
    for (i, V) in enumerate(V_band)
        Vs = [V - dV, V, V + dV]
        Bm = propagation_constant(Vs, material, omega0) .+ propagation_constant(Vs, correction)
        curv[i] = (Bm[1] - 2Bm[2] + Bm[3]) / dV^2
    end
    return curv
end

# Pure flatness objective: minimize the variance of the local curvature
# across the band. No target value to hit and no hidden truth to recover —
# any theta that reduces this is a genuinely better design.
#
# `curv` itself is ~1e-27 in magnitude, so its *variance* is ~1e-54 — and
# d(loss)/d(theta), computed exactly by ForwardDiff, is correspondingly
# minuscule in absolute terms. Adam's `sqrt(v_hat) + eps_adam` denominator
# is then dominated by eps_adam (1e-8) instead of the actual gradient RMS,
# which strangles every step to ~lr*|g|/eps_adam ≈ 0 regardless of lr — this
# is exactly what happened on the first attempt (theta stayed at ~1e-56,
# betas_final ~1e-83, loss unchanged to reported precision). Normalizing
# curv by its own natural scale before computing the variance rescales the
# loss (and its gradient) to O(1), without changing the argmin — minimizing
# L or c·L for a positive constant c has the same optimum, so this is a
# conditioning fix only, not a change of objective.
const curv_ref = 1e-27  # ~ typical |β₂|; matches the observed pre-optimization std dev

function loss(theta::AbstractVector{T}) where {T}
    curv = local_curvatures(theta) ./ curv_ref
    m = sum(curv) / length(curv)
    return sum(x -> (x - m)^2, curv) / length(curv)
end

theta0 = zeros(4)  # start with no correction: bare material dispersion
curv_before = local_curvatures(theta0)
var_before = loss(theta0)  # dimensionless (units of curv_ref²)

theta = copy(theta0)
mvec = zeros(4)
vvec = zeros(4)
adam_b1, adam_b2, eps_adam, lr = 0.9, 0.999, 1e-8, 5e-2
n_iters = 1000
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

betas_final = theta .* scale
curv_after = local_curvatures(theta)
var_after = loss_history[end]  # dimensionless (units of curv_ref²)
std_before = sqrt(var_before) * curv_ref  # back to physical s²/m
std_after = sqrt(var_after) * curv_ref

@printf(
    "Local-curvature std dev: %.3e s²/m (before) -> %.3e s²/m (after), %.1fx flatter\n",
    std_before,
    std_after,
    std_before / std_after,
)
println("Optimized correction β₂..β₅ [s²/m, s³/m, s⁴/m, s⁵/m]: ", betas_final)

# Standard fiber-optics D-parameter [ps/(nm·km)] for a readable plot:
# D = -(2πc/λ²)·β₂, and 1 [s/m²] = 1e6 [ps/(nm·km)].
D_of(curv, lam) = -(2π * c / lam^2) * curv * 1e6

D_before = D_of.(curv_before, band_lambdas)
D_after = D_of.(curv_after, band_lambdas)

plt = plot(
    band_lambdas .* 1e9,
    D_before;
    label="before (bare material)",
    xlabel="Wavelength [nm]",
    ylabel="D [ps/(nm·km)]",
    title="AD-driven dispersion flattening",
    linewidth=2,
    marker=:circle,
)
plot!(
    plt,
    band_lambdas .* 1e9,
    D_after;
    label="after (AD-optimized correction)",
    linewidth=2,
    marker=:circle,
)
savefig(plt, joinpath(output_dir, "flattening_before_after.png"))
println("Saved: ", joinpath(output_dir, "flattening_before_after.png"))

plt_loss = plot(
    1:n_iters,
    loss_history;
    yscale=:log10,
    xlabel="Adam iteration",
    ylabel="loss (variance of curvature/curv_ref, dimensionless)",
    title="Flattening convergence",
    legend=false,
    linewidth=2,
)
savefig(plt_loss, joinpath(output_dir, "flattening_convergence.png"))
println("Saved: ", joinpath(output_dir, "flattening_convergence.png"))
