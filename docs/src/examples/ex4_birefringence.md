```@meta
CurrentModule = JuGNLSE
```

# Example 4: Soliton Trapping in a Birefringent Fiber

**Reproducing the polarization-locking phenomenon of Menyuk (1988)**

DOI: [10.1364/JOSAB.5.000392](https://doi.org/10.1364/JOSAB.5.000392)

---

## Physical Background

In a linearly birefringent fiber, the two polarization axes (x and y) have different group
velocities. A pulse launched at 45° will normally split into two temporally separated
sub-pulses after a "walk-off length" ``L_W = T_0 / |\delta|``, where ``\delta = \beta_{1x} - \beta_{1y}``
is the group-velocity mismatch (walk-off).

Menyuk (1988) showed that above a critical power, **cross-phase modulation (XPM)** can
overcome the walk-off: the faster and slower solitons trap each other through the XPM
potential and travel together as a **bound polarization pair**, with the same group velocity.

The trapping condition is approximately:
```math
\frac{L_D}{L_W} < \frac{\sqrt{3}}{2}
\quad \Longleftrightarrow \quad
|\delta| < \frac{\sqrt{3}}{2} \frac{T_0}{L_D}
```

## Simulation Setup

We reproduce the trapping effect using `BirefringentMedium` with equal energy on both axes
and a group-velocity mismatch that is just inside/outside the trapping threshold.

```julia
using JuGNLSE

# ─── Fiber parameters ────────────────────────────────────────────────────────
lambda0 = 1550e-9          # [m]
beta2   = -21.5e-27        # [s²/m] anomalous, same on both axes
gamma   = 0.0011           # [1/(W·m)]

# ─── Soliton parameters ─────────────────────────────────────────────────────
T0   = 1e-12               # 1 ps soliton half-width
P0   = abs(beta2) / (gamma * T0^2)   # fundamental soliton power [W]
LD   = T0^2 / abs(beta2)             # dispersion length [m]
FWHM = 2 * log(1 + sqrt(2)) * T0    # FWHM [s]

println("Fundamental soliton: P₀ = $(round(P0; sigdigits=3)) W,  L_D = $(round(LD; sigdigits=3)) m")

# ─── Grid ────────────────────────────────────────────────────────────────────
grid = create_grid(2^12, 80e-12, lambda0)

# ─── Walk-off: Δβ₁ = 1 ps/m → walk-off length L_W = T₀/|δ| = 1 ps / (1 ps/m) = 1 m ──
delta_beta1 = 1e-12          # [s/m] group-velocity mismatch

# ─── Case A: Low birefringence — solitons trap (below threshold) ─────────────
# trapping criterion: |δ| < √3/2 * T₀/L_D = √3/2 / L_D (for T₀=1ps)
trap_threshold = sqrt(3)/2 * T0 / LD   # [s/m]
println("Trapping threshold |δ| < $(round(trap_threshold*1e12; sigdigits=2)) ps/m")

# ─── Both cases: axes share dispersion, differ only in β₁ ────────────────────
disp_x = TaylorDispersion([beta2],  0.0)              # reference axis
disp_y = TaylorDispersion([beta2], +delta_beta1)      # faster axis (walk-off)

med_A = BirefringentMedium(
    5.0,           # 5 m fiber
    gamma, 0.0,
    disp_x, disp_y,
    0.0,           # Δβ₀ = 0 (no phase mismatch)
    lambda0,
)

# ─── Initial pulse: 45° → equal amplitude on both axes ──────────────────────
Ax = sech_pulse(grid, P0, FWHM).At   # x-axis: full N=1 soliton
Ay = copy(Ax)                          # y-axis: identical copy → 45° launch
vpulse = VectorialPulse(Ax, Ay, grid)

params = SimParams(;
    medium      = med_A,
    z_saves     = 200,
    solver      = SSFM(1e-3),   # fixed-step SSFM, 1 mm steps
    raman_model = nothing,
)

vsol_trap = solve(vpulse, params)

# ─── Case B: High birefringence — solitons walk off and separate ─────────────
# Five times larger birefringence → above threshold
disp_y_b = TaylorDispersion([beta2], 5 * delta_beta1)

med_B = BirefringentMedium(5.0, gamma, 0.0, disp_x, disp_y_b, 0.0, lambda0)
vsol_sep = solve(vpulse, SimParams(; medium=med_B, z_saves=200,
                                    solver=SSFM(1e-3), raman_model=nothing))

# ─── Diagnostic: temporal separation between polarization components ──────────
function centroid_t(A_col, t)
    S = abs2.(A_col)
    return sum(t .* S) / sum(S)
end

# Temporal separation at output  (should be small for trapping, large for walk-off)
t = vsol_trap.t
tc_x_trap = centroid_t(vsol_trap.At[:, 1, end], t)
tc_y_trap = centroid_t(vsol_trap.At[:, 2, end], t)
tc_x_sep  = centroid_t(vsol_sep.At[:, 1, end],  t)
tc_y_sep  = centroid_t(vsol_sep.At[:, 2, end],  t)

println("\nTemporal separation at z = 5 m:")
println("  Trapped  (δ = $(delta_beta1*1e12) ps/m): Δt = ",
        round((tc_y_trap - tc_x_trap)*1e12; sigdigits=2), " ps")
println("  Walk-off (δ = $(5*delta_beta1*1e12) ps/m): Δt = ",
        round((tc_y_sep - tc_x_sep)*1e12; sigdigits=2), " ps  (expect ≈ L·δ = ",
        round(5.0 * 5*delta_beta1*1e12; sigdigits=2), " ps in linear limit)")
```

## Expected Results

| Case | Walk-off ``\delta`` | Temporal separation at ``z = 5`` m |
|:---|:---|:---|
| Trapped | 1 ps/m (< threshold) | < 1 ps — solitons co-propagate |
| Separated | 5 ps/m (> threshold) | ≈ 25 ps — linear walk-off |

## Effect of Δβ₀ (Phase Mismatch)

A non-zero birefringence phase mismatch `deltabeta0` suppresses the coherent FWM coupling term.
For ``\Delta\beta_0 \gg 1/L`` the FWM term averages out and only SPM/XPM survive:

```julia
# High phase mismatch: FWM averages out
med_highbire = BirefringentMedium(
    5.0, gamma, 0.0, disp_x, disp_y,
    1e6,       # Δβ₀ = 10⁶ 1/m  (beat length ~ 6 µm — very high birefringence)
    lambda0,
)
vsol_hb = solve(vpulse, SimParams(; medium=med_highbire, z_saves=200,
                                    solver=SSFM(1e-3), raman_model=nothing))
```

## References

> C. R. Menyuk, "Stability of solitons in birefringent optical fibers. II. Arbitrary
> amplitudes," *J. Opt. Soc. Am. B* **5**, 392–402 (1988).
> DOI: [10.1364/JOSAB.5.000392](https://doi.org/10.1364/JOSAB.5.000392)

> C. R. Menyuk, "Nonlinear pulse propagation in birefringent optical fibers,"
> *IEEE J. Quantum Electron.* **23**, 174–176 (1987).
> DOI: [10.1109/JQE.1987.1073308](https://doi.org/10.1109/JQE.1987.1073308)
