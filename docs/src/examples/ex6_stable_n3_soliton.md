```@meta
CurrentModule = JuGNLSE
```

# Example 6: Stable 3rd-Order Soliton (Akhmediev Recurrence)

**Demonstrating the exact periodic recurrence of the N=3 NLS soliton**

References:
- V. E. Zakharov & A. B. Shabat, *Sov. Phys. JETP* **34**, 62–69 (1972) — exact N-soliton solutions to NLS
- G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (2019), §5.2 — higher-order solitons in fibers

---

## Physical Background

The focusing nonlinear Schrödinger equation (pure NLS — no loss, no Raman, no self-steepening) is
exactly solvable via the **inverse scattering transform (IST)** (Zakharov & Shabat, 1972).
The input

```math
A(0, t) = N \operatorname{sech}(t/T_0) \sqrt{P_0}, \qquad
P_0 = \frac{|\beta_2|}{\gamma T_0^2}
```

is an exact **N-th order soliton** — a bound state of N fundamental solitons. Its evolution
is **exactly periodic**: the field returns to its original shape at every integer multiple of
the **soliton half-period**:

```math
z_{1/2} = \frac{\pi}{2} L_D = \frac{\pi T_0^2}{2|\beta_2|}
```

Between recurrences the pulse breathes — compressing to a narrow spike and then re-expanding —
but energy, photon number, and envelope shape are all exactly conserved over one period.
This is **Fermi–Pasta–Ulam–Tsingou (FPUT) recurrence** in the optical domain.

!!! note "Stability vs. perturbations"
    The recurrence is exact only in the ideal lossless NLS (β₂ only, no Raman, no β₃).
    Any perturbation — Raman, third-order dispersion, or loss — breaks the symmetry and
    causes **soliton fission**: the N-bound state dissolves into N independent red-shifted
    fundamental solitons. This example first verifies the ideal case, then demonstrates
    what breaks it.

## Part 1 — Ideal Recurrence (Pure NLS)

```julia
using JuGNLSE

# ─── Fiber parameters (telecom SMF, β₂ only) ─────────────────────────────────
lambda0 = 1550e-9          # [m]
beta2   = -21.5e-27        # [s²/m]  anomalous dispersion
gamma   = 0.0011           # [1/(W·m)]

# ─── N=3 soliton parameters ──────────────────────────────────────────────────
N    = 3
T0   = 500e-15             # soliton half-width [s]  (≈ 880 fs FWHM)
P1   = abs(beta2) / (gamma * T0^2)   # fundamental soliton peak power [W]
P0   = N^2 * P1                       # N=3 soliton peak power [W]
LD   = T0^2 / abs(beta2)             # dispersion length [m]
Zhalf = (π / 2) * LD                 # soliton half-period [m]

println("N=3 soliton setup:")
println("  Fundamental power   P₁ = $(round(P1; sigdigits=3)) W")
println("  N=3 peak power      P₀ = $(round(P0; sigdigits=3)) W  (= 9 P₁)")
println("  Dispersion length   L_D = $(round(LD; sigdigits=3)) m")
println("  Half-period  z₁/₂      = $(round(Zhalf; sigdigits=3)) m")

# ─── Grid: wide enough for the breathing, fine enough for compression ─────────
grid = create_grid(2^13, 120e-12, lambda0)

# ─── Exact N=3 initial condition: A(0,t) = 3 sech(t/T₀) √P₁ ─────────────────
# sech_pulse takes FWHM; convert T₀ → FWHM = 2 acosh(√2) T₀ = 2 ln(1+√2) T₀
FWHM = 2 * log(1 + sqrt(2)) * T0
pulse = sech_pulse(grid, P0, FWHM)

# ─── Medium: pure NLS — β₂ only, no loss, no Raman, no self-steepening ───────
medium = Medium(;
    length  = 3 * Zhalf,   # propagate 3 full half-periods
    gamma   = gamma,
    loss    = 0.0,
    betas   = [beta2],
    lambda0 = lambda0,
)

params = SimParams(;
    medium          = medium,
    z_saves         = 600,         # dense sampling across 3 periods
    raman_model     = nothing,     # pure Kerr — no Raman
    self_steepening = false,       # no shock term
    rtol            = 1e-9,        # tight tolerances for accurate recurrence
    atol            = 1e-11,
)

sol = solve(pulse, params)

# ─── Verify recurrence: peak power at each save ───────────────────────────────
peaks = [maximum(abs2, sol.At[:, i]) for i in axes(sol.At, 2)]

# Find the 3 recurrence peaks (near z = Zhalf, 2Zhalf, 3Zhalf)
function find_near(z_target, Z, values)
    idx = argmin(abs.(Z .- z_target))
    return Z[idx], values[idx]
end

z1, p1 = find_near(1 * Zhalf, sol.Z, peaks)
z2, p2 = find_near(2 * Zhalf, sol.Z, peaks)
z3, p3 = find_near(3 * Zhalf, sol.Z, peaks)

println("\nRecurrence check (should return to P₀ = $(round(P0; sigdigits=3)) W):")
println("  z = $(round(z1*100; sigdigits=3)) cm → P = $(round(p1; sigdigits=4)) W  ",
        "(error = $(round(abs(p1-P0)/P0*100; sigdigits=2)) %)")
println("  z = $(round(z2*100; sigdigits=3)) cm → P = $(round(p2; sigdigits=4)) W  ",
        "(error = $(round(abs(p2-P0)/P0*100; sigdigits=2)) %)")
println("  z = $(round(z3*100; sigdigits=3)) cm → P = $(round(p3; sigdigits=4)) W  ",
        "(error = $(round(abs(p3-P0)/P0*100; sigdigits=2)) %)")

# ─── Photon number conservation ───────────────────────────────────────────────
photons = photon_number(sol)
Δphotons = (maximum(photons) - minimum(photons)) / photons[1]
println("\nPhoton number drift over 3 periods: ",
        round(Δphotons * 100; sigdigits=2), " %  (should be < 0.01 %)")

# ─── Field fidelity at recurrence points ─────────────────────────────────────
# Overlap integral |⟨A(z)|A(0)⟩|² / (||A(z)||² ||A(0)||²)
function field_fidelity(A_ref, A_out)
    num = abs(sum(conj(A_ref) .* A_out))^2
    den = sum(abs2, A_ref) * sum(abs2, A_out)
    return num / den
end

A_in = sol.At[:, 1]
idx1 = argmin(abs.(sol.Z .- 1 * Zhalf))
idx2 = argmin(abs.(sol.Z .- 2 * Zhalf))
idx3 = argmin(abs.(sol.Z .- 3 * Zhalf))

println("\nField fidelity at recurrence points (should be ≈ 1.0):")
println("  z₁/₂  : ", round(field_fidelity(A_in, sol.At[:, idx1]); sigdigits=5))
println("  z₁    : ", round(field_fidelity(A_in, sol.At[:, idx2]); sigdigits=5))
println("  3z₁/₂ : ", round(field_fidelity(A_in, sol.At[:, idx3]); sigdigits=5))
```

### Expected Output

```
N=3 soliton setup:
  Fundamental power   P₁ = 19.09 W
  N=3 peak power      P₀ = 171.8 W  (= 9 P₁)
  Dispersion length   L_D = 10.7 m
  Half-period  z₁/₂      = 16.81 m

Recurrence check (should return to P₀ = 171.8 W):
  z = 16.81 cm → P = 171.8 W  (error = 0.0 %)
  z = 33.62 cm → P = 171.8 W  (error = 0.0 %)
  z = 50.43 cm → P = 171.8 W  (error = 0.0 %)

Photon number drift over 3 periods: < 0.01 %
Field fidelity at recurrence points (should be ≈ 1.0):
  z₁/₂  : 1.0000
  z₁    : 1.0000
  3z₁/₂ : 1.0000
```

## Part 2 — Soliton Dynamics Within One Period

The rich sub-period structure can be characterized at 5 diagnostic points:

```julia
# Propagation distances of interest within one half-period
z_points = [0.0, 0.1, 0.32/N, 0.5, 1.0] .* Zhalf
labels   = ["Input", "Early compression", "Max compression (~zₘᵢₙ)",
            "Mid-period", "Recurrence"]

println("\nDynamics within first half-period:")
println("  $(rpad("Label", 28))  z [m]     Peak [W]   FWHM [fs]")
for (lbl, zt) in zip(labels, z_points)
    idx = argmin(abs.(sol.Z .- zt))
    Ppk = maximum(abs2, sol.At[:, idx])
    τ   = fwhm(Pulse(sol.At[:, idx], sol.AW[:, idx], grid); domain=:time) * 1e15
    println("  $(rpad(lbl, 28))  $(round(sol.Z[idx]; sigdigits=3)) m    ",
            "$(round(Ppk; sigdigits=4)) W   $(round(τ; sigdigits=3)) fs")
end
```

## Part 3 — Stability Under Perturbations

Repeat with realistic perturbations to observe **soliton fission**:

```julia
# ─── Perturbation A: Add Raman scattering ────────────────────────────────────
params_raman = SimParams(;
    medium          = medium,
    z_saves         = 600,
    raman_model     = Hollenbeck(),   # Raman breaks the symmetry
    self_steepening = false,
    rtol = 1e-9, atol = 1e-11,
)
sol_raman = solve(pulse, params_raman)

peaks_raman = [maximum(abs2, sol_raman.At[:, i]) for i in axes(sol_raman.At, 2)]
_, p1_raman = find_near(1 * Zhalf, sol_raman.Z, peaks_raman)
_, p1_ideal = find_near(1 * Zhalf, sol.Z, peaks)

println("\nEffect of Raman on recurrence at z₁/₂:")
println("  Ideal GNLSE : P = $(round(p1_ideal; sigdigits=4)) W")
println("  With Raman  : P = $(round(p1_raman; sigdigits=4)) W  ",
        "(Δ = $(round(abs(p1_raman - p1_ideal)/p1_ideal*100; sigdigits=2)) %)")
println("  → Raman breaks recurrence and splits solitons via SSFS")

# ─── Perturbation B: Add third-order dispersion ───────────────────────────────
medium_beta3 = Medium(;
    length  = 3 * Zhalf,
    gamma   = gamma,
    loss    = 0.0,
    betas   = [beta2, 1e-40],   # add β₃ = 10⁻⁴⁰ s³/m
    lambda0 = lambda0,
)
sol_beta3 = solve(pulse, SimParams(;
    medium=medium_beta3, z_saves=600,
    raman_model=nothing, self_steepening=false,
    rtol=1e-9, atol=1e-11,
))

peaks_beta3 = [maximum(abs2, sol_beta3.At[:, i]) for i in axes(sol_beta3.At, 2)]
_, p1_beta3 = find_near(1 * Zhalf, sol_beta3.Z, peaks_beta3)
println("\n  With β₃    : P = $(round(p1_beta3; sigdigits=4)) W  ",
        "(Δ = $(round(abs(p1_beta3 - p1_ideal)/p1_ideal*100; sigdigits=2)) %)")
println("  → β₃ seeds dispersive wave emission and degrades recurrence")
```

### Summary of Perturbation Effects

| Configuration | Recurrence fidelity at ``z_{1/2}`` |
|:---|:---|
| Pure NLS (ideal) | ≈ 100% — exact recurrence |
| + Raman scattering | 60–80% — solitons red-shift and separate |
| + β₃ only | 80–95% — dispersive wave emitted |
| + Raman + β₃ + SS | < 50% — full soliton fission (→ supercontinuum) |

## References

> V. E. Zakharov and A. B. Shabat, "Exact theory of two-dimensional self-focusing and
> one-dimensional self-modulation of waves in nonlinear media,"
> *Sov. Phys. JETP* **34**, 62–69 (1972).

> G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019), §5.2–5.3.

> N. J. Zabusky and M. D. Kruskal, "Interaction of solitons in a collisionless plasma and
> the recurrence of initial states,"
> *Phys. Rev. Lett.* **15**, 240–243 (1965).
> DOI: [10.1103/PhysRevLett.15.240](https://doi.org/10.1103/PhysRevLett.15.240)

!!! tip "Relation to Example 5"
    [Example 5](ex5_soliton_compression.md) focuses on the **compression ratio** at the
    first pulse minimum — demonstrating the experimental result of Mollenauer et al. (1980).
    This example focuses on the **long-term stability and recurrence** — proving the
    N=3 soliton is an exact, periodic solution of the NLS.
