```@meta
CurrentModule = JuGNLSE
```

# Example 2: Soliton Self-Frequency Shift

**Reproducing the Raman-induced spectral red-shift first observed by Mitschke & Mollenauer (1986)
and explained analytically by Gordon (1986)**

DOIs:
- [10.1364/OL.11.000659](https://doi.org/10.1364/OL.11.000659) — Mitschke & Mollenauer, Opt. Lett. **11**, 659 (1986)
- [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662) — Gordon, Opt. Lett. **11**, 662 (1986)

---

## Physical Background

When a fundamental soliton propagates in an anomalous-dispersion fiber, the Raman gain
spectrum causes the higher-frequency components to amplify the lower-frequency ones, producing
a continuous **self-frequency shift (SSFS)** toward longer wavelengths. Gordon's analysis
gives the shift rate:

```math
\frac{d\Omega_R}{dz} = -\frac{8 T_R}{15} \frac{|\beta_2|}{T_0^4}
```

where ``T_R \approx 3`` fs is the Raman slope parameter and ``T_0`` is the soliton half-width.
The shift is **stronger for shorter pulses** (scales as ``T_0^{-4}``).

## Simulation: Fundamental Soliton (N=1)

We propagate a fundamental sech² soliton (N = 1) over 10 soliton periods and track the
spectral centroid to directly observe the red-shift.

```julia
using JuGNLSE

# ─── Fiber parameters (standard telecom SMF-like) ───────────────────────────
lambda0 = 1550e-9       # [m] center wavelength
beta2   = -21.5e-27     # [s²/m] anomalous dispersion
gamma   = 0.0011        # [1/(W·m)] nonlinear coefficient

# ─── Soliton parameters ─────────────────────────────────────────────────────
T0   = 200e-15                        # soliton half-width [s]  (≈ 353 fs FWHM)
P0   = abs(beta2) / (gamma * T0^2)    # fundamental soliton peak power [W]
LD   = T0^2 / abs(beta2)              # dispersion length [m]
Zsol = (π / 2) * LD                   # soliton period [m]

println("Peak power P₀      = ", round(P0; sigdigits=4), " W")
println("Dispersion length  = ", round(LD; sigdigits=4), " m")
println("Soliton period     = ", round(Zsol * 1e2; sigdigits=3), " cm")

# ─── Grid and medium ────────────────────────────────────────────────────────
grid = create_grid(2^12, 50e-12, lambda0)

medium = Medium(;
    length  = 10 * Zsol,    # 10 soliton periods
    gamma   = gamma,
    loss    = 0.0,
    betas   = [beta2],
    lambda0 = lambda0,
)

# ─── Fundamental sech² soliton ──────────────────────────────────────────────
# FWHM = 2 * acosh(√2) * T0 = 2 * ln(1+√2) * T0
FWHM = 2 * log(1 + sqrt(2)) * T0
pulse = sech_pulse(grid, P0, FWHM)

# ─── Run: with Raman (SSFS) ─────────────────────────────────────────────────
params_raman = SimParams(;
    medium          = medium,
    z_saves         = 200,
    raman_model     = BlowWood(),   # Blow & Wood (1989)
    self_steepening = false,
)
sol_raman = solve(pulse, params_raman)

# ─── Run: without Raman (reference — no shift) ───────────────────────────────
params_kerr = SimParams(;
    medium          = medium,
    z_saves         = 200,
    raman_model     = nothing,
    self_steepening = false,
)
sol_kerr = solve(pulse, params_kerr)

# ─── Analyse: spectral centroid shift ────────────────────────────────────────
function centroid_wavelength(sol)
    [begin
        S = abs2.(sol.AW[:, i])
        ω_c = sum(sol.W .* S) / sum(S)
        2π * c / ω_c
    end for i in axes(sol.AW, 2)]
end

λ_raman = centroid_wavelength(sol_raman) .* 1e9  # [nm]
λ_kerr  = centroid_wavelength(sol_kerr)  .* 1e9

Δλ = λ_raman[end] - λ_kerr[end]
println("\nSpectral centroid shift after 10 soliton periods: ",
        round(Δλ; digits=1), " nm")

# ─── Gordon analytical estimate ──────────────────────────────────────────────
# dΩ/dz = -8 T_R |β₂| / (15 T₀⁴), T_R ≈ 3 fs
TR = 3e-15
dΩdz = -8 * TR * abs(beta2) / (15 * T0^4)
Ω_shift = dΩdz * 10 * Zsol
ω0 = 2π * c / lambda0
λ_gordon = 2π * c / (ω0 + Ω_shift) * 1e9  # [nm]
println("Gordon analytical centroid: ", round(λ_gordon - lambda0 * 1e9; digits=1), " nm shift")
```

## Expected Results

- **Without Raman**: soliton maintains its shape and spectral center indefinitely (soliton stability)
- **With Raman**: continuous red-shift proportional to ``z`` and ``T_0^{-4}``
- Shift after 10 soliton periods ≈ **10–15 nm** for T₀ = 200 fs in SMF

## Pulse-Width Dependence

To verify the ``T_0^{-4}`` scaling, run simulations for different pulse widths:

```julia
T0_values = [100e-15, 150e-15, 200e-15, 300e-15]

for T0 in T0_values
    P0   = abs(beta2) / (gamma * T0^2)
    LD   = T0^2 / abs(beta2)
    Zsol = (π/2) * LD
    FWHM = 2 * log(1 + sqrt(2)) * T0

    grid   = create_grid(2^12, max(20e-12, 30 * T0), lambda0)
    medium = Medium(; length=5*Zsol, gamma=gamma, loss=0.0, betas=[beta2], lambda0=lambda0)
    pulse  = sech_pulse(grid, P0, FWHM)
    params = SimParams(; medium=medium, z_saves=50, raman_model=BlowWood())
    sol    = solve(pulse, params)

    λ_out = 2π*c / (sum(sol.W .* abs2.(sol.AW[:,end])) / sum(abs2.(sol.AW[:,end]))) * 1e9
    λ_in  = 2π*c / (sum(sol.W .* abs2.(sol.AW[:,1  ])) / sum(abs2.(sol.AW[:,1  ]))) * 1e9
    println("T₀ = $(round(T0*1e15)) fs  →  Δλ = $(round(λ_out - λ_in; digits=1)) nm")
end
```

## References

> F. M. Mitschke and L. F. Mollenauer, "Discovery of the soliton self-frequency shift,"
> *Opt. Lett.* **11**, 659–661 (1986).
> DOI: [10.1364/OL.11.000659](https://doi.org/10.1364/OL.11.000659)

> J. P. Gordon, "Theory of the soliton self-frequency shift,"
> *Opt. Lett.* **11**, 662–664 (1986).
> DOI: [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662)

> K. J. Blow and D. Wood, "Theoretical description of transient stimulated Raman scattering
> in optical fibers," *IEEE J. Quantum Electron.* **25**, 2665–2673 (1989).
> DOI: [10.1109/3.40655](https://doi.org/10.1109/3.40655)
