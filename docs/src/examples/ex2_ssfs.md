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

```@example ex2
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
FWHM = 2 * log(1 + sqrt(2)) * T0
pulse = sech_pulse(grid, P0, FWHM)

# ─── Run: with Raman (SSFS) ─────────────────────────────────────────────────
params_raman = SimParams(;
    medium          = medium,
    z_saves         = 200,
    raman_model     = BlowWood(),
    self_steepening = false,
)
sol_raman = solve(pulse, params_raman; progress=false)

# ─── Run: without Raman (reference) ─────────────────────────────────────────
params_kerr = SimParams(;
    medium          = medium,
    z_saves         = 200,
    raman_model     = nothing,
    self_steepening = false,
)
sol_kerr = solve(pulse, params_kerr; progress=false)
```

```@example ex2; hide = true
using Plots
gr()

z_cm = sol_raman.Z .* 100
λ_raman = [begin
    S = abs2.(sol_raman.AW[:, i])
    ω_c = sum(sol_raman.W .* S) / sum(S)
    2π * 2.99792458e8 / ω_c * 1e9
end for i in axes(sol_raman.AW, 2)]

λ_kerr = [begin
    S = abs2.(sol_kerr.AW[:, i])
    ω_c = sum(sol_kerr.W .* S) / sum(S)
    2π * 2.99792458e8 / ω_c * 1e9
end for i in axes(sol_kerr.AW, 2)]

plot(z_cm, λ_raman, label="With Raman (SSFS)", xlabel="Distance z (cm)", ylabel="Centroid Wavelength (nm)", color=:crimson, lw=2.0)
plot!(z_cm, λ_kerr, label="No Raman (Kerr only)", color=:black, ls=:dash, lw=1.5, plot_title="Soliton Self-Frequency Shift (SSFS)")
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
