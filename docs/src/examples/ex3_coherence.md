```@meta
CurrentModule = JuGNLSE
```

# Example 3: Supercontinuum Coherence

**Reproducing Fig. 2 of Dudley & Coen, Opt. Lett. 27, 1180 (2002)**

DOI: [10.1364/OL.27.001180](https://doi.org/10.1364/OL.27.001180)

---

## Physical Background

Supercontinuum generation driven by **modulation instability (MI)** is highly sensitive to
quantum noise. Each shot in an experiment starts from a slightly different vacuum-fluctuation
seed, leading to shot-to-shot spectral fluctuations that reduce the degree of coherence:

```math
\left|g_{12}^{(1)}(\omega)\right| =
\frac{\left|\langle A_i^*(\omega) A_j(\omega)\rangle_{i \neq j}\right|}{\langle |A(\omega)|^2\rangle}
\in [0, 1]
```

Dudley & Coen showed that for **short femtosecond pulses** (high N), coherence
``|g_{12}^{(1)}|`` approaches 1 over the full SC bandwidth ("coherent regime"), while for
**longer pulses or CW** (MI-seeded), coherence degrades to ~0 ("incoherent regime").

## Simulation: Ensemble of Noisy Shots

We run an ensemble of M = 20 independent shots, each seeded with a different quantum noise
realization, then compute the pairwise coherence estimator.

```julia
using JuGNLSE

# ─── Fiber parameters: same PCF as Dudley 2006 ──────────────────────────────
betas = [
    -11.83e-27, 8.1076e-41, -9.5229e-56,
     2.0737e-70, -5.3943e-85, 1.3486e-99,
    -2.5495e-114, 3.0524e-129, -1.7140e-144,
]
lambda0 = 835e-9
gamma   = 0.11

# ─── Grid ───────────────────────────────────────────────────────────────────
grid = create_grid(2^13, 12.5e-12, lambda0)

# ─── Helper: run one ensemble ────────────────────────────────────────────────
function run_ensemble(T_FWHM, P0, L, M=20)
    medium = Medium(;
        length  = L,
        gamma   = gamma,
        loss    = 0.0,
        betas   = betas,
        lambda0 = lambda0,
    )
    params = SimParams(;
        medium          = medium,
        z_saves         = 2,           # only need input + output
        raman_model     = Hollenbeck(),
        self_steepening = true,
    )

    # Clean (noise-free) pulse
    clean_pulse = sech_pulse(grid, P0, T_FWHM)

    # Run M noisy shots
    solutions = Vector{Solution}(undef, M)
    for i in 1:M
        noisy_pulse = add_noise(clean_pulse; photons_per_mode=1.0)
        solutions[i] = solve(noisy_pulse, params)
    end
    return solutions
end

# ─── Case A: 50 fs pulse, 10 kW — coherent SC regime ────────────────────────
println("Case A: 50 fs pump (coherent regime)...")
sols_A = run_ensemble(50e-15, 10_000.0, 0.15)

# ─── Case B: 150 fs pulse, same energy — MI-dominated, incoherent regime ────
# Scale P₀ to keep same pulse energy (E ∝ P₀ × T_FWHM)
println("Case B: 150 fs pump (incoherent regime)...")
E_ref = 10_000.0 * 50e-15           # reference energy
P0_B  = E_ref / 150e-15             # same energy, longer pulse
sols_B = run_ensemble(150e-15, P0_B, 0.15)

# ─── Coherence ───────────────────────────────────────────────────────────────
g12_A = spectral_coherence(sols_A)   # wavelength-resolved |g₁₂⁽¹⁾(ω)|
g12_B = spectral_coherence(sols_B)

using Statistics
println("\nMean coherence |g₁₂⁽¹⁾|:")
println("  50 fs pump  : ", round(mean(g12_A); digits=3), " (expect ≈ 1.0)")
println("  150 fs pump : ", round(mean(g12_B); digits=3), " (expect < 0.5)")

# ─── Energy conservation check ───────────────────────────────────────────────
photons_A = photon_number.(sols_A)
println("\nPhoton number drift (shot 1): ",
    round((photons_A[1][end] - photons_A[1][1]) / photons_A[1][1] * 100; sigdigits=2), " %")
```

## Expected Results

| Configuration | Expected ``\langle|g_{12}^{(1)}|\rangle`` |
|:---|:---|
| 50 fs, 10 kW (soliton fission) | ≈ 0.95–1.00 (fully coherent) |
| 150 fs, same energy (MI onset) | ≈ 0.3–0.6 (partial coherence) |
| CW / long pulse | ≈ 0.0–0.1 (incoherent) |

The transition between coherent and incoherent SC is controlled by the fission length
``L_{\rm fiss} = L_D / N`` — shorter pulses fission earlier before MI develops.

## Quantum Noise Model

JuGNLSE implements the standard Dudley & Coen quantum-noise seed:

```julia
# One-photon-per-mode quantum noise (the default)
noisy = add_noise(pulse; photons_per_mode=1.0, quantum_model=:gaussian)

# Alternative: Dudley & Coen phase-only seed (fixed amplitude, random phase)
noisy = add_noise(pulse; photons_per_mode=1.0, quantum_model=:phase_only)

# Add laser RIN and phase noise
noisy = add_noise(pulse;
    rin       = rin_rms(-150.0, 1e9),  # -150 dBc/Hz over 1 GHz
    phase_rms = 0.01,                  # 10 mrad RMS phase jitter
)
```

## References

> J. M. Dudley and S. Coen, "Coherence properties of supercontinuum spectra generated in
> photonic crystal and tapered optical fibers,"
> *Opt. Lett.* **27**, 1180–1182 (2002).
> DOI: [10.1364/OL.27.001180](https://doi.org/10.1364/OL.27.001180)

> J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in photonic crystal fiber,"
> *Rev. Mod. Phys.* **78**, 1135–1184 (2006). Section VI.B.
> DOI: [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)
