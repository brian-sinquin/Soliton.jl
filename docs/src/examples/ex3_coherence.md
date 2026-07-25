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

```@example ex3
using JuGNLSE
using Statistics

betas = [
    -11.83e-27, 8.1076e-41, -9.5229e-56,
     2.0737e-70, -5.3943e-85, 1.3486e-99,
    -2.5495e-114, 3.0524e-129, -1.7140e-144,
]
lambda0 = 835e-9
gamma   = 0.11
grid = create_grid(2^13, 12.5e-12, lambda0)

function run_ensemble(T_FWHM, P0, L, M=5)
    medium = Medium(; length=L, gamma=gamma, loss=0.0, betas=betas, lambda0=lambda0)
    params = SimParams(; medium=medium, z_saves=2, raman_model=Hollenbeck(), self_steepening=true)
    clean_pulse = sech_pulse(grid, P0, T_FWHM)
    solutions = Vector{Solution}(undef, M)
    for i in 1:M
        noisy_pulse = add_noise(clean_pulse; photons_per_mode=1.0)
        solutions[i] = solve(noisy_pulse, params; progress=false)
    end
    return solutions
end

sols_A = run_ensemble(50e-15, 10_000.0, 0.15)
g12_A = spectral_coherence(sols_A)
println("Mean coherence |g₁₂⁽¹⁾| (50 fs pump): ", round(mean(g12_A); digits=3))
```

```@example ex3; hide = true
using Plots
gr()

wl_nm = 2π * 2.99792458e8 ./ grid.W .* 1e9
plot(wl_nm, g12_A, label="50 fs Pump Coherence |g₁₂⁽¹⁾|", xlabel="Wavelength (nm)", ylabel="Degree of Coherence", xlims=(400, 1600), ylims=(0, 1.05), color=:navy, lw=1.5, plot_title="Supercontinuum Coherence Spectrum")
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
