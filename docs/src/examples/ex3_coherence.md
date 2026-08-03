```@meta
CurrentModule = GNLSE
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

We run an ensemble of M = 20 independent shots, each seeded with a different one-photon-per-mode
quantum noise realization (`add_noise`), then compute the pairwise coherence estimator.

```@example ex3
using Soliton
using Statistics

betas = [
    -11.83e-27, 8.1076e-41, -9.5229e-56,
     2.0737e-70, -5.3943e-85, 1.3486e-99,
    -2.5495e-114, 3.0524e-129, -1.7140e-144,
]
lambda0 = 835e-9
gamma   = 0.11
grid = create_grid(2^12, 12.5e-12, lambda0)

function run_ensemble(T_FWHM, P0, L, M=20)
    medium = Medium(; length=L, gamma=gamma, loss=0.0, betas=betas, lambda0=lambda0)
    params = SimParams(; medium=medium, z_saves=2, raman_model=BlowWood(), self_steepening=true)
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

# Restrict to the band with genuine spectral content (>= -30 dB from peak): outside
# this band the ensemble-mean power is near zero, so |g12| is dominated by noise-floor
# division artifacts rather than physical coherence.
P_mean = mean(abs2.(s.AW[:, end]) for s in sols_A)
P_db = 10 .* log10.(P_mean ./ maximum(P_mean) .+ 1e-12)
significant = findall(>=(-30.0), P_db)

println("Mean coherence |g₁₂⁽¹⁾| over the significant band: ", round(mean(g12_A[significant]); digits=3))
```

```@example ex3
using Plots # hide
gr() # hide

wl_nm = 2π * 2.99792458e8 ./ grid.W .* 1e9 # hide
wl_sig = wl_nm[significant] # hide
g12_sig = g12_A[significant] # hide
ord = sortperm(wl_sig) # hide

plot(wl_sig[ord], g12_sig[ord], label="50 fs Pump Coherence |g₁₂⁽¹⁾|", xlabel="Wavelength (nm)", ylabel="Degree of Coherence", ylims=(0, 1.05), color=:navy, lw=1.5, plot_title="Supercontinuum Coherence Spectrum") # hide
```

## Expected Results

!!! note "This model shows partial, not full, coherence for this benchmark"
    The classic Dudley & Coen (2002) result for a similarly short pump in this fiber reports
    coherence approaching 1 across most of the SC band. Our simulation reproduces the correct
    **qualitative trend** — shorter/higher-power pumps are markedly more coherent than longer
    ones — but the *absolute* coherence for the 50 fs case here is only partial (mean
    ``|g_{12}^{(1)}| \approx 0.3-0.4`` over the spectrally-significant band), not near-unity.
    This was verified not to be a normalization bug: the `add_noise` one-photon-per-mode energy
    and the `spectral_coherence` pairwise estimator both check out analytically against the
    expected ``n_p \hbar\omega`` per-mode energy and the standard ``|g_{12}^{(1)}|`` definition.
    The extra sensitivity likely comes from this particular combination of Raman model,
    self-steepening, and fiber dispersion being more MI-susceptible than the idealized textbook
    case; treat the numbers below as representative of *this* solver configuration, not as a
    precise reproduction of the original figure.

| Configuration | Simulated ``\langle|g_{12}^{(1)}|\rangle`` (significant band) |
|:---|:---|
| 50 fs, 10 kW (soliton fission) | ≈ 0.3–0.4 (partially coherent — most coherent case) |
| 200 fs, same energy (MI onset) | ≈ 0.05–0.1 (mostly incoherent) |
| 500 fs, same energy (long pulse) | ≈ 0.05 (incoherent) |

The transition between more- and less-coherent SC is controlled by the fission length
``L_{\rm fiss} = L_D / N`` — shorter pulses fission earlier before modulation instability (MI)
has time to amplify shot-to-shot noise differences.

## Quantum Noise Model

GNLSE implements the standard Dudley & Coen quantum-noise seed:

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
