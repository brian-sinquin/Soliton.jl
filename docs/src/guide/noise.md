```@meta
CurrentModule = Soliton
```

# Noise and Stochastic Modeling

Soliton provides a comprehensive suite of tools for modeling noise in optical systems, spanning from fundamental quantum vacuum fluctuations to classical laser noise and random birefringence (Polarization Mode Dispersion).

These tools are designed to support ensemble simulations, where multiple independent realizations of a noisy system are computed to extract statistical properties like spectral coherence.

## Pulse Noise (`add_noise`)

The [`add_noise`](@ref) function injects physically-motivated noise into an initial optical [`Pulse`](@ref). It supports four independent noise mechanisms:

```julia
noisy_pulse = add_noise(
    clean_pulse;
    rng = Random.default_rng(),
    photons_per_mode = 1.0,
    quantum_model = :gaussian,
    rin = 0.0,
    phase_rms = 0.0,
    linewidth_hz = 0.0
)
```

### 1. Quantum Noise (Spontaneous Emission Seed)

Quantum vacuum fluctuations act as the fundamental seed for nonlinear instabilities like Modulation Instability (MI) and Raman amplification.

- **`photons_per_mode`**: Sets the noise energy per spectral bin. The default is `1.0`, corresponding to the standard one-photon-per-mode model (Dudley & Coen, 2002). A value of `0.5` corresponds to the vacuum zero-point energy ($\hbar\omega / 2$). Set to `0.0` to disable.
- **`quantum_model`**:
    - `:gaussian` (default): Draws both quadratures from independent normal distributions. This provides Rayleigh-distributed amplitude and uniform phase, correctly modeling a coherent state or thermal vacuum.
    - `:phase_only`: Uses a fixed amplitude across all modes with a uniformly random phase. This is the classic seed used in early supercontinuum studies.

### 2. Relative Intensity Noise (RIN)

RIN models classical, shot-to-shot fluctuations in the laser's peak power.

- **`rin`**: The fractional RMS power fluctuation ($\sigma_P / P$). For example, `rin = 0.01` corresponds to a 1% RMS power jitter.

If you have a RIN specification in dBc/Hz over a specific bandwidth, you can convert it using the provided utility:
```julia
# Convert -140 dBc/Hz over a 10 GHz bandwidth to a fractional RMS RIN
my_rin = rin_rms(-140.0, 10e9)
```

### 3. Shot-to-Shot Phase Noise

Models a common-mode optical phase jitter applied to the entire pulse uniformly.

- **`phase_rms`**: The RMS phase fluctuation in radians.

### 4. Laser Linewidth (Colored Phase Noise)

Models the finite temporal coherence of the laser source as a Wiener (random-walk) phase process.

- **`linewidth_hz`**: The Full-Width at Half-Maximum (FWHM) of the laser's Lorentzian power spectrum. This generates a phase that drifts according to Brownian motion in the time domain.

## Polarization Mode Dispersion (PMD)

In vectorial simulations using `VectorialPulse` and `BirefringentMedium`, long fiber spans exhibit random variations in birefringence orientation and strength. This is modeled using the [`PMDElement`](@ref), a lumped element that applies a random first-order PMD realization.

```julia
# Create a PMD element with a mean Differential Group Delay (DGD) of 1 ps
pmd = PMDElement(1e-12)
```

The `PMDElement` draws a random DGD from a Maxwell-Boltzmann distribution (mean = `mean_dgd`) and a uniform random Principal State of Polarization (PSP) angle. It then applies the corresponding frequency-dependent Jones matrix to the pulse.

Because it is a `LumpedElement`, it can be easily cascaded in a piping workflow:

```julia
# Simulate a long link composed of multiple 10 km spans, each with 0.5 ps mean DGD
span = BirefringentMedium(10e3, ...)
pmd  = PMDElement(0.5e-12)

out_pulse = vpulse |> span |> pmd |> span |> pmd |> span
```

## Ensemble Coherence

To study the shot-to-shot stability of a process (like supercontinuum generation), you can simulate an ensemble of pulses, each with an independent noise realization, and compute the complex degree of first-order coherence $|g_{12}^{(1)}(\omega)|$.

```julia
function run_noisy_shot(clean_pulse, params)
    # Each call draws new random variables
    noisy_pulse = add_noise(clean_pulse; photons_per_mode=1.0)
    return solve(noisy_pulse, params)
end

# Run 20 independent simulations
ensemble = [run_noisy_shot(clean_pulse, params) for _ in 1:20]

# Compute the wavelength-resolved coherence
g12 = spectral_coherence(ensemble)
```
