# Changelog

All notable changes to Soliton.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2026-09-20

### Added

- **B-integral self-focusing indicator** (`b_integral`, `b_integral_profile`): the accumulated nonlinear (Kerr) phase `B(z) = ∫ γ(z')·P_peak(z') dz'` along a propagation, the standard diagnostic for self-focusing/beam-breakup risk in high-peak-power laser chains (CPA amplifiers, fiber amplifiers). Supports `Solution` and `VectorialSolution`, every `NonlinearityModel` variant (constant, z-dependent, frequency-dependent, effective-area), and a `SimParams` convenience overload. Documented in the physics guide and API reference, with analytic-invariant regression tests (pure-SPM peak-power conservation gives a closed-form `B = γP₀L` check, plus a tapered-γ(z) closed form).
- **Three-Photon Absorption (3PA) support in `SemiconductorMedium`** ([src/types.jl](src/types.jl), [src/nonlinearity.jl](src/nonlinearity.jl)):
  - New `alpha3` field (α₃, units m³/W²; default `0.0`, i.e. disabled) modeling the nonlinear-loss channel that dominates over TPA in the mid-infrared (roughly 2.2–4 μm), where the photon energy falls below half the bandgap.
  - Field-domain loss term `-α₃/(2·Aeff²)·|A|⁴·A` and an additive free-carrier generation term `α₃/(3ħω₀Aeff³)·|A|⁶` in `_semiconductor_spm`, consistent with the existing TPA formalism (Lin, Painter & Agrawal, *Opt. Express* **15**, 16604 (2007)).
  - `alpha3 = 0` reproduces prior TPA-only behavior exactly (verified to `rtol=1e-12`).
- **Analytic ground-truth cross-validation** (`test/test_semiconductor.jl`): a dedicated, independently-coded fine RK4 integrator solves the decoupled scalar ODE `dP/dz = -α₂P²/Aeff - α₃P³/Aeff²` (valid with dispersion/Kerr/FCA/FCR disabled) and is compared against `SemiconductorMedium`'s full propagation output to `rtol=1e-5`, for TPA-only, 3PA-only, and combined cases.
- **External adversarial validation for TPA** (`test/generate_reference_data.py`, `test/test_adversarial.jl`, Scenario 7): TPA is now cross-checked against `gnlse-python`'s independently-implemented solver (`scipy.solve_ivp`) using the standard "complex γ = Kerr + i·TPA" trick (`γ = γ_r + i·α₂/(2·Aeff)`), which its Kerr-only nonlinear step accepts without modification. 21 peak-power checkpoints (1% tolerance) plus a full-field cross-correlation (≥0.999) at the waveguide output.
- **Three new documentation examples**, replacing the single combined TPA/FCR example:
  - [Example 8 — Silicon TPA Optical Limiter](docs/src/examples/ex8_silicon_tpa.md): power sweep showing the classic TPA saturable-limiter curve.
  - [Example 11 — Mid-IR Three-Photon Absorption](docs/src/examples/ex11_silicon_3pa.md): TPA (∝P) vs 3PA (∝P²) fractional-loss power-law scaling, the experimental fingerprint distinguishing the two channels.
  - [Example 12 — Free-Carrier Lifetime Pump-Probe](docs/src/examples/ex12_freecarrier_decay.md): reconstructs a standard pump-probe carrier-lifetime measurement, recovering the input `τ_c` from a delay sweep.
- **"Known Limitations" section** added to the `SemiconductorMedium` docstring and the semiconductor guide, explicitly documenting: the shared-`Aeff` approximation across nonlinear orders, linear-only carrier recombination (no Auger/diffusion), the omitted 3PA-associated nonlinear-refraction term, the self-steepening scope, and per-channel validation status (TPA: external + internal; 3PA/FCA/FCR: internal ODE ground-truth only — no suitable external open package was found; PyNLO, MEEP, and Tidy3D were evaluated and ruled out).
- **`test/test_coverage.jl`**: targeted regression tests for public exports that had no direct test coverage — `lorentzian_pulse`'s analytic envelope, `rin_rms`/`add_noise` RIN and quantum-noise statistics (checked against expected mean/std and an analytic noise-energy formula), the `photon_number` ↔ absolute-photon-count relationship, `spectral_centroid` under a known frequency shift, `dispersion_length`/`nonlinear_length`/`soliton_number` against textbook values, `spectrogram`'s output shape/positivity, `track_solitons` on a stable N=1 soliton, and `dispersive_wave_wavelength` phase-matching against a constructed β₃ with a known root — plus a solver × medium × Raman × self-steepening combination matrix (`ERK4IP`/`SSFM` × Raman on/off × shock on/off).
- Angular-frequency/wavelength conversion helpers `wavelength_to_omega` and
  `omega_to_wavelength`, matching the `ω = 2πc/λ` convention used internally
  by `Grid`/`Medium`.
- General-purpose decibel/power-ratio conversions: `db_to_linear_power`,
  `linear_power_to_db`, `db_to_linear_amplitude`, `linear_amplitude_to_db`,
  `db_to_np`, `np_to_db`, `dbm_to_watt`, `watt_to_dbm`.
- `photon_energy` (E = ħω = hc/λ), and `n2_aeff_to_gamma` /
  `gamma_aeff_to_n2` for converting between nonlinear refractive index n₂,
  effective mode area, and the waveguide nonlinear coefficient γ.
- Analytic pulse-shape energy/peak-power relations `pulse_energy_estimate`
  and `peak_power_estimate` for the sech/Gaussian/Lorentzian envelopes, plus
  repetition-rate helpers `average_power`, `pulse_energy_from_average`, and
  `peak_power_from_average` for converting lab-style average-power/rep-rate
  measurements to per-pulse energy and peak power.
- `soliton_period`, the fundamental soliton period z₀ = (π/2)L_D referenced
  in `soliton_number`'s docstring.
- Modulation-instability analysis: `modulation_instability_gain`,
  `mi_peak_frequency`, and `mi_bandwidth` for the scalar MI gain spectrum of
  a CW/quasi-CW pump in anomalous dispersion.
- `instantaneous_frequency`, the time-domain chirp δω(t) = -dφ/dt of a pulse
  envelope, via unwrapped-phase finite differences.

### Changed

- **`SSFM`/`AdaptiveSSFM` were documented as symmetric second-order Strang splitting; they are not.** The nonlinear substep in both is explicit Euler, making the overall method first-order despite the surrounding linear half-steps. Docstrings on `SSFM`, `AdaptiveSSFM`, and both `propagate` methods now carry an explicit warning correcting this, backed by a new empirical convergence test (`test/test_solvers.jl`) that measures the convergence order across a 4x step-count sweep for the fundamental-soliton exact solution and asserts it lands at first order (0.9–1.2), not second (~2).
- Added missing/expanded docstrings for `pulse_energy`/`peak_power` (`VectorialPulse` methods), `spectral_bandwidth`, `photon_number` (including how to recover an absolute photon count), `rin_rms`, `lorentzian_pulse`, and `cw_pulse`.
- Self-steepening scope correction: the shock/self-steepening weighting is now applied only to the Kerr/TPA/3PA nonlinear response, not to Free-Carrier Absorption/Refraction (FCA/FCR), which are a physically separate, slowly-varying process. Implemented via a new `buf_f2` frequency-domain buffer on `PhysicsModel`, letting the two contributions be FFT'd and combined separately.
- `README.md`, `docs/src/index.md`, `docs/src/physics.md`, `docs/src/guide/semiconductor.md`, `docs/src/examples/index.md`: updated to mention 3PA and the B-integral alongside existing features and reference the new examples.
- `docs/make.jl`: navigation updated for the two new example pages (11, 12) and the renamed Example 8.
- CI: the Documentation workflow job now runs with `JULIA_NUM_THREADS=auto`. Example 10 ("Multithreaded Parameter Sweep") documents itself as needing multiple threads to be fast, but the job previously always ran single-threaded, so that one `@example` block alone could dominate the build — this cut the job's wall time from ~10 minutes to ~5.5 minutes. Also dropped a redundant standalone `doctest(Soliton)` step: `makedocs` already runs doctests as part of the main build, and the package currently has no `jldoctest` blocks for the extra step to have covered.
- CI: the benchmark workflow no longer runs on pull requests (only on pushes to `master`/tags and manual dispatch). It doesn't gate anything — no regression threshold is enforced, it just posts a summary — so running it on every PR added a non-blocking ~3 minute check with nothing actionable for reviewers; per-commit history on `master` remains the meaningful place to track it.
- Unified the several ad hoc `10^(x/10)`/`10^(x/20)`/`log(10)/10` decibel
  conversions scattered across `elements.jl` (`Amplifier`/`Attenuator`),
  `dispersion.jl` (`Medium.loss`/gain dB → Np), `types.jl`
  (`AmplifyingMedium`'s `g0_db`), `nonlinearity.jl` (ASE noise figure),
  `analysis.jl` (`rin_rms`), and `fibers.jl` (`HollowCoreFiber`'s gamma and
  confinement-loss dB conversion) to call the new named conversion functions.
  Purely a naming/DRY cleanup — the underlying formulas and numerical results
  are unchanged.
- Reduced `instantaneous_frequency` allocation by ~9x (measured at N=4096) by
  isolating its finite-difference loop behind a function barrier, avoiding
  boxing from `Pulse`'s type-erased `grid` field.
- Local docs builds (`DOCS_DRAFT=true julia --project=docs docs/make.jl`) can
  now skip executing the example pages' full GNLSE solves for faster
  iteration; CI is unaffected and still runs every example to completion.

### Fixed

- **`spectral_centroid` docstring had the sign convention backwards**: it read "positive/negative for red/blue shifts", but ⟨ω−ω₀⟩ > 0 means a *higher* frequency relative to the carrier, i.e. a **blue** shift (and negative ⟨ω−ω₀⟩ is red). Corrected to "positive/negative for blue/red shifts".
- Silicon photonics docs and examples previously implied TPA/FCA/FCR were validated the same way as the core Kerr/Raman/dispersion engine (against `gnlse-python`); this is now accurately scoped per nonlinear channel.
- A unit test's decibel cross-check asserted the wrong expected value
  (double-counted the amplitude/power dB squaring); no library code was
  affected.

## [0.2.2] - 2026-09-07

### Fixed

- Preserve FFT ordering when converting scalar and vectorial `Solution` objects
  back to pulses, including when frequency-domain output was not saved.
- Apply scalar and vectorial filter transfer functions in the same FFT ordering
  as pulse spectra.
- Correct Kerr normalization in amplifying media and keep saturated gain
  independent of the nonlinear spectral coefficient, with and without Raman.
- Compute photon number correctly from time-domain output when frequency-domain
  output was not saved.
- Normalize spectral ordering in pulse and solution coherence calculations, and
  support solutions without saved frequency-domain output.
- Use the correct detuning-frequency axis for SHG FROG traces.
- Reject grids with fewer than two points or an odd number of points, which are
  incompatible with the package's FFT grid conventions.

### Changed

- Clarified the frequency-domain storage conventions for scalar and vectorial
  pulses and the input convention for filter transfer functions.
- Expanded regression coverage for cascades, filters, nonlinear gain, Raman,
  self-steepening, spectral analysis, and FFT edge cases.

## [0.2.1] - 2026-08-03

### Changed

- Renamed the package and Julia module from GNLSE.jl to Soliton.jl while
  preserving the package UUID.
- Updated documentation, tests, workflows, and precompiled sysimage artifact
  names for the Soliton.jl name.
- Updated GitHub Actions dependencies used by the release workflows.

### Added

- Added the Zenodo DOI badge and citation metadata.

## [0.2.0] - 2026-08-02

### Changed

- Renamed JuGNLSE.jl to GNLSE.jl for Julia General registry compatibility.

[Unreleased]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/brian-sinquin/Soliton.jl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brian-sinquin/Soliton.jl/releases/tag/v0.2.0
