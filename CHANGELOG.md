# Changelog

All notable changes to Soliton.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2026-09-19

### Added

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
