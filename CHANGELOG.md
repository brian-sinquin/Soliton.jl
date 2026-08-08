# Changelog

All notable changes to Soliton.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — `three-photon-absorption` branch

### Added

- **Three-Photon Absorption (3PA) support in `SemiconductorMedium`** ([src/types.jl](src/types.jl), [src/nonlinearity.jl](src/nonlinearity.jl)):
  - New `alpha3` field (α₃, units m³/W²; default `0.0`, i.e. disabled) modeling the nonlinear-loss channel that dominates over TPA in the mid-infrared (roughly 2.2–4 μm), where the photon energy falls below half the bandgap.
  - Field-domain loss term `-α₃/(2·Aeff²)·|A|⁴·A` and an additive free-carrier generation term `α₃/(3ħω₀Aeff³)·|A|⁶` in `_semiconductor_spm`, consistent with the existing TPA formalism (Lin, Painter & Agrawal, *Opt. Express* **15**, 16604 (2007)).
  - `alpha3 = 0` reproduces prior TPA-only behavior exactly (verified to `rtol=1e-12`).
- **Self-steepening scope correction**: the shock/self-steepening weighting is now applied only to the Kerr/TPA/3PA nonlinear response, not to Free-Carrier Absorption/Refraction (FCA/FCR), which are a physically separate, slowly-varying process. Implemented via a new `buf_f2` frequency-domain buffer on `PhysicsModel`, letting the two contributions be FFT'd and combined separately.
- **Analytic ground-truth cross-validation** (`test/test_semiconductor.jl`): a dedicated, independently-coded fine RK4 integrator solves the decoupled scalar ODE `dP/dz = -α₂P²/Aeff - α₃P³/Aeff²` (valid with dispersion/Kerr/FCA/FCR disabled) and is compared against `SemiconductorMedium`'s full propagation output to `rtol=1e-5`, for TPA-only, 3PA-only, and combined cases.
- **External adversarial validation for TPA** (`test/generate_reference_data.py`, `test/test_adversarial.jl`, Scenario 7): TPA is now cross-checked against `gnlse-python`'s independently-implemented solver (`scipy.solve_ivp`) using the standard "complex γ = Kerr + i·TPA" trick (`γ = γ_r + i·α₂/(2·Aeff)`), which its Kerr-only nonlinear step accepts without modification. 21 peak-power checkpoints (1% tolerance) plus a full-field cross-correlation (≥0.999) at the waveguide output.
- **Three new documentation examples**, replacing the single combined TPA/FCR example:
  - [Example 8 — Silicon TPA Optical Limiter](docs/src/examples/ex8_silicon_tpa.md): power sweep showing the classic TPA saturable-limiter curve.
  - [Example 11 — Mid-IR Three-Photon Absorption](docs/src/examples/ex11_silicon_3pa.md): TPA (∝P) vs 3PA (∝P²) fractional-loss power-law scaling, the experimental fingerprint distinguishing the two channels.
  - [Example 12 — Free-Carrier Lifetime Pump-Probe](docs/src/examples/ex12_freecarrier_decay.md): reconstructs a standard pump-probe carrier-lifetime measurement, recovering the input `τ_c` from a delay sweep.
- **"Known Limitations" section** added to the `SemiconductorMedium` docstring and the semiconductor guide, explicitly documenting: the shared-`Aeff` approximation across nonlinear orders, linear-only carrier recombination (no Auger/diffusion), the omitted 3PA-associated nonlinear-refraction term, the self-steepening scope, and per-channel validation status (TPA: external + internal; 3PA/FCA/FCR: internal ODE ground-truth only — no suitable external open package was found; PyNLO, MEEP, and Tidy3D were evaluated and ruled out).

### Changed

- `README.md`, `docs/src/index.md`, `docs/src/physics.md`, `docs/src/guide/semiconductor.md`, `docs/src/examples/index.md`: updated to mention 3PA alongside TPA and reference the new examples.
- `docs/make.jl`: navigation updated for the two new example pages (11, 12) and the renamed Example 8.

### Fixed

- Silicon photonics docs and examples previously implied TPA/FCA/FCR were validated the same way as the core Kerr/Raman/dispersion engine (against `gnlse-python`); this is now accurately scoped per nonlinear channel.
