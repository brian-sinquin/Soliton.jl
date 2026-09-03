# Changelog

All notable changes to Soliton.jl are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
as applied to Julia packages (pre-1.0: the minor version is bumped for new
features, the patch version for fixes).

## [Unreleased]

Adds automatic-differentiation support. No exported name changed and no
forward-simulation result changed, so existing code keeps working unmodified;
the new capability is opt-in via a weak dependency.

### Added

- **Automatic differentiation through the fixed-step `SSFM` solver**, in both
  reverse and forward mode, via [Enzyme.jl](https://enzyme.mit.edu/julia/).
  Gradients are validated end-to-end against finite differences for
  `TaylorDispersion.betas` (both modes) and for the pulse envelope (reverse);
  forward and reverse additionally agree with each other to ~4 significant
  figures on the same derivative. `medium.gamma`, `AdaptiveSSFM` and `ERK4IP`
  are **not** yet supported. See
  [`docs/src/dev/adjoint_ad.md`](docs/src/dev/adjoint_ad.md).
- `SolitonEnzymeExt`, a package extension supplying the `EnzymeRules` pair for
  the FFTW `mul!(y, plan, x)` boundary that Enzyme cannot differentiate on its
  own. Loaded automatically when `Enzyme` is present; `Enzyme` is a weak
  dependency, so it costs nothing to users who do not need AD.
- `AbstractFFTs` as a direct dependency (previously reached only transitively
  through FFTW), used for the plan adjoint.
- Six worked AD design examples under `.github/scripts/`: Sellmeier inverse
  design, dispersion flattening, dispersive-wave design with nonlinear
  verification, Enzyme-through-SSFM pulse compression, recovery of the
  fundamental soliton shape from propagation invariance alone, and location of
  a soliton-effect compressor's optimal operating point. The last one is
  cross-checked against the published empirical optimum
  (`z_opt/z₀ ≈ 0.32/N + 1.1/N²`, `F_c ≈ 4.1N`) rather than only against
  itself, and uses forward mode — one free parameter, so reverse mode would
  tape 500 steps to produce a single derivative.
- Developer documentation: an AD readiness/roadmap page and a comparison
  against other differentiable-programming packages
  ([`docs/src/dev/ad_ecosystem_review.md`](docs/src/dev/ad_ecosystem_review.md)).
- CI: a dedicated `enzyme` job, and one job per AD example so a regression in
  any single example surfaces as its own named check.
- `@strict_ctor`, an internal macro that writes a struct's validating inner
  constructor from just its validation body.

### Changed

- `Grid`, `TaylorDispersion`, `TabulatedDispersion` and `SellmeierDispersion`
  are now parametric over their element type `T<:Real` instead of being
  hard-typed to `Float64`. Plain `Float64` arguments — the overwhelmingly
  common case — still produce `Float64` containers with bit-identical values;
  the change is what lets AD dual numbers propagate through the dispersion,
  loss and gain math. `propagation_constant`, `dispersion_operator`,
  `loss_vector!` and `gain_vector!` were relaxed to match.
- `PhysicsModel`'s buffer type bounds were relaxed from the concrete
  `ComplexF64`/`Float64` to `<:Complex`/`<:Real`, so the container reflects
  whatever element type the grid/medium pipeline actually produced rather than
  force-converting.
- `propagation_constant` for `TaylorDispersion` accumulates its Taylor series
  by rebinding rather than mutating in place, which gives each term an
  unambiguous activity under reverse-mode AD. Numerically identical.
- All four solvers (`SSFM`, `AdaptiveSSFM`, `ERK4IP`, vectorial `SSFM`) no
  longer stage the frequency-domain field through `model.buf_f1` before
  saving it. Both following uses only read it, so this removes one N-element
  copy per save point. Behaviour-preserving: an example's printed output is
  bit-identical before and after.

### Fixed

- Argument validation on `TabulatedDispersion` and `SellmeierDispersion` had
  silently stopped running. Making these structs parametric caused Julia to
  auto-generate a default inner constructor that is *more specific* than the
  validating outer constructor for same-element-type vector arguments, so
  same-typed calls dispatched around the checks entirely. Validation now lives
  in an explicit inner constructor, which suppresses the auto-generated one.

## [0.2.1] and earlier

See the [commit history](https://github.com/brian-sinquin/Soliton.jl/commits/master)
— releases before this changelog was introduced are not retrospectively
documented here.
