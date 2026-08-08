# Second-Order (χ⁽²⁾) Nonlinearity Roadmap

Tracking branch: `second-order` (based on `features`).

Goal: add second-order (χ⁽²⁾) nonlinear propagation — second-harmonic generation
(SHG), sum-/difference-frequency generation (SFG/DFG), optical parametric
amplification (OPA) — to Soliton.jl, alongside the existing χ⁽³⁾ (Kerr/Raman/TPA/3PA)
physics.

Two implementation strategies were identified (see prior research summary in this
session). We are shipping them in two phases:

- **Phase 1 (now)**: classical coupled-envelope χ⁽²⁾ model — two GNLSE-style envelopes
  (fundamental + second harmonic / signal + idler) coupled through a χ⁽²⁾ term, each
  with its own dispersion, walk-off, and a quasi-phase-matching (QPM) profile.
- **Phase 2 (later)**: unified broadband single-field model, following
  T. Voumard et al., *"Simulating supercontinua from mixed and cascaded
  nonlinearities,"* APL Photonics **8**, 036114 (2023),
  [10.1063/5.0135252](https://doi.org/10.1063/5.0135252) (open-source reference
  implementation: [pyChi](https://pychi.readthedocs.io)) — needed for genuine
  multi-octave χ⁽²⁾+χ⁽³⁾ cascaded supercontinuum, which Phase 1's two-envelope model
  cannot capture.

---

## Phase 1 — Coupled-Envelope χ⁽²⁾ (✅ implemented)

### Physics

Two envelopes on the same time/frequency grid, fundamental `A₁` (ω₀) and second
harmonic `A₂` (2ω₀), each obeying their own linear dispersion/loss plus a χ⁽²⁾
coupling term:

```math
\frac{\partial A_1}{\partial z} = i\hat{D}_1 A_1 + i\kappa(z)\, A_2 A_1^{*} e^{-i\Delta k_0 z}
```
```math
\frac{\partial A_2}{\partial z} = i\hat{D}_2 A_2 + i\kappa(z)\, A_1^{2} e^{+i\Delta k_0 z}
```

`κ` is a single, real, normalized coupling coefficient [1/(√W·m)] — using the
*same* `κ` in both equations (rather than separate `κ₁, κ₂`) makes the system
exactly power-conserving in the lossless case by construction (derived and
verified during implementation; see the `SecondOrderMedium` docstring). `Δk₀ =
β₀(2ω₀) − 2β₀(ω₀)` is supplied directly (not derivable from `dispersion_fund`/
`dispersion_sh`, which — like every dispersion model in Soliton — omit the
removable β₀ term). Periodic poling for QPM is modeled as an idealized
period-`poling_period` sign-flipping square wave on `κ(z)`.

### API (as implemented)

Mirrors `BirefringentMedium`/`VectorialPulse` (also a coupled two-field model),
via a new shared `CoupledPulse <: AbstractPulse` abstract type that both
`VectorialPulse` and `SecondOrderPulse` subtype, letting the existing generic SSFM
coupled-field propagator (`src/solvers/ssfm_vectorial.jl`) serve both without
duplication:

```julia
struct SecondOrderMedium{T<:Real} <: AbstractMedium
    length::T
    kappa::T                          # normalized coupling [1/(√W·m)], not deff/Aeff/n
    dispersion_fund::DispersionModel  # expanded around ω₀
    dispersion_sh::DispersionModel    # expanded around 2ω₀
    deltak0::T                        # phase mismatch [1/m], supplied directly
    poling_period::T                  # QPM period [m]; 0.0 = disabled
    loss_fund::T
    loss_sh::T
    lambda0::T                        # fundamental center wavelength
end
```

Open design questions, resolved:

- **Envelope representation**: `SecondOrderPulse`/`SecondOrderSolution`, dedicated
  types (not a reuse of `VectorialPulse`/`VectorialSolution` — same `N×2`
  structure, but distinct types for API clarity, since "fundamental/SH" and
  "x/y polarization" are different physical concepts). Both now share the
  `CoupledPulse` abstract type so the SSFM solver code isn't duplicated.
- **Grid/frequency reference**: the grid is defined at the fundamental's `ω₀`
  (`grid.V = ω-ω₀`); the SH branch's dispersion is evaluated on the *shifted*
  array `grid.V .- grid.omega0` (`= ω-2ω₀`), since `dispersion_sh`'s Taylor
  coefficients are relative to its own carrier. This reuses the existing
  `propagation_constant`/`dispersion_operator` machinery unchanged — no new
  dispersion infrastructure was needed.
- **Simultaneous χ⁽³⁾**: descoped from Phase 1 as planned — pure χ⁽²⁾ only,
  documented as a known limitation. Combined χ⁽²⁾+χ⁽³⁾ is Phase 2's job.
- **Self-steepening**: not applied to the χ⁽²⁾ term in Phase 1 (documented
  limitation), rather than trying to resolve a per-branch shock convention
  prematurely.
- **Coupling parametrization**: exposed as a single normalized `kappa`
  [1/(√W·m)] rather than `deff`+`Aeff`+refractive indices — sidesteps needing
  per-branch refractive-index infrastructure while keeping the coupled-mode
  physics itself rigorously derived (see Manley-Rowe verification below);
  mirrors how `gamma` is a direct lumped Kerr coefficient elsewhere in the
  package. `Aeff` was dropped from the roadmap's original sketch since it's
  already folded into `kappa`.
- **Solver support**: only `SSFM` (fixed-step), not the adaptive `ERK4IP` —
  same limitation `BirefringentMedium` already has, inherited directly since
  both dispatch through the same `CoupledPulse`-typed propagator.

### Implementation checklist

- [x] `SecondOrderMedium` type in `src/types.jl` (keyword constructor, validation)
- [x] `SecondOrderPulse`/`SecondOrderSolution` types + `CoupledPulse` shared abstract
      type (`src/types.jl`)
- [x] `_second_order_spm` nonlinear function + `build_physics_model` in
      `src/nonlinearity.jl`
- [x] Solver wiring in `src/solver.jl` (`propagate`/`solve`/pipeline-stage
      promotion) and generalizing `src/solvers/ssfm_vectorial.jl` to dispatch on
      `CoupledPulse` instead of `VectorialPulse` specifically
- [x] QPM period → `κ(z)` square-wave handling
- [x] `docs/src/guide/second_order.md` guide page + docstrings
- [x] `test/test_second_order.jl`: 31 tests (constructor validation, Manley-Rowe
      conservation, undepleted-pump `sinc²`, depleted-pump exact `tanh²`,
      `kappa=0` no-op vs. scalar solver, QPM restoring suppressed conversion)
- [x] Example: [`docs/src/examples/ex13_ppln_shg.md`](docs/src/examples/ex13_ppln_shg.md)
      — PPLN waveguide SHG: unpoled-vs-poled conversion at LN's natural phase
      mismatch, depleted-pump `tanh²` conversion curve, and `sinc²` QPM
      acceptance bandwidth, with literature-realistic parameters (Fejer et al.
      1992; Parameswaran et al. 2002)

### Testing / validation

No external package was found for adversarial cross-validation (same landscape
as the 3PA/FCA gap — `gnlse-python` has no χ⁽²⁾ support, and pyChi's unified
broadband field isn't a direct drop-in comparison for a two-envelope coupled
model without building an equivalent pyChi scenario first). Validation instead
relies on **exact closed-form physics**, which is a stronger check than an
internal-only regression test:

1. **Undepleted-pump analytic SHG**: `|A₂(L)|² = κ²P₁₀L²sinc²(Δk₀L/2)` vs.
   numerical output, swept over `Δk₀` — passes to `rtol=1e-3`.
2. **Depleted-pump exact SHG** (`Δk₀=0`): `P₂(z)/P₁(0) = tanh²(κ√P₁(0)·z)`,
   compared at 11 z-points from undepleted through >75% depletion — passes to
   `rtol=1e-3`. This is a stronger check than the undepleted limit alone since
   it validates the full nonlinear (non-perturbative) coupled dynamics, not
   just the small-signal regime.
3. **Manley-Rowe / total power conservation**: `d(P₁+P₂)/dz = 0` for a lossless
   medium, checked at multiple `Δk₀` values including nonzero — passes to
   `rtol=1e-4`.
4. **pyChi cross-check**: still a good stretch-goal follow-up, not done in this
   pass.

---

## Phase 2 — Unified Broadband Field (later)

Deferred. Notes for when this is picked up:

- Requires a spectral grid wide enough to hold ω₀ *and* 2ω₀ (and higher
  harmonics/cascaded orders) simultaneously — a materially larger change than
  Phase 1, likely a new solver mode rather than a new `Medium` subtype.
- Follows pyChi's real/analytic-field UPPE formulation: χ⁽³⁾ SPM/THG and χ⁽²⁾ SFG/DFG
  all expressed as different quadratic/cubic combinations of one field via a
  positive/negative-frequency split (their `numbasht`/`numbathg` kernels), rather
  than hand-coupled per-harmonic envelopes.
- This is what actually captures **cascaded** χ⁽²⁾:χ⁽²⁾ ≈ effective-χ⁽³⁾ physics and
  genuine multi-octave χ⁽²⁾+χ⁽³⁾ supercontinuum — Phase 1's two-envelope model
  cannot represent this.
- Needs the anti-aliasing strategy pyChi implements for multi-octave grids, and
  likely their custom adaptive 5th-order solver (or an equivalent) rather than
  reusing ERK4IP as-is.
- Re-evaluate whether pyChi itself is usable as a direct adversarial reference at
  that point (it's the authoritative independent implementation of exactly this
  physics).

## References

- T. Voumard, M. Ludwig, T. Wildi, F. Ayhan, V. Brasch, L. G. Villanueva & T. Herr,
  "Simulating supercontinua from mixed and cascaded nonlinearities," *APL Photonics*
  **8**, 036114 (2023). DOI: [10.1063/5.0135252](https://doi.org/10.1063/5.0135252)
- T. Voumard, M. Ludwig, T. Wildi & T. Herr, "Efficient Simulation of Supercontinua
  from Cubic, Quadratic and Cascaded Nonlinearities," CLEO-PR 2022,
  [CWP2F.04](https://opg.optica.org/abstract.cfm?uri=CLEOPR-2022-CWP2F_04)
- pyChi documentation: <https://pychi.readthedocs.io>
- R. W. Boyd, *Nonlinear Optics*, 4th ed., Ch. 2 (three-wave mixing, SHG, QPM)
