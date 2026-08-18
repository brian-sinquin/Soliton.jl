# Automatic Differentiation & Adjoint Propagation

This page assesses Soliton.jl's readiness for **adjoint-mode / reverse-mode
automatic differentiation (AD)** — the capability needed to compute gradients
of a scalar objective (e.g. "match this target spectrum", "maximize peak
power at the fiber output") with respect to simulation inputs (`γ`, `β_n`,
fiber length, loss, pulse shape, ...) for gradient-based optimization
(dispersion engineering, pulse-shaping inverse design, parameter fitting).

It is **not** a user guide to a working feature: as of this writing, Soliton.jl
does **not** support AD through `solve`/`propagate` — the FFTW adjoint rule
(roadmap step 1) is implemented and tested for the underlying `mul!`/nonlinear-
step building blocks, but no solver (`SSFM`/`AdaptiveSSFM`/`ERK4IP`) has been
differentiated end-to-end yet (roadmap steps 2–4). The sections below
document *why*, what's done, and what a full implementation would still
require, so the remaining work can be scoped and picked up deliberately
rather than discovered piecemeal.

## Summary

| AD approach | Status | Blocker |
|---|---|---|
| Forward-mode on dispersion/loss/gain math only (`propagation_constant`, `dispersion_operator`, `loss_vector`/`gain_vector`) | **Works today** | These are pure arithmetic, no FFTW involved. `Grid`, `TaylorDispersion`/`TabulatedDispersion`/`SellmeierDispersion`, and these functions' signatures are now generic over `<:Real` — see [Architecture audit](#architecture-audit), points 1–2. |
| Forward-mode through full pulse propagation (ForwardDiff.jl, `Dual` numbers) | **Blocked** | FFTW only accepts `Float32`/`Float64`/`ComplexF32`/`ComplexF64` buffers; `Dual`-typed arrays cannot be passed to `plan_fft`/`plan_ifft` at all, regardless of how generic the surrounding types are. |
| Reverse-mode via array overloading (Zygote.jl) | **Blocked** | The propagation loop is written as in-place mutation (`@.`, `mul!`, `copyto!` into pre-allocated buffers) for zero-allocation performance; Zygote does not differentiate through mutating array updates without a full rewrite to `Zygote.Buffer`/non-mutating style, which would defeat the current performance design. |
| Reverse-mode via source transformation (Enzyme.jl) | **FFTW boundary + single nonlinear step done; full solver not yet done** | Enzyme differentiates the compiled, concretely-typed, mutating code directly (no `Dual` overloading needed), so it is not affected by the FFTW or mutation blockers above. The hand-written `EnzymeRules` adjoint for the FFTW `ccall` boundary (`ext/SolitonEnzymeExt.jl`, a weak-dependency extension) is implemented and validated against finite differences on `to_freq`/`to_time` and on the full `_spm` nonlinear step; differentiating an entire `SSFM`/`ERK4IP` propagation loop is roadmap steps 2–4, not yet done. |
| Gradient-free (`solve_sweep` + Optim.jl/BlackBoxOptim.jl/surrogate optimization) | **Works today** | No code changes needed; recommended near-term path for optimizing over full propagation until adjoint support lands. |

**Recommendation:** target **Enzyme.jl** as the AD backend for adjoint
propagation, not Zygote or ForwardDiff. See [Recommended path](#recommended-path-enzymejl).

## Architecture audit

Four independent properties of the current implementation each block AD in a
different way. All four were confirmed by reading the source directly (no
running Julia session was available in this environment, so no numerical
verification was performed — see [Open work](#open-work-not-done-here)).

### 1. `PhysicsModel` was hard-typed to `Float64`/`ComplexF64` — **now relaxed**

`src/nonlinearity.jl`'s `PhysicsModel` used to constrain its buffers to the
concrete `TA <: AbstractArray{ComplexF64}`/`TVR <: AbstractVector{Float64}`.
It is now `TA <: AbstractArray{<:Complex}`/`TVR <: AbstractVector{<:Real}`,
so the container honestly reflects whatever element type the `Grid`/
`Medium`/dispersion pipeline feeding `build_physics_model` actually produces
(GPU array types, or `ComplexF32` if that whole pipeline is instantiated in
`Float32`), instead of force-converting to `ComplexF64`. `_spm`, `_spm_raman`,
`_amplifying_spm`, `_amplifying_spm_raman`, and `_semiconductor_spm` all
already worked generically on `model`'s own buffers and needed no change;
the only spot that had silently assumed `ComplexF64` on a field with this
relaxed bound (`model.RW::Vector{ComplexF64}` in the two Raman nonlinear
steps) was loosened to `model.RW::AbstractVector{<:Complex}` to match.

Two things this does **not** do: it does not make `omega0`/`dt`/`fr` (still
plain `Float64` scalar fields) generic, and it does not touch
`inject_ase_noise!` (`AmplifyingMedium`'s ASE step) or `_vectorial_spm_fwm`
(`BirefringentMedium`), which still hard-require `ComplexF64` buffers. A
`Dual`-typed template still can't construct a working `PhysicsModel` either
way — that's blocker 3 below, which this change doesn't touch.

### 2. `Grid` was unconditionally `Float64` — **now relaxed**

`create_grid` used to always materialize `Grid{Float64}(...)` regardless of
the element type of `time_window`/`wavelength`. It now computes
`T = promote_type(typeof(time_window), typeof(wavelength), Float64)` and
builds a `Grid{T}` — plain `Float64` arguments (the overwhelmingly common
case) still produce exactly `Grid{Float64}` with bit-identical values, while
AD dual numbers passed for either argument now propagate through `t`, `V`,
`W`, `dt`, `omega0`. The `DispersionModel` family (`TaylorDispersion`,
`TabulatedDispersion`, `SellmeierDispersion`) was hard-typed to `Float64` the
same way and got the same treatment — each is now `{T<:Real}` over its own
coefficient vectors, and `propagation_constant`/`dispersion_operator`/
`loss_vector!`/`gain_vector!` were relaxed from `AbstractVector{Float64}` to
`AbstractVector{<:Real}` to match. This is the part of the pipeline that
**does** work end-to-end today: none of it touches FFTW, so
`propagation_constant`/`dispersion_operator` are now differentiable
(ForwardDiff or Enzyme) with respect to `β_n`, Sellmeier `B`/`C`, tabulated
`beta`, `medium.loss`, or `medium.g0` right now, with no further work. This
is the "dispersion engineering" half of the optimization use case; getting a
gradient through actual pulse *propagation* still needs the Enzyme work
below, because of blocker 3.

### 3. FFTW cannot transform generic-typed arrays

`plan_fft`/`plan_ifft` (used throughout `build_physics_model` and every
`_spm*`/`_amplifying_spm*`/`_vectorial_spm_fwm` nonlinear-step function) call
into the FFTW C library, which only implements transforms for `Float32`,
`Float64`, and their complex counterparts. There is no generic/`Dual`
fallback wired in here (unlike, say, `BigFloat`, which has a slow generic
AbstractFFTs fallback usable by packages like `GenericFFT.jl`). This is a
**hard, library-level wall** for `Dual`-number forward-mode AD — it applies
independently of points 1–2, and relaxing the type constraints there would
not by itself unblock ForwardDiff.

### 4. Zero-allocation design relies on in-place mutation

`src/solvers/erk4ip.jl` reuses a fixed set of pre-allocated buffers
(`k1`...`k5`, `Nu`, `U_temp`, `u_temp`, `r`, `U4_fft`, `U5_fft`, `u4`,
`model.buf_t1`/`buf_t2`/`buf_f1`) across every RK stage and every step via
`@.`, `mul!`, and `copyto!`. This is exactly the style Zygote cannot
differentiate without conversion to `Zygote.Buffer` (or a non-mutating
rewrite), and exactly the style Enzyme *is* designed for (it tracks writes to
existing memory via shadow buffers rather than requiring pure functions).

### Secondary considerations (relevant once the above are solved)

- **Adaptive step control.** `ERK4IP`/`AdaptiveSSFM` accept/reject steps and
  choose step count based on a runtime-computed local error / phase bound.
  This is differentiable in principle (branching on a runtime value is fine
  for AD as long as you're not sitting exactly on a decision boundary), but
  it means the number of forward steps — and therefore the reverse-mode
  memory/checkpointing cost — is data-dependent. A `SSFM`/fixed-step solver
  is a much simpler first differentiation target than the adaptive `ERK4IP`
  default.
- **Stochastic ASE noise.** `AmplifyingMedium` propagation calls
  `inject_ase_noise!`, which draws `randn(rng)` each accepted step. Pathwise
  (reparameterization) gradients work fine here since the noise scale itself
  is a deterministic function of the state, but the `rng` must be seeded and
  fixed for a reproducible adjoint pass — this is already how the existing
  `rng` keyword works, so no design change is needed, just something to keep
  in mind when validating gradients against finite differences (use the same
  seed on both sides).

## Recommended path: Enzyme.jl

Enzyme differentiates LLVM IR of the actual compiled, concretely-typed
program — it does not need generic/`Dual`-compatible element types (so
blocker 3 above is irrelevant to it) and it natively supports mutating,
buffer-reusing code via shadow memory (so blocker 4 is exactly its intended
use case). Blockers 1 and 2 never needed relaxing for Enzyme specifically —
that only mattered for a `Dual`-overloading approach like ForwardDiff, which
is independently blocked by FFTW itself — but relaxing them anyway (done
above) is what makes the dispersion/loss/gain math differentiable today, and
doesn't cost Enzyme anything either way.

The one piece of real work is the FFTW boundary: `mul!(y, plan, x)` for an
FFTW plan is a `ccall` into a C library, which Enzyme cannot look inside of.
It needs one hand-written rule pair
(`EnzymeRules.augmented_primal`/`EnzymeRules.reverse`, or the simpler
`EnzymeRules.forward`/reverse-via-known-adjoint pattern) registered for
`LinearAlgebra.mul!(::AbstractVecOrMat, ::FFTW.FFTWPlan, ::AbstractVecOrMat)`.
This is mathematically simple to derive — the DFT is a linear operator, so
its VJP (vector-Jacobian product) is multiplication by the Hermitian adjoint
of the DFT matrix, which for FFTW's unnormalized convention is `N` times the
inverse transform (`ifft(v) = (1/N) * fft_adjoint(v)`, so
`fft_adjoint(v) = N * ifft(v)`, and symmetrically for `plan_ifft`). Once
written and tested against finite differences on a small grid, this single
rule covers every solver and every physics model in the package (they all
route through the same `to_freq`/`to_time` plans).

**Confirmed with a working Enzyme install** (a later session had Julia 1.12 +
Enzyme v0.13 available, where the original assessment above had none):

- `Enzyme.autodiff(Reverse, ...)` on the pure elementwise nonlinearity
  (`u .* abs2.(u)`, no FFT) differentiates correctly — matches blocker 4's
  prediction that mutation/buffer-reuse alone is not a problem for Enzyme.
- Isolating just `mul!(y, model.to_freq, u)` and differentiating through it
  throws `Enzyme.Compiler.EnzymeNoDerivativeError`, not a silent wrong
  answer, with a precise origin: `No augmented forward pass found for
  ejlstr$fftw_alignment_of$...libfftw3-3.dll`, from
  `FFTW.assert_applicable`'s pointer-alignment sanity check (`fft.jl`'s
  `alignment_of`) that runs *before* the actual transform ccall. So the rule
  needs to be registered at the `mul!`/`LinearAlgebra.mul!` level (as
  planned above), not by trying to patch through FFTW's internal alignment
  check.
- Running `Enzyme.autodiff` on the real `_spm(u, model, z)` (the full
  nonlinear step, FFT included) does **not** throw this error at all — by
  default it hits a *different*, earlier error
  (`EnzymeRuntimeActivityError`, because `model`'s pre-allocated buffers are
  marked `Const` while being used as scratch for active/differentiable
  values) that recommends `set_runtime_activity`. Turning that on makes the
  call "succeed" with **no error and a gradient that is silently all
  zero** — verified wrong against a finite-difference check at the pulse
  peak (finite difference: non-zero; Enzyme: exactly `0.0 + 0.0im`
  everywhere). This is a materially worse failure mode than the isolated-FFT
  case: an unregistered FFTW rule at the boundary of a larger call doesn't
  always surface as `EnzymeNoDerivativeError` — it can silently drop the
  gradient contribution through the ccall instead. **Any gradient obtained
  from Enzyme on code touching `to_freq`/`to_time` before the `mul!` rule
  below exists must be checked against finite differences before being
  trusted; a successful, error-free `autodiff` call is not evidence of a
  correct one.**

### Proposed staged roadmap

0. **Done.** Parameterize the pure-math layer (`Grid`, `TaylorDispersion`/
   `TabulatedDispersion`/`SellmeierDispersion`, `propagation_constant`,
   `dispersion_operator`, `loss_vector!`/`gain_vector!`) over `<:Real`
   instead of hard-coded `Float64`, and relax `PhysicsModel`'s buffer bounds
   from concrete `ComplexF64`/`Float64` to `<:Complex`/`<:Real`. This was
   done without a Julia toolchain available to run the test suite — see
   [Open work](#open-work-not-done-here).
1. **Done.** Added `Enzyme` as a weak dependency and
   [`ext/SolitonEnzymeExt.jl`](https://github.com/brian-sinquin/Soliton.jl/blob/master/ext/SolitonEnzymeExt.jl),
   registering the `EnzymeRules.augmented_primal`/`EnzymeRules.reverse` pair
   for `LinearAlgebra.mul!(y, plan::AbstractFFTs.Plan, x)`. The adjoint is
   `N * inv(plan)` for an unnormalized transform (`to_time`/`plan_fft`) and
   `inv(plan) / N` for an already-normalized one
   (`to_freq`/`plan_ifft`/`ScaledPlan`) — see the math derivation in the
   extension's docstring. Validated in
   [`test/test_enzyme.jl`](https://github.com/brian-sinquin/Soliton.jl/blob/master/test/test_enzyme.jl)
   against finite differences: matches to ~1e-6/1e-7 relative error on both
   `to_time` and `to_freq` in isolation, and on the full `_spm` nonlinear
   step (FFT + Kerr term) end-to-end. Confirms the `EnzymeRuntimeActivityError`
   from differentiating a full `PhysicsModel`-based step is fixed by calling
   `Enzyme.autodiff` with `Enzyme.Duplicated(model, Enzyme.make_zero(model))`
   instead of `Enzyme.Const(model)` — not by `set_runtime_activity`, which
   (as documented above) silently produces a wrong zero gradient instead.
   This runs as its own CI job (`enzyme` in `.github/workflows/CI.yml`), kept
   separate from the main test matrix rather than added to every cell, since
   Enzyme's large LLVM-based artifact and its `i686`/nightly-Julia support
   are less exercised than the package's existing dependencies; the job
   covers Julia 1.12 on `ubuntu-latest`/`x64` only.
2. Differentiate a single **fixed-step** `SSFM` propagation end-to-end
   (`SSFM`, not `ERK4IP`) for a scalar loss such as
   `sum(abs2, At_out[:, end])`, with respect to `medium.gamma` and
   `TaylorDispersion.betas`. Validate against finite differences.
3. Extend to `AdaptiveSSFM`, then `ERK4IP` (the default solver), once the
   fixed-step case is solid — these add data-dependent step counts on top of
   the same per-step machinery.
4. Expose a small convenience layer (e.g. `soliton_gradient(loss_fn, pulse,
   params)`) suitable for driving `Optimization.jl`/`Optim.jl`-style
   gradient-based optimizers — this is the actual "future optimization
   solving" use case (dispersion engineering, γ/loss/β_n fitting to a target
   spectrum, pulse-shaping inverse design).

None of this has been implemented yet; this document is the scoping/assessment
step.

## What works today, with zero code changes

Gradient-free optimization is fully usable right now: pair
[`solve_sweep`](@ref) (or a plain loop) with a black-box optimizer — Optim.jl's
Nelder-Mead, `BlackBoxOptim.jl`, CMA-ES, or Bayesian optimization all treat
`solve`/`propagate` as an opaque forward simulator and need no differentiability
at all. For small parameter counts (a handful of `β_n`, `γ`, fiber length),
this is a reasonable near-term substitute for adjoint-based optimization
while gradient support is built out.

## Open work not done here

The initial assessment and roadmap step 0 were done by reading the source
only, with no Julia toolchain available in that environment. A follow-up
session had a working Julia 1.12 toolchain and validated the two items that
were previously just claims:

- **Ran the existing test suite.** `] test` initially reported 3 failures,
  all pre-existing `@test_throws ArgumentError` checks on `TabulatedDispersion`
  (too-few-samples, unsorted `detuning`) and `SellmeierDispersion`
  (mismatched `B`/`C` length) that silently stopped throwing. Root cause:
  parameterizing these structs over `{T<:Real}` made Julia auto-generate a
  default inner constructor (`TabulatedDispersion(::Vector{T}, ::Vector{T})
  where T`) that is *more specific* than the validating outer constructor for
  same-element-type vector arguments, so same-typed calls silently dispatched
  around the validation. Fixed in `src/types.jl` by moving each check into an
  explicit `TabulatedDispersion{T}(...)`/`SellmeierDispersion{T}(...)` inner
  constructor — defining any inner constructor suppresses Julia's default
  one. `Medium`'s existing inner-constructor pattern was already immune to
  this and needed no change. All 371 tests now pass, confirming the default
  `Float64`/`ComplexF64` path is unchanged.
- **Confirmed the forward-mode claim.** `ForwardDiff.gradient` through
  `propagation_constant` succeeds for both `TaylorDispersion.betas` and
  `SellmeierDispersion.B`, producing finite non-zero gradients, on a small
  grid with no code changes beyond the constructor fix above.
- **Installed Enzyme and confirmed the exact FFTW error surface** (step 1's
  first open item above): isolating `mul!(y, model.to_freq, u)` and
  differentiating it throws `EnzymeNoDerivativeError` at
  `FFTW.assert_applicable`'s `alignment_of` ccall — not at the transform
  itself, but at a pointer-alignment check that runs first. Differentiating
  the full `_spm` nonlinear step (FFT included) instead hits a *different*
  error first (`EnzymeRuntimeActivityError`, from `model`'s buffers being
  `Const` while used as active scratch space), and — this is the important
  part — silently returns an all-zero, verifiably wrong gradient once that
  error is worked around with `set_runtime_activity` instead of a real
  `EnzymeRules` pair. See [Recommended path](#recommended-path-enzymejl) for
  the full writeup; this used a temporary scratch environment (`Pkg.develop`
  + `Pkg.add("Enzyme")` outside this package's own `Project.toml`).
- **Wrote and landed the `EnzymeRules` pair as a real weak-dependency
  extension.** `Enzyme` is now in `[weakdeps]`/`[extensions]` in
  `Project.toml`, `ext/SolitonEnzymeExt.jl` implements
  `augmented_primal`/`reverse` for `mul!(y, plan::AbstractFFTs.Plan, x)`, and
  `test/test_enzyme.jl` validates it end-to-end against finite differences
  (isolated `to_time`, isolated `to_freq`, and the full `_spm` step). Also
  confirmed the `EnzymeRuntimeActivityError`/silent-zero-gradient case is
  resolved by using `Enzyme.Duplicated(model, Enzyme.make_zero(model))`
  rather than `Enzyme.Const(model)` — this is a caller-side concern (how you
  invoke `Enzyme.autodiff`), not something the extension itself can paper
  over. Added as its own CI job (`enzyme` in `.github/workflows/CI.yml`,
  Julia 1.12/`ubuntu-latest`/`x64` only) rather than folded into the main
  test matrix, since Enzyme's LLVM-based artifact is large and its
  `i686`/nightly-Julia support is comparatively untested next to the
  package's existing dependencies.

Still not done:

- Differentiating an entire propagation loop (`SSFM`, then `AdaptiveSSFM`/
  `ERK4IP`) end-to-end for a realistic loss function — roadmap steps 2–4.
  What's validated so far is the `mul!` rule and a single nonlinear-step
  function in isolation, not a multi-step solve.
- Benchmarking the memory/performance cost of reverse-mode checkpointing
  through the adaptive step controllers.
- The convenience `soliton_gradient(loss_fn, pulse, params)` wrapper
  (roadmap step 4) that would make this usable without hand-rolling
  `Enzyme.autodiff`/`Duplicated`/`make_zero` calls.

Anyone picking this up should start by writing the `mul!` `EnzymeRules` pair
described above and re-running the `_spm`/finite-difference comparison from
this session to confirm it fixes both the isolated-FFT error and the
full-model silent-zero-gradient case.
