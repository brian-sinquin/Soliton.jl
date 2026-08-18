# Automatic Differentiation & Adjoint Propagation

This page assesses Soliton.jl's readiness for **adjoint-mode / reverse-mode
automatic differentiation (AD)** — the capability needed to compute gradients
of a scalar objective (e.g. "match this target spectrum", "maximize peak
power at the fiber output") with respect to simulation inputs (`γ`, `β_n`,
fiber length, loss, pulse shape, ...) for gradient-based optimization
(dispersion engineering, pulse-shaping inverse design, parameter fitting).

It is **not** a user guide to a working feature: as of this writing, Soliton.jl
does **not** support AD through `solve`/`propagate`. The sections below
document *why*, and what a compatible implementation would require, so the
work can be scoped and picked up deliberately rather than discovered
piecemeal.

## Summary

| AD approach | Status | Blocker |
|---|---|---|
| Forward-mode (ForwardDiff.jl, `Dual` numbers) | **Blocked** | FFTW only accepts `Float32`/`Float64`/`ComplexF32`/`ComplexF64` buffers; `Dual`-typed arrays cannot be passed to `plan_fft`/`plan_ifft` at all. |
| Reverse-mode via array overloading (Zygote.jl) | **Blocked** | The propagation loop is written as in-place mutation (`@.`, `mul!`, `copyto!` into pre-allocated buffers) for zero-allocation performance; Zygote does not differentiate through mutating array updates without a full rewrite to `Zygote.Buffer`/non-mutating style, which would defeat the current performance design. |
| Reverse-mode via source transformation (Enzyme.jl) | **Feasible, not yet implemented** | Enzyme differentiates the compiled, concretely-typed, mutating code directly (no `Dual` overloading needed), so it is not affected by the FFTW or mutation blockers above. It *does* need a hand-written adjoint rule for the FFTW `ccall` boundary, since Enzyme cannot see through opaque C calls automatically. |
| Gradient-free (`solve_sweep` + Optim.jl/BlackBoxOptim.jl/surrogate optimization) | **Works today** | No code changes needed; recommended near-term path for optimization until adjoint support lands. |

**Recommendation:** target **Enzyme.jl** as the AD backend for adjoint
propagation, not Zygote or ForwardDiff. See [Recommended path](#recommended-path-enzymejl).

## Architecture audit

Four independent properties of the current implementation each block AD in a
different way. All four were confirmed by reading the source directly (no
running Julia session was available in this environment, so no numerical
verification was performed — see [Open work](#open-work-not-done-here)).

### 1. `PhysicsModel` is hard-typed to `Float64`/`ComplexF64`

`src/nonlinearity.jl`:

```julia
struct PhysicsModel{
    TF, TT, NL, TG, TA <: AbstractArray{ComplexF64}, TVR <: AbstractVector{Float64}, TRW
}
    ...
    D::TA
    ...
    buf_t1::TA
    buf_t2::TA
    buf_f1::TA
    ...
end
```

`TA` is constrained to `AbstractArray{ComplexF64}` and `TVR` to
`AbstractVector{Float64}` — concrete element types, not `<:Complex`/`<:Real`.
Even though `_to_device`/`similar(template, eltype(host_array), ...)` is
written generically (with GPU arrays in mind), the default `template` is
`zeros(ComplexF64, grid.N)`, and the struct's type parameters reject any
other element type outright. A `Dual`-typed template would fail to construct
a `PhysicsModel` at all.

### 2. `Grid` is unconditionally `Float64`

`src/grid.jl`:

```julia
function create_grid(resolution::Int, time_window::Real, wavelength::Real)
    ...
    Grid{Float64}(N, t, V, W, dt, omega0, wavelength)
end
```

Regardless of the element type of `time_window`/`wavelength` passed in, the
grid is always materialized as `Grid{Float64}`. Differentiating with respect
to grid-defining parameters (e.g. center wavelength) would need this
relaxed to `Grid{T}` — though per point 3 below, this alone would not be
sufficient for forward-mode.

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
use case). This means blockers 1 and 2 (the hard-coded `Float64`/`ComplexF64`
type parameters) also do not need to be relaxed for Enzyme to work — they
only matter for a `Dual`-overloading approach like ForwardDiff, which is
independently blocked by FFTW itself.

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

### Proposed staged roadmap

1. Add `Enzyme` as a `test`-only (or weak) dependency; write and validate the
   `EnzymeRules` pair for `mul!` with `plan_fft`/`plan_ifft`, checked against
   finite differences on a tiny grid (`N=32`).
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

This assessment is based on static code reading only — no Julia toolchain
was available in the environment it was written in, so none of the following
were actually run:

- Installing Enzyme and confirming which FFTW calls it errors on today
  (`Enzyme.autodiff` on `_propagate_erk4ip!` or `_spm`/`_spm_raman` directly,
  to get the exact "unsupported ccall" error/stack rather than inferring it).
- Writing and testing the proposed `EnzymeRules` pair for `mul!` with an FFTW
  plan.
- Benchmarking the memory/performance cost of reverse-mode checkpointing
  through the adaptive step controllers.

Anyone picking this up should start by reproducing step 1 of the roadmap
above on a machine with a working Julia + Enzyme install, since the exact
Enzyme error surface is the fastest way to confirm (or correct) the analysis
here.
