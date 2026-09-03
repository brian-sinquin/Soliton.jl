# AD ecosystem review: what other packages do differently

A comparison of Soliton.jl's AD implementation against established
differentiable-programming packages, and what is worth borrowing. Companion to
[Automatic Differentiation & Adjoint Propagation](adjoint_ad.md), which records
*how* the current implementation got here; this page records what the rest of
the ecosystem has settled on and where we diverge from it.

Findings are ordered by (value ÷ effort), not by interest.

## Round 1 status

| # | Finding | Status |
|---|---------|--------|
| 1 | `AbstractFFTs` already ships an adjoint-plan type (`p'`) | **Done** — used as a test oracle |
| 2 | Reverse-rule tape scratch raises peak memory | **Done** — reverted to transient |
| 3 | Shadows are rebuilt every optimizer iteration | Open |
| 4 | Hand-rolled finite differences in tests and examples | Open |
| 5 | Hand-rolled Adam, four times over | Open |
| 6 | No forward-mode rule | **Done** — `EnzymeRules.forward` added |
| 7 | Parameters and workspace live in one struct | Open — 0.3 |
| 8 | AD backend is hard-wired to Enzyme | Open — see round 2 |
| 9 | No checkpointing — tape grows linearly with steps | Open — **now top priority** |
| 10 | No continuous/analytic adjoint | Open — **now discouraged**, see round 2 |

---

## 1. `AbstractFFTs` already has an adjoint plan

`AbstractFFTs` defines `Base.adjoint(p::Plan)` returning an `AdjointPlan`, and
its own ChainRules extension uses exactly that for the pullback of `*(P, x)`:

```julia
Base.adjoint(p::Plan{T}) where {T} = AdjointPlan{eltype(inv(p)), typeof(p)}(p)
Base.adjoint(p::ScaledPlan) = ScaledPlan(p.p', p.scale)
Base.:*(p::AdjointPlan, x::AbstractArray) = adjoint_mul(p.p, x)

function adjoint_mul(p::Plan{T}, x::AbstractArray, ::FFTAdjointStyle) where {T}
    dims = fftdims(p)
    N = normalization(T, size(p), dims)
    pinv = inv(p)
    return (inv(N) * pinv) * x
end
```

Two things follow.

**Our derivation is confirmed, independently.** Upstream derives the
normalization from `fftdims(p)` and `size(p)` — the transformed dimensions, not
the array length. That is precisely the correction made in
`SolitonEnzymeExt._adjoint_plan`, arrived at separately.

**We cannot simply delegate to it.** `AdjointPlan` does not support
`LinearAlgebra.mul!` — `adjoint_mul` builds a fresh `ScaledPlan` and returns a
newly allocated array. Adopting `p'` wholesale would reintroduce the allocations
removed earlier. Our factored `(padj, scale)` form is the `mul!`-able variant of
the same math.

*Resolved:* the test suite now asserts `scale * (padj * y) ≈ p' * y` for all four
plan shapes we build, making upstream a maintained oracle for our normalization,
and the complex-to-complex assumption is documented as a warning admonition
rather than left implicit. Real-input plans (`plan_rfft`) need boundary-aware
Nyquist scaling, which upstream's `RFFTAdjointStyle` handles and our fast path
does not — adding them means deferring to `p'` for those styles.

## 2. The tape scratch traded peak memory for allocation count

Moving the reverse pass's scratch buffer into the tape removed one allocation per
adjoint, but a tape entry is retained from `augmented_primal` until its matching
`reverse` — so it held one array *per taped `mul!` call*, all live at once, where
the previous code allocated transiently (peak: one array).

Invisible at test sizes (90 steps × 2 `mul!` × 32 KiB at `N = 2¹¹` is under
6 MB); at `N = 2¹⁴` over 10,000 steps it is ~5 GB of extra live tape. Since
finding 9 identifies tape memory as the binding constraint, that was the wrong
side of the trade.

*Resolved:* `reverse` allocates transient scratch again. The genuinely free wins
— the fused scale-and-accumulate (one fewer pass over `N`) and the eliminated
per-call `ScaledPlan` wrapper — are kept.

## 3. Rebuild shadows once, not every iteration

Every example does this inside its optimization loop:

```julia
for k in 1:n_iters
    shadow = Enzyme.make_zero(model)      # full deepcopy of PhysicsModel, per iteration
    Enzyme.autodiff(..., Enzyme.Duplicated(model, shadow), ...)
end
```

`make_zero` deep-copies the entire structure. Across a 300-iteration run that is
300 redundant allocations of every buffer in `PhysicsModel`, when a single shadow
zeroed in place with `Enzyme.make_zero!` would do. (Caveat: `make_zero!` is
documented to fail on some immutable structures — worth checking against
`PhysicsModel` specifically rather than assuming.)

This is a special case of the principle
[DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/)
builds its API around: a `prepare_gradient` step produces reusable data
structures depending only on the *type and size* of the input, so one-time costs
amortize across calls. Our shadow is exactly such a structure.

## 4. Stop hand-rolling finite differences

Both the test suite and every example compute gradient checks with a hand-written
central difference at a hand-picked step size. This project spent several CI
cycles on a gradient "mismatch" that turned out to be FD noise — the FD estimate
was self-inconsistent by 19–475% between two step sizes while the Enzyme gradient
was correct throughout.

[FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)'s
`central_fdm(5, 1)` uses a higher-order stencil with Richardson extrapolation and
adapts the step size, giving an estimate accurate to near machine precision
instead of one that has to be eyeballed for step-size sensitivity.

## 5. Adam is a solved problem

Four examples each carry their own Adam. One copy shipped a real bug — the
`eps = 1e-8` denominator dominating a ~1e-20 gradient, so the optimizer made
bit-identical no-progress steps for 150 iterations.
[Optimisers.jl](https://github.com/FluxML/Optimisers.jl) provides `Adam` with a
`setup`/`update!` interface over arbitrary nested structures. Replacing four
copies with one dependency removes the code *and* the class of bug.

## 6. Forward mode

`SolitonEnzymeExt` originally implemented only `augmented_primal`/`reverse`, so
`Enzyme.Forward` failed at the FFT boundary and reverse mode was the only option
even for single-parameter designs.

This matters because reverse mode is the wrong tool for most of our own examples.
`ad_ssfm_enzyme_compression.jl` optimizes **one scalar** (β₂) through a 60-step
solver: reverse mode tapes the entire propagation to get a single derivative,
where forward mode carries one tangent at O(1) memory. For `n_params = 1` forward
mode is strictly better on both axes.

*Resolved,* and now exercised by a real design problem:
`.github/scripts/ad_soliton_compression_optimum.jl` locates a soliton
compressor's optimal pump power through the full solver in forward mode, and
agrees with the published optimum on the soliton order to 0.13 %.

`EnzymeRules.forward` is implemented. For a linear operator the rule
is nearly trivial — the tangent obeys the same map as the primal
(`mul!(y.dval, plan.val, x.dval)`), with a `Const` `x` zeroing the output tangent
rather than leaving it stale. `AbstractFFTs` ships `frule`s alongside its
`rrule`s for the same reason.

## 7. Separate parameters from workspace

`PhysicsModel` holds three different kinds of thing in one struct: transform
plans (not differentiable), physics parameters (the things one wants gradients
of), and scratch buffers (pure workspace).

Essentially every mature differentiable package separates these.
[Lux.jl](https://lux.csail.mit.edu/) makes it explicit and total — a model is
`(model, ps, st)`, with parameters passed separately precisely so AD has one
clean object to differentiate. SciML's solvers take `(u, p, t)` with all
differentiable quantities in `p`.

Three current problems are downstream of not doing this:

- **`gamma` is not differentiable** — an immutable scalar field cannot be mutated
  in place the way `D` is, and rebuilding the model inside the differentiated
  closure trips `EnzymeRuntimeActivityError`.
- **Shadows are larger than needed** — `make_zero(model)` shadows the plans and
  `aux_data` too.
- **The `Duplicated(model, ...)` rule is hard to explain** — it exists only
  because workspace buffers share a struct with parameters.

Breaking change; scope for 0.3.

## 8. The AD backend is hard-wired

Differentiating anything today means writing `Enzyme.autodiff` with correct
activity annotations by hand and knowing the
`Duplicated(model, make_zero(model))` rule.
[ADTypes.jl](https://github.com/SciML/ADTypes.jl) and DifferentiationInterface
have become the ecosystem's answer — users write `AutoEnzyme()` and the package
handles the rest. Round 2 sharpens what this can and cannot buy us.

## 9. Tape memory is the real ceiling

Reverse-mode AD through an explicit time-stepping loop stores every intermediate,
so memory grows linearly with step count. At `N = 2¹⁴` a single field array is
256 KiB, and a fixed-step SSFM keeps on the order of half a dozen live per step —
so a 10,000-step run, an ordinary supercontinuum simulation for this package,
tapes on the order of 10 GB. Our tests and examples run 1–90 steps at
`N = 2⁶`–`2¹¹` and never approach this, which is why it has not surfaced.

[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)
(Argonne) implements Revolve/binomial checkpointing — provably optimal schedules
from Griewank & Walther's `revolve` (TOMS 2000) — plus periodic and online
schemes, designed specifically for AD of time-stepping loops. Instead of storing
all `n` steps it stores `O(log n)` checkpoints and recomputes the rest.

Our fixed-step SSFM loop is close to the ideal case: uniform steps, a small
well-defined state (`U_mid`, `z`), no adaptivity.

## 10. Continuous adjoints

SciMLSensitivity's menu is the reference taxonomy:

- **`BacksolveAdjoint`** — reconstruct the forward solution by integrating
  backwards. Least memory of any option, but documented as unstable on stiff
  problems and needing checkpoints to stay accurate.
- **`InterpolatingAdjoint(checkpointing=true)`** — interpolate the stored forward
  solution; memory falls to one interpolation interval, at the cost of roughly
  one extra forward pass.
- **`GaussAdjoint`** — generally preferred upstream; also supports checkpointing.

The domain-specific opportunity is that the adjoint of the NLSE is itself an
NLSE-like equation, integrable backwards with the *same* SSFM machinery — the
classic adjoint-method trick from photonic inverse design, where the gradient
costs one extra solve regardless of parameter count.

**Round 2 revises this downward.** See below.

---

# Round 2: the wider AD landscape

Round 1 stayed inside the packages closest to our own problem. This round looks
at the broader set of famous AD-capable frameworks — Julia and otherwise — and
mostly serves to *re-rank* round 1 rather than add to it.

## A. Diffrax settles the checkpointing-vs-continuous-adjoint question

[Diffrax](https://docs.kidger.site/diffrax/api/adjoints/) (JAX) is the most
directly comparable framework to what Soliton.jl would become: a differentiable
ODE/SDE solver library whose users routinely differentiate through thousands of
steps. Its verdict is unusually blunt.

Its **default** is `RecursiveCheckpointAdjoint` — plain autodiff through the
solver, made tractable by checkpointing (discretise-then-optimise). Its docs
state that this default "is usually a better choice than
`diffrax.BacksolveAdjoint`", and that because checkpointing already gives low
memory usage while `BacksolveAdjoint`'s "computed gradients will also be
approximate", checkpointing "is essentially always preferred in practice".

This directly re-ranks round 1:

- **Finding 9 (checkpointing) is the single highest-value structural item**, not
  merely one option among two. It is what an established, mature library in the
  same problem class chose as its default.
- **Finding 10 (continuous adjoint) should be demoted, not pursued.** The
  approximate-gradient failure mode is exactly what a lossy or Raman-damped
  backward reconstruction would suffer, and we would be re-deriving a method the
  reference implementation advises against. Keep it documented as an
  understood-and-rejected option rather than a roadmap item.

That is a genuinely useful outcome: the most expensive item on the round-1 list
(XL effort, research-scale) can be struck, and the effort redirected to the item
below it.

## B. Enzyme and Mooncake rules are mutually incompatible

Julia's reverse-AD landscape now has two serious source-transformation backends:
Enzyme.jl and [Mooncake.jl](https://chalk-lab.github.io/Mooncake.jl/). They
**rolled out their own rule designs, which are not mutually compatible** —
Mooncake handles a large subset of Julia out of the box but its rule system is
less expressive than Enzyme's.

The consequence for us is worth stating plainly, because it looks like a
limitation and is actually a justified commitment:

- Our FFTW rule is written as `EnzymeRules`. It therefore works with Enzyme and
  nothing else. Rewriting it as a ChainRules `rrule` would not help, because
  ChainRules consumers (Zygote, ReverseDiff) cannot differentiate our mutating,
  buffer-reusing solver *at all* — the very design that makes the solver fast is
  what rules them out. `adjoint_ad.md`'s blocker 4 already established this.
- So finding 8 (ADTypes/DifferentiationInterface) buys **ergonomics, not
  portability**. A `Soliton.gradient(loss, params; backend=AutoEnzyme())` façade
  is worth building for the API — users should not have to know the
  `Duplicated(model, make_zero(model))` rule — but nobody should expect
  `AutoZygote()` to start working. The honest framing is "one supported backend,
  behind a standard interface", not "backend-agnostic".
- [Reactant.jl](https://enzymead.github.io/Reactant.jl/) is the one path that
  could change the performance picture: it is a compilation system running atop
  Enzyme, and SciMLSensitivity already exposes a `ReactantVJP`. Worth watching
  for a GPU/XLA story, not worth adopting now.

## C. Photonics inverse design converged on an API shape

[Meep](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/)'s
adjoint solver and [ceviche](https://github.com/google/ceviche-challenges) are
the reference points for gradient-based photonic design, and they agree on the
user-facing shape:

- The user declares a **design region** (Meep's `MaterialGrid`) and an
  **objective function** of physically meaningful outputs — mode coefficients /
  S-parameters, DFT fields, LDOS.
- The framework runs a forward solve, then an adjoint solve, and hands back the
  gradient with respect to the design variables.
- The optimization library is wrapped *around* that gradient, not baked in.

Our examples currently make the user assemble all of this by hand: build the
model, write the loss, get the activity annotations right, hand-roll Adam.
Findings 5, 7 and 8 together are what would close that gap, and the target to aim
at is this API shape — "here is my design variable, here is my objective" —
rather than a thin wrapper over `autodiff`. Worth noting that ceviche uses plain
autograd rather than anything exotic: the value is in the framing, not the AD
machinery.

## D. What round 2 does not change

- Findings 1, 2 and 6 were the right things to do first and are done.
- Findings 3, 4, 5 remain small, independent, and worth doing whenever convenient.
- Finding 7 remains the deepest structural difference and a 0.3 item.

## Revised order

1. **Finding 9, checkpointing** — promoted to the top structural item on
   Diffrax's evidence. It is what decides whether AD here works on real
   simulations rather than examples.
2. Findings 3, 4, 5 — small, independent, do whenever convenient.
3. Findings 7 + 8 together, as a 0.3 API, aiming at the design-region/objective
   shape from section C rather than a thin `autodiff` wrapper.
4. **Finding 10 — strike.** Document as considered and rejected; the reference
   implementation in this problem class advises against it.

## Sources

- [AbstractFFTs.jl `definitions.jl`](https://github.com/JuliaMath/AbstractFFTs.jl/blob/master/src/definitions.jl)
- [Diffrax — Adjoints](https://docs.kidger.site/diffrax/api/adjoints/) and [FAQ](https://docs.kidger.site/diffrax/further_details/faq/)
- [SciMLSensitivity.jl — sensitivity algorithm selection](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/)
- [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)
- [DifferentiationInterface.jl — backends](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/dev/explanation/backends/)
- [Reactant.jl — automatic differentiation](https://enzymead.github.io/Reactant.jl/stable/tutorials/automatic-differentiation)
- [Meep adjoint solver tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/)
- [ceviche-challenges](https://github.com/google/ceviche-challenges)
- [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl), [Optimisers.jl](https://github.com/FluxML/Optimisers.jl), [Lux.jl](https://lux.csail.mit.edu/)
