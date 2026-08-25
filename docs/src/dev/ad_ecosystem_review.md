# AD ecosystem review: what other packages do differently

A comparison of Soliton.jl's AD implementation against established
differentiable-programming packages, and what is worth borrowing. Companion to
[Automatic Differentiation & Adjoint Propagation](adjoint_ad.md), which records
*how* the current implementation got here; this page records what the rest of
the ecosystem has settled on and where we diverge from it.

Findings are ordered by (value ÷ effort), not by interest.

## Summary

| # | Finding | Effort | Payoff |
|---|---------|--------|--------|
| 1 | `AbstractFFTs` already ships an adjoint-plan type (`p'`) | XS | Correctness oracle for free |
| 2 | Reverse-rule tape scratch raises peak memory | XS | Undoes a regression we introduced |
| 3 | Shadows are rebuilt every optimizer iteration | S | ~150–300 wasted model allocations/run |
| 4 | Hand-rolled finite differences in tests and examples | S | Removes a whole class of false alarms |
| 5 | Hand-rolled Adam, four times over | S | Deletes code that already shipped a bug |
| 6 | No forward-mode rule | M | Strictly cheaper for few-parameter designs |
| 7 | Parameters and workspace live in one struct | M | Unblocks `gamma`; explains `Duplicated` |
| 8 | AD backend is hard-wired to Enzyme | M | Roadmap step 4 |
| 9 | No checkpointing — tape grows linearly with steps | L | The actual scaling ceiling |
| 10 | No continuous/analytic adjoint | XL | O(1) memory in step count |

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
`SolitonEnzymeExt._adjoint_plan`, arrived at separately. The agreement is worth
having as a test rather than a comment: `p'` is a free, upstream-maintained
oracle for our factorization, and a cross-check against it would catch a
normalization regression without us having to re-derive anything.

**We cannot simply delegate to it.** `AdjointPlan` does not support
`LinearAlgebra.mul!` — `adjoint_mul` builds a fresh `ScaledPlan` and returns a
newly allocated array. Adopting `p'` wholesale would reintroduce exactly the
allocations removed in the last pass. Our factored `(padj, scale)` form is the
`mul!`-able variant of the same math, and should stay.

Worth recording: upstream handles `RFFTAdjointStyle`/`IRFFTAdjointStyle` with
boundary-aware scaling at the Nyquist bin. Our rule assumes `ℂⁿ → ℂⁿ` and would
be silently wrong for a real-input plan. Soliton never builds one today, but the
assumption should be asserted rather than assumed.

## 2. The tape scratch trades peak memory for allocation count

This one is a correction to a change made in the previous cleanup pass.

Moving the reverse pass's scratch buffer into the tape removed one allocation
per adjoint. But a tape entry is retained from `augmented_primal` until its
matching `reverse` — so where the old code allocated transiently inside
`reverse` (peak cost: one array at a time), the new code holds one array *per
taped `mul!` call*, all live simultaneously.

At the sizes in our tests and examples this is invisible: 90 steps × 2 `mul!` ×
32 KiB at `N = 2¹¹` is under 6 MB. At the sizes this package is actually for it
is not: at `N = 2¹⁴` each field array is 256 KiB, so a 10,000-step run holds
~20,000 scratch arrays ≈ 5 GB of tape *that the previous code did not*.

Since finding 9 identifies tape memory as the binding constraint on
differentiating realistic runs, this is the wrong side of the trade. The fused
scale-and-accumulate (one fewer pass over `N`) and the eliminated `ScaledPlan`
wrapper are worth keeping; the tape scratch should go back to being a transient
allocation in `reverse`.

## 3. Rebuild shadows once, not every iteration

Every example does this inside its optimization loop:

```julia
for k in 1:n_iters
    shadow = Enzyme.make_zero(model)      # full deepcopy of PhysicsModel, per iteration
    Enzyme.autodiff(..., Enzyme.Duplicated(model, shadow), ...)
end
```

`make_zero` deep-copies the entire structure. Across a 300-iteration run that is
300 redundant allocations of every buffer in `PhysicsModel`, when a single
shadow zeroed in place with `Enzyme.make_zero!` would do. (Caveat:
`make_zero!` is documented to fail on some immutable structures — worth checking
against `PhysicsModel` specifically rather than assuming.)

This is a special case of the principle
[DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/)
builds its whole API around: a `prepare_gradient` step produces reusable data
structures that depend only on the *type and size* of the input, not its values,
so one-time costs are amortized across calls. Our shadow is exactly such a
structure. Whatever else we adopt, the "prepare once, reuse across iterations"
split is the right shape for a gradient API here.

## 4. Stop hand-rolling finite differences

Both the test suite and every example compute gradient checks with a hand-written
two-point or central difference at a hand-picked step size. This session spent
several CI cycles on a gradient "mismatch" that turned out to be FD noise — the
FD estimate was self-inconsistent by 19–475% between two step sizes, while the
Enzyme gradient was correct throughout.

[FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl) exists
for this: `central_fdm(5, 1)` uses a higher-order stencil with Richardson
extrapolation and adapts the step size, giving an estimate accurate to near
machine precision instead of one that has to be eyeballed for step-size
sensitivity. Adopting it would have collapsed that entire debugging arc into a
single unambiguous number, and it removes the standing need to sanity-check the
checker.

`ChainRulesTestUtils.test_rrule` is the same idea packaged for rule authors,
though it targets ChainRules rather than EnzymeRules, so it fits finding 8 better
than this one.

## 5. Adam is a solved problem

Four examples each carry their own Adam:

```julia
mvec .= adam_b1 .* mvec .+ (1 - adam_b1) .* dtheta
vvec .= adam_b2 .* vvec .+ (1 - adam_b2) .* dtheta .^ 2
m_hat = mvec ./ (1 - adam_b1^k)
...
```

One of those copies shipped a real bug — the `eps = 1e-8` denominator dominating
a ~1e-20 gradient, so the optimizer made bit-identical no-progress steps for 150
iterations, costing a full CI round to diagnose.
[Optimisers.jl](https://github.com/FluxML/Optimisers.jl) provides `Adam` (and the
rest) with a `setup`/`update!` interface that works on arbitrary nested
structures. Replacing four copies with one dependency removes the code *and* the
class of bug.

## 6. There is no forward-mode rule

`SolitonEnzymeExt` implements only `EnzymeRules.augmented_primal`/`reverse`.
`Enzyme.Forward` through any Soliton solver therefore fails at the FFT boundary,
and the `ForwardDiff`-based examples can only touch `propagation_constant`, never
the solver.

This matters more than it looks, because reverse mode is the wrong tool for most
of our own examples. `ad_ssfm_enzyme_compression.jl` optimizes **one scalar**
(β₂) through a 60-step solver. Reverse mode tapes the entire propagation to get
a single derivative; forward mode would carry one tangent alongside the primal at
O(1) memory and roughly one extra function evaluation. For `n_params = 1`,
forward mode is strictly better on both axes.

For a linear operator the forward rule is nearly trivial — the tangent obeys the
same map as the primal:

```julia
mul!(y.val,  plan.val, x.val)
mul!(y.dval, plan.val, x.dval)   # plan is linear
```

`AbstractFFTs` ships `frule`s alongside its `rrule`s for exactly this reason.

## 7. Separate parameters from workspace

`PhysicsModel` currently holds three different kinds of thing in one struct:

- transform plans (`to_freq`, `to_time`) — not differentiable, not mutable
- physics parameters (`D`, `gamma`, `gamma_W`, `W`, `fr`, `RW`) — the things one
  actually wants gradients of
- scratch buffers (`buf_t1`, `buf_t2`, `buf_f1`) — pure workspace

Essentially every mature differentiable package separates these.
[Lux.jl](https://lux.csail.mit.edu/) makes it explicit and total: a model is
`(model, ps, st)` — architecture, parameters, state — with parameters passed
separately precisely so AD has one clean object to differentiate. SciML's solvers
take `(u, p, t)` with all differentiable quantities in `p`.

Three current problems are downstream of not doing this:

- **`gamma` is not differentiable.** It is an immutable scalar field, so it
  cannot be mutated in place the way `D` is, and rebuilding the model inside the
  differentiated closure trips `EnzymeRuntimeActivityError`. `adjoint_ad.md`
  already identifies "hold `gamma` in a mutable container" as the fix; a
  parameter struct is that fix, generalized.
- **Shadows are larger than they need to be.** `make_zero(model)` shadows the
  plans and `aux_data` too, not just the arrays that carry gradient.
- **The `Duplicated(model, ...)` rule is hard to explain.** It exists because
  workspace buffers are inside the same struct as parameters. With them
  separated, "duplicate the workspace, duplicate the parameters you want
  gradients of" is a rule users can derive rather than memorize.

This is the largest architectural difference between Soliton.jl and the
ecosystem, and it is a breaking change — worth scoping for 0.3, not 0.2.x.

## 8. The AD backend is hard-wired

Differentiating anything today means writing `Enzyme.autodiff` with correct
activity annotations by hand, and knowing the `Duplicated(model, make_zero(model))`
rule. That is a lot of specialist knowledge for "I want a gradient."

[ADTypes.jl](https://github.com/SciML/ADTypes.jl) and DifferentiationInterface
have become the ecosystem's answer: users write `AutoEnzyme()`, `AutoForwardDiff()`,
`AutoZygote()` and the package handles the rest. This is already roadmap step 4
in `adjoint_ad.md` ("expose a small convenience layer suitable for driving
Optimization.jl/Optim.jl-style optimizers"); the ecosystem has since standardized
what that layer should look like, so it no longer needs designing from scratch.

Pairs naturally with finding 3: the preparation object is where the reusable
shadow lives.

## 9. Tape memory is the real ceiling

Reverse-mode AD through an explicit time-stepping loop stores every intermediate,
so memory grows linearly with step count. At `N = 2¹⁴` a single field array is
256 KiB, and a fixed-step SSFM keeps on the order of half a dozen live per step —
so a 10,000-step run, which is an ordinary supercontinuum simulation for this
package, tapes on the order of 10 GB. Our tests and examples run 1–90 steps at
`N = 2⁶`–`2¹¹` and never approach this, which is exactly why it has not surfaced.

Checkpointing is the standard answer, and it is well-developed in Julia.
[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)
(Argonne) implements Revolve/binomial checkpointing — provably optimal schedules
from Griewank & Walther's `revolve` (TOMS 2000) — plus periodic and online
schemes, and is designed specifically for AD of time-stepping loops. Instead of
storing all `n` steps it stores `O(log n)` checkpoints and recomputes the rest,
trading a small constant factor of extra compute for a logarithmic memory
footprint.

Our fixed-step SSFM loop is close to the ideal case for this: uniform steps, a
small well-defined state (`U_mid`, `z`), and no adaptivity.

## 10. Continuous adjoints, for later

SciMLSensitivity's menu is the reference taxonomy for this problem, and it maps
onto ours directly:

- **`BacksolveAdjoint`** — reconstruct the forward solution by integrating
  backwards. Uses the least memory of any option, but is documented as unstable
  on stiff problems and needing checkpoints to stay accurate.
- **`InterpolatingAdjoint(checkpointing=true)`** — interpolate the stored forward
  solution; memory falls to holding one interpolation interval, at the cost of
  roughly one extra forward pass.
- **`GaussAdjoint`** — generally preferred upstream; also supports checkpointing.

The domain-specific opportunity is that the adjoint of the NLSE is itself an
NLSE-like equation, so it can in principle be integrated backwards with the *same*
SSFM machinery — the classic adjoint-method trick used throughout photonic inverse
design, where the gradient costs one additional solve regardless of parameter
count. Implemented as an `EnzymeRules` pair for `propagate` itself (rather than
letting Enzyme differentiate the loop), this would give O(1) memory in step count.

Two honest caveats before anyone starts: backward reconstruction of a
lossy or Raman-damped propagation is the unstable direction, which is the same
failure mode `BacksolveAdjoint` carries upstream; and any such rule must be
validated against the existing taped gradient, not just against finite
differences. This is a research-scale item, listed for completeness rather than
as a near-term plan.

---

## Suggested order

1. Findings 1–5 are small, independent, and individually testable. Finding 2 in
   particular reverses a regression we introduced and should go first.
2. Finding 6 (forward mode) is self-contained and immediately useful to the
   compression example.
3. Findings 7–8 belong together and imply a 0.3 API.
4. Finding 9 is the one that decides whether AD here scales past toy problems.
5. Finding 10 is research.

## Sources

- [AbstractFFTs.jl `definitions.jl`](https://github.com/JuliaMath/AbstractFFTs.jl/blob/master/src/definitions.jl) and its ChainRulesCore extension
- [SciMLSensitivity.jl — sensitivity algorithm selection](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/)
- [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)
- [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/)
- [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)
- [Enzyme.jl FAQ](https://enzymead.github.io/Enzyme.jl/stable/) and `make_zero!` ([issue #1661](https://github.com/EnzymeAD/Enzyme.jl/issues/1661))
