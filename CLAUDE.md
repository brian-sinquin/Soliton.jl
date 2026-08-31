# Soliton.jl — working notes

GNLSE solver for optical pulse propagation in nonlinear media (Julia ≥1.10).
General contributor workflow is in `CONTRIBUTING.md`; this file holds the
context that is expensive to rediscover, and is currently weighted toward the
**automatic-differentiation (AD)** work, which is the active line of development.

## Orientation

| Question | Read |
|---|---|
| What is differentiable today, and why the rest isn't | `docs/src/dev/adjoint_ad.md` (start at "Current status") |
| What other AD packages do better, ranked by value/effort | `docs/src/dev/ad_ecosystem_review.md` |
| Working, validated AD call sites | `test/test_enzyme.jl` |
| The FFTW adjoint rule itself | `ext/SolitonEnzymeExt.jl` |
| End-to-end design problems built on it | `.github/scripts/ad_*.jl` |

Those two docs are the source of truth. Do not restate them here — extend them.

## Environment realities

**There is no Julia toolchain in this container, and there is no way to get
one:** `julialang-s3.julialang.org` is blocked by the egress proxy (403 on
CONNECT). Every claim about numerical behaviour must come from CI logs, not
from local execution or reasoning. This is the single biggest constraint on how
AD work proceeds here — budget for it rather than rediscovering it.

Practical consequence: write the change, push, read the real job log, quote real
numbers. A green checkmark is not evidence on its own — check the assertion
count actually went up, since a testset that silently fails to run still passes.

Relevant CI (`.github/workflows/`):

- `CI.yml` → `test` (matrix 1.10/1.12/pre × x64/x86), `enzyme`
  ("Enzyme adjoint-AD extension", Julia 1.12/x64 only — Enzyme's artifact is
  large and its x86/nightly support is comparatively untested), `docs`.
- `example-ad-dispersion.yml` → one job per `ad_*` example, on `pull_request`.
  It never commits plots back on PR runs (five parallel jobs would race);
  `workflow_dispatch` runs do.

The `enzyme` job prints `[diag]` lines with Enzyme-vs-FD numbers for every
gradient surface, and ends with a test count — currently **22**. That count is
the check that new testsets really ran.

Pushing to this branch cancels the previous run mid-flight (`concurrency:
cancel-in-progress`), so jobs showing `cancelled` after a rapid second push are
not failures — but they are also not verified. Re-check them on the newer run.

## Traps that have already cost real time

**`examples/` is git-ignored.** `/examples/` is in `.gitignore`; only
`.github/scripts/*.jl` is tracked, and that is what CI runs. The two are manual
mirrors and **do drift** (they are out of sync right now, harmlessly, by two
comment lines). Edit `.github/scripts/`; treat `examples/` as a local scratch
copy.

**`Duplicated(model, Enzyme.make_zero(model))`, always — never
`Const(model)`.** `propagate` and the nonlinear-step functions write active,
pulse-derived data through `model`'s scratch buffers (`buf_t1`/`buf_t2`/
`buf_f1`), and `_spm` *returns* `model.buf_f1` itself. A `Const` model gives
Enzyme no shadow for those buffers and the gradient silently becomes exactly
`0.0` — no error. This holds regardless of which argument is actually being
optimized, and has been hit from both directions (Const model + Duplicated
betas, and Const model + Duplicated pulse). `test/test_enzyme.jl`'s six-case
bisection ladder exists to catch a regression here.

**Build `PhysicsModel`/`Pulse` once, outside the differentiated closure, then
mutate in place.** Constructing either *inside* trips
`EnzymeRuntimeActivityError` — for the model, because `_to_device` builds both
active (`D`) and inactive (`gamma_W`/`W`) fields through one generic helper.

**`set_runtime_activity` is not a free pass.** It is correct for genuinely mixed
activity (`propagate`'s `At_out`, whose first column is Const-derived and the
rest active). It has also been observed to *silently zero* a gradient when the
real problem was a Const/Active misclassification. Never adopt it without an
independent finite-difference check on the same quantity.

**Suspect the finite differences before suspecting Enzyme.** Several CI rounds
went into a "mismatch" that was FD noise: the FD estimate disagreed with itself
by 19–475% across two step sizes while Enzyme was right throughout. Two
independent signals now exist — forward vs reverse mode agree to ~4 significant
figures on the `betas` derivative, far tighter than either matches FD — so a
disagreement at the 1e-3 level is FD, not AD. Diagnose by computing FD at two
step sizes; if they disagree with each other, the checker is the problem.
`abs()` at a numerically-zero point is a known legitimate source of this.

**Watch the physics before blaming the AD.** One long hunt ended at a missing
`grid.dt` in an energy normalization (`E = ∫|A|²dt ≈ dt·Σ|A|²`), which scaled a
whole experiment by `√dt ≈ 4e-8` and made every downstream number meaningless.
Print raw intermediates and check them against expected magnitudes early.

## Next steps

Ranked in `docs/src/dev/ad_ecosystem_review.md` ("Revised order"). In short:

1. **Checkpointing** — the one structural item that decides whether AD here
   works on real simulations. Tape memory grows linearly with step count
   (~10 GB for a 10,000-step run at `N = 2¹⁴`); tests only ever run 1–90 steps
   at `N = 2⁶`–`2¹¹`, which is why it has not bitten yet. Diffrax defaults to
   checkpointing and calls it "essentially always preferred".
2. Small and independent: reuse one `make_zero!`d shadow across optimizer
   iterations instead of rebuilding it; adopt FiniteDifferences.jl for gradient
   checks; adopt Optimisers.jl instead of four hand-rolled Adams.
3. A 0.3 API: separate parameters from workspace in `PhysicsModel` (this is
   what unblocks a `gamma` gradient), behind an ADTypes-style façade.
4. **Do not** build a continuous/backsolve adjoint — considered and rejected;
   see the review for why.

## Conventions

- Formatting: `.JuliaFormatter.toml` (blue style, margin 92). **Not** enforced
  by CI; match it anyway.
- Public API is stable at 0.2.1. The AD work so far has changed no exported
  name and no forward-simulation result — keep it that way, or say so loudly.
- Removing a redundant buffer copy is behaviour-preserving and should be
  provable: the soliton example's printed numbers were bit-identical before and
  after the solver cleanup. Use that as the bar.
