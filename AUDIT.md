# Soliton.jl research-quality audit

Date: 2026-09-06. Package version: 0.2.1. Revision: `ae7e12ed8277c0acac876c29c32d20fcb5c17c71`.

## Executive assessment

**The package has useful analytical tests and a reasonably efficient scalar ERK4IP core, but is not yet reliable across its advertised research API.** Several supported paths silently solve the wrong problem. The most urgent defects are the first-order nonlinear update in all SSFM variants, incorrect spectral ordering in scalar cascades and filters, suppressed Kerr nonlinearity in amplifying media, and frozen z-dependent loss/gain.

Passing the current suite would not resolve these findings. Many tests check construction, finite output, agreement between paths sharing code, or qualitative trends. There are genuine Gaussian broadening, soliton phase, loss, and tapered-SPM checks; these should be retained and extended, not characterized as mere smoke tests.

Severity used here:

- **Critical:** silently incorrect central numerical/physical behavior.
- **High:** incorrect supported feature, potential nontermination, or serious validation gap.
- **Medium:** robustness, performance, documentation, or scientific interpretation weakness.
- **Low:** cleanup and maintainability.

## Scope and method

Read all 18 Julia files under `src/` (including four solver implementation files), all 15 Julia files under `test/` (runner plus 14 test files), both Python reference generators, and the five vendored `cnlse` Python source files. Inspected package/test dependency declarations and CI test wiring. Reference CSVs are numerical fixtures, not additional executable tests; their generation and consumption were audited rather than treating their values as independently established truth.

Source locations below are repository-relative and refer to this revision. Findings distinguish source-derived conclusions from runtime probes. No package source, tests, configuration, or fixtures were edited. Diagnostic scripts/logs and this report were written under `/tmp`. Julia is available as 1.12.7 on Linux aarch64. Runtime results appear at the end of this report.

The test inventory classifies every named behavioral testset and its assertions; repeated assertions in parameter loops are grouped with the loop. It does not count repeated loop iterations as independent physics benchmarks. Coverage is static, method-aware inspection, not an instrumented line/branch coverage percentage.

## Phase 1 — Dead code, clarity, and correctness

### 1.1 Critical/high numerical and API findings

#### C1 — SSFM is first order despite the symmetric-method name

Locations: `src/solvers/ssfm.jl:79–83`, `src/solvers/ssfm_vectorial.jl:78–82`, `src/solvers/adaptive_ssfm.jl:73–77`; names/docs in `src/solvers.jl:45–64`.

Each nonlinear update is `U_nl = U_mid + h*N(U_mid)`. Evaluating at the midpoint in **z** after a linear half-step does not make forward Euler a second-order nonlinear integrator. The composition of exact linear half-steps with this Euler update is generally first order globally and is not a symmetric Strang integrator.

For pure Kerr with no dispersion, an individual time sample obeys `A' = i*gamma*|A|²*A`. The exact update preserves intensity. The implemented update instead gives `|A_next|² = |A|²*(1 + (h*gamma*|A|²)²)`, creating artificial energy on every step. A small step can hide this in cross-solver comparisons.

**Measured:** CW propagation with gamma=1, P=1, L=0.5 and h=0.05, 0.025, 0.0125 gives relative complex-field errors 0.0139214, 0.00697727, 0.00349164. Halving ratios are 1.99525 and 1.99828, consistent with order one, not order two.

**Action:** use the exact nonlinear phase flow where applicable; use an appropriately accurate nonlinear sub-integrator for generalized Raman/shock/gain/carrier/vector terms. Document the actual order and add step-halving tests against an independent exact solution for each variant. Do not claim generalized nonlinear terms can all be handled by the elementary Kerr exponential.

#### C2 — Scalar cascade conversion produces inconsistent time/frequency fields

Location: `src/solver.jl:158–165`; saved spectra are shifted at `src/solvers/erk4ip.jl:281–284` and corresponding SSFM saves.

`Solution.AW` is monotonic/shifted, whereas `Pulse.AW` must be FFT-natural. `Pulse(sol)` copies the final saved AW directly without `ifftshift`. Consequently `fft(pulse.AW) != pulse.At`. All scalar cascade and piping paths using this constructor inherit the error. With `save_freq=false`, the same constructor indexes the empty AW matrix and fails.

**Measured:** an otherwise identity linear propagation followed by `Pulse(sol)` yields `norm(AW-ifft(At))/norm(AW) = sqrt(2)`.

**Action:** reconstruct AW from the final At, or unshift saved AW with an explicit empty-spectrum fallback. Validate cascade equivalence against one longer homogeneous propagation using the full complex field, and verify the Pulse transform invariant. Vectorial conversion already reconstructs AW from At and does not have this particular ordering bug.

#### C3 — Filters multiply FFT-natural spectra by monotonic transfer samples

Locations: `src/elements.jl:121–133`.

Both scalar and vector filters evaluate `transfer_function` on `grid.W` but multiply `pulse.AW`, which is FFT-natural. The transfer vector must be reordered. A low-pass about the carrier currently selects the wrong bins.

**Measured:** a broad carrier-centered passband `abs(w-omega0)<1e13` applied to a centered 1 ps Gaussian retains only about `5.49e-33` of its energy, despite encompassing its meaningful spectrum.

**Action:** align transfer samples with AW; test DC and off-carrier tones with exactly known transmission and phase. The existing test only requires lower peak power, which almost complete extinction satisfies. The scalar cascade ordering error can also partially mask the filter ordering error; independent tests are essential.

#### C4 — Amplifying-medium Kerr term is smaller by an extra omega0 factor

Locations: `src/nonlinearity.jl:388–435`, especially 397–404 and 425–433; `_resolve_gamma` at 257–277.

For numeric gamma, the builder stores `gamma/omega0`. `_amplifying_spm` constructs `i*(gamma/omega0)*|u|²*u`, then multiplies its spectrum by `gamma_W/omega0`. Without shock, `gamma_W=omega0`, so the Kerr term remains `i*gamma/omega0`, whereas the passive medium correctly reconstructs `i*gamma`. The Raman amplifier variant has the same problem.

For frequency-dependent gamma, the final multiplier also scales the **gain-saturation correction** by gamma(omega)/omega0, incorrectly tying gain saturation to Kerr strength. Gain and Kerr terms should be assembled separately with their respective units and spectral dependence.

**Measured:** with g0=0 and otherwise identical media at 1550 nm, the amplifying/passive Kerr derivative norm ratio is `8.228698061258025e-16`, i.e. `1/omega0`, rather than one.

**Action:** define one clear gamma normalization contract, separate gain correction from nonlinear spectral weighting, and add a zero-gain amplifier-to-passive equivalence check plus an exact SPM phase check. Current EDFA tests mostly measure energy, which does not detect a missing Kerr phase.

#### H1 — z-dependent linear loss/gain is evaluated only at z=0

Locations: `src/dispersion.jl:255–280`; builders in `src/nonlinearity.jl:347,470,670,766–789`; all propagation loops.

The operator supports a z argument, but model construction uses its default z=0 and propagation never refreshes D. Thus two-argument loss/gain functions do not evolve along the fiber. `test/test_loss_gain.jl:43–53` merely checks completion.

**Measured:** loss `(w,z)->1+2z` dB/m over L=0.5 gives energy ratio 0.89125094 (constant initial loss). The integrated-loss prediction is `10^(-0.75/10)=0.84139514`.

Also, `src/dispersion.jl:178–190` cannot distinguish a one-argument `loss(z)` from `loss(omega)` by arity: it always calls a usable one-argument function with omega. The documented z-only form is misleading.

**Action:** introduce an explicit frequency/position callable contract and integrate position-dependent linear operators at the appropriate stages. Test `exp(-integral(alpha dz))` and `exp(integral(g dz))` with known integrals.

#### H2 — Spectral/function gain disables saturation and ASE silently

Locations: `src/nonlinearity.jl:390,411,549`.

Every `g0` that is not `Real` is replaced by 0.0 in saturation and noise calculations. The linear builder still applies the vector/function gain, so the result amplifies without the advertised gain saturation or ASE. The vector-gain test only checks energy increase.

**Action:** evaluate spectral gain consistently in both saturation and ASE, or reject unsupported gain forms explicitly. Check scalar gain against an identical constant vector/callback, including deterministic gain saturation and ensemble ASE variance.

#### H3 — Adaptive-step validation permits nontermination and incorrect control

Locations: `src/solvers.jl:31–41,72–80`; `src/solvers/erk4ip.jl:137,179–182,250–270`; `src/solvers/adaptive_ssfm.jl:51–92`.

ERK4IP does not validate `dz_init>0`; zero on an otherwise well-behaved field can repeatedly accept a zero-length step without advancing. Negative/nonfinite steps and infinite tolerances are also not properly rejected. AdaptiveSSFM validates only phi_max: dz_min/dz_max positivity/order and dz_init are unchecked. Its loop has no error-rejection or nonfinite-field escape.

The phase controller also assumes `model.gamma*omega0` is physical gamma for all models. For a z callback, `model.gamma(z)` already returns physical gamma and gets multiplied by omega0 again. For frequency-dependent models `_resolve_gamma` stores 1.0, so the controller uses omega0 instead of a meaningful spectral gamma bound. Semiconductor models store raw gamma too. These paths can force impractically many steps at the minimum bound.

The absolute `1e-14` m target threshold can skip propagation entirely for sufficiently short valid media and save unchanged states at z=0. A phase-only rule also does not bound dispersion/nonlinearity splitting error, gain saturation, or carrier effects.

**Action:** validate finite positive controls and ordered bounds, reject nonfinite states, require forward progress, expose a step/rejection limit, use scale-aware endpoint handling, and derive the control from physical coefficients for each supported model. Test pathological inputs with an external timeout so a regression cannot hang CI.

#### H4 — Odd and single-point grids are accepted too far into construction

Location: `src/grid.jl:48–76`.

Only positive resolution is required. The detuning range has N-1 entries for odd N. Measured N=3 and N=5 produce `(length(t),length(V))=(3,2)` and `(5,4)`. N=1 fails during construction rather than receiving a clear minimum-grid-size validation (on this Julia version the endpoint range raises ArgumentError; downstream `t[2]` is independently invalid).

`fftshift` and `ifftshift` are also used interchangeably for D under the implicit even-N assumption. The inclusive time grid gives `dt=time_window/(N-1)` and FFT period `N*dt`, which must be explicitly documented when comparing external grids; changing this is a convention migration, not an isolated cleanup.

**Action:** either enforce even N>=2 or implement odd-grid frequencies/shifts correctly. Add N=0,1,2,3,5 and even non-power-of-two tests, including FFT-bin tone propagation.

#### H5 — `save_freq=false` has inconsistent semantics and breaks consumers

Locations: `src/solvers/adaptive_ssfm.jl:33–37,98`; `src/solver.jl:164`; `src/analysis.jl:150–161,405–406,499`; `src/recipes.jl:36`.

AdaptiveSSFM always allocates/saves AW (measured shape `(256,2)` despite `save_freq=false`). ERK4IP and fixed SSFM honor the flag, but scalar Pulse conversion, solution coherence, tracking, and plotting assume stored spectra.

`photon_number(solution)` substitutes total spectral power divided by omega0 when AW is absent. Parseval does not turn the frequency-weighted sum `sum(|AW|²/W)` into this expression. It is a narrowband approximation and can disagree precisely when shock/Raman shifts make a photon diagnostic useful.

**Action:** define whether downstream operations reconstruct spectra or reject unavailable data; use the exact reconstructed spectrum for photon number or explicitly expose a differently named approximation. Test save modes for every solver and consumer.

#### H6 — SHG-FROG frequency axis is scaled incorrectly

Location: `src/analysis.jl:448–473`.

Squaring/gating the envelope changes the carrier to 2*omega0; it does not double the FFT detuning-bin spacing, which remains `2*pi/(N*dt)`. Returning `V_shg=2*grid.V` mislabels the computed FFT. A shifted input tone provides a direct diagnostic. The present FROG test does not inspect the frequency axis.

Also specify normalization: `ifft(signal)` is not the dimensional integral written in the docstring without N*dt scaling, and the documented negative Fourier sign conflicts with the package's inverse-FFT frequency convention. Delay shifts use nearest-neighbor sample rounding, not interpolation; quantify the resulting resolution error.

**Action:** return a consistent detuning axis (or absolute `2*omega0+V`) and document normalization/sign; test analytic Gaussian trace widths, frequency shifts, and delay symmetry.

### 1.2 Dead code and unreachable/redundant branches

| Finding | Evidence | Recommended disposition |
|---|---|---|
| Unexported, internally unused compatibility wrapper `propagate_erk4ip` | `src/solvers/erk4ip.jl:88–98`; no caller found outside its own definition/docstrings | Candidate for deprecation/removal; first check external qualified users. It is callable, not intrinsically unreachable. |
| Unused amplifier `D_base` buffer | `src/nonlinearity.jl:487`; stored in aux_data but never read | Remove duplicate allocation or implement its intended dynamic-operator purpose as part of H1. |
| `PhysicsModel.W` is redundant for propagation | Field at `nonlinearity.jl:71`; steppers/operators use gamma_W; W is built repeatedly and inspected by API tests | Not fully dead: tests and potential users inspect the returned model. Consolidate representation after defining model API. |
| Redundant nonempty check / dead `0.0` alternative | `src/analysis.jl:569`; the function already returned at 551–562 if zero_crossings was empty | Simplify the else branch. |
| Intended z-only loss/gain fallback does not implement that contract | `src/dispersion.jl:181–189`; normal one-argument callable selects `val(omega)` | Replace arity guessing with explicit semantics (H1), not just delete a branch. |
| Unused cascade enumeration index | `src/solver.jl:130` uses `(i,stage)` but never i | Iterate stages directly or include i in contextual errors. |
| Initial fixed-step `z=0.0` assignments are overwritten before their read | `ssfm.jl:65`, `ssfm_vectorial.jl:64` | Low-priority cleanup. |
| Vector/semiconductor buf_t2/buf_t1 respectively can be unused on particular operators | Common buffer layout in `nonlinearity.jl` | Path-specific workspace overhead, not globally dead fields; consider specialized workspace types. |

Other internal functions have callers or dispatch roles: `_resolve_gamma`, `eval_gamma`, `_semiconductor_gamma_at_z`, `_to_device`, `_loss_gain_eltype`, `_eval_loss_or_gain!`, `choose_nonlinear_term`, `_fwhm`, `gas_n2`, `_capillary_confinement_loss_dB_per_m`, `_propagate_erk4ip!`, and nonlinear kernels. `PhysicsModel`, `AbstractPulse`, `AbstractMedium`, `AbstractComponent`, and `@strict_ctor` participate in construction/type hierarchy. Abstract bases and framework callbacks (`Base.show`, callable objects, recipes, precompile workload) should not be deleted merely because no ordinary textual call exists.

No large disabled Julia implementation blocks were found. Repeated `# gnlse-python:` formula snippets in pulses/Raman are provenance comments, not abandoned executable Julia. They are verbose and should be checked against the actual implementation. Dead test-local values include `E_in` in `test_edfa.jl:8`, unused `t_ref` in adversarial scenario 5, and unused diagnostic axes in `test_solvers.jl:187,193`; in the latter cases, adding assertions is more useful than deleting the variables.

### 1.3 Naming, docstrings, and contracts

- **Misattached main type docstring:** the large `Medium` documentation at `types.jl:146–171` precedes `abstract type AbstractMedium`, not `struct Medium`. Medium has a keyword-method docstring later, but its principal type/field documentation is attached to the wrong binding. Audit with Julia help/doc metadata, not only text proximity.
- Most exported bindings have docstrings; the main problem is accuracy and method coverage. `FiberSpec` documents only a generic description without fields/constructor units. `apply`'s generic prose does not explain Filter ordering/absolute-frequency argument, vector variants, or attenuation semantics. `pulse_energy`/`peak_power` vector methods and callable SimParams/element interfaces need explicit method contracts. The old positional SimParams tolerance constructor is retained without a clear deprecation/support policy.
- `Pulse` docs (`types.jl:618,625–626`) say AW has units sqrt(W)*s and equals N*ifft(At), while implementation uses ifft(At) without that dimensional scaling. With this discrete convention AW has the field's units; a physical Fourier integral needs N*dt scaling. Vector AW docs repeat the units issue. Clarify shifted Solution versus FFT-natural Pulse AW prominently.
- `spectral_centroid` docs (`analysis.jl:125–126`) reverse red/blue labels: positive detuning is higher frequency/blue, negative is red. The implementation and Raman direction test use the physically consistent sign.
- `soliton_number` docs (`analysis.jl:195`) label the soliton period `Tfission`; period `pi*LD/2` and approximate perturbation-induced fission length `LD/N` are distinct distances. `track_solitons` actually finds the global maximum peak-power snapshot and global spectrum centroid; it does not identify separate solitons or establish irreversible fission.
- ERK4IP docs/comments claim three FFT pairs per accepted step (`erk4ip.jl:18,80,176`), but the current FSAL loop evaluates k2,k3,k4,k5, each with a time transform and nonlinear spectral transform for pure SPM: **four pairs**, before save-time transforms. Raman adds another pair per nonlinear evaluation. `U5_fft` is a third-order estimate, not a fifth-order solution. Rename it and `u4`/k5 descriptions to avoid confusion.
- `erk4ip.jl:114–116` says SimParams.medium has an abstract field; the actual `SimParams{S,M}` is parameterized and stores `medium::M`. `nonlinearity.jl:191–193` similarly discusses a union that is now represented by the TRW parameter. Remove stale optimization history from implementation explanations.
- `solve` docs repeatedly describe ERK4IP only despite solver dispatch. Vectorial solve supports SSFM only; default ERK4IP with a vector pulse gives missing-method behavior, not a clear compatibility error. Vectorial solve does not accept rng; cascade/sweep have no explicit RNG threading contract.
- Semiconductor model construction silently ignores `raman_model`; vector model construction warns and falls back to Kerr. Explicitly document/reject unsupported flags. `_resolve_gamma` frequency/area overloads ignore the shock flag; define whether gamma(omega) incorporates shock physics and test both states rather than relying on ambiguous double-counting conventions.
- `step_index_aeff` calls itself a single-mode model but only enforces a lower V limit; state its approximation's validity range. `MarcuseAeff` defers invalid radius/NA rejection to invocation. Hollow-core documentation calls a bare capillary dispersion formula anti-resonant while separately acknowledging missing wall resonances; tighten terminology.
- `dispersive_wave_wavelength` silently uses gamma=0 for model/callback gamma, a fixed 1e12 rad/s exclusion, an approximate cubic fallback ignoring nonlinear/higher-order terms, and pump wavelength as a no-root sentinel (`analysis.jl:525,540,551–561`). Return a status/no-root result and verify the phase-matching residual. Its final else selects the most distant negative root, unlike the nearest-positive policy.

### 1.4 Module organization and dependency structure

All source files are included into one `Soliton` module. The source comment “Include submodules” is inaccurate: there are no source submodules and no circular include graph. The ordering is broadly sensible: solver declarations -> types/elements -> grid/pulses/physics -> stepper implementations -> public solver/analysis/presets/recipes.

There are fragile implicit dependencies: `solve` refers to photon_number defined in a later include (valid Julia deferred function lookup); Random symbols reach solver kernels through earlier elements/pulses includes; physics refers to constants and types in earlier files. The `import ..build_physics_model`/other parent-relative imports in solver files are confusing in a single-module include architecture and rely on enclosing bindings. Explicitly document namespace assumptions or use same-module references.

`types.jl` (1039 lines) mixes dispersion, media, pulse/grid, Raman, nonlinearity configurations, displays, and constructor metaprogramming. `nonlinearity.jl` (833 lines) mixes modal area, four builders, Kerr/Raman, gain, noise, and carriers. `solver.jl` versus `solvers.jl` is hard to distinguish. Suggested future separation: solver configuration/interface, media definitions, field/result types, physical kernels/builders, and diagnostic routines. Avoid a cosmetic reorganization before the numerical defects are fixed.

## Phase 2 — Performance and scaling

### 2.1 What is already good

- FFTW MEASURE plans are constructed once per `build_physics_model`, not inside ERK4IP/fixed/adaptive propagation loops. Vector plans transform dimension 1 only.
- ERK4IP preallocates stage vectors, error estimates, time/frequency scratch and exponential operators (`erk4ip.jl:141–158`); the exponential is recomputed in place only when h changes. Scalar/fixed vector SSFM allocate scratch once and precompute linear exponentials.
- `_spm` and `_spm_raman` use fused broadcasts and planned `mul!`, returning a borrowed scratch buffer. Solvers copy stage outputs where lifetime requires it. Those copies are necessary for correctness, not obvious waste.
- Raman convolution is FFT-based, not a direct quadratic convolution. Semiconductor carrier recurrence is O(N), not a repeated prefix integration.
- `spectral_coherence` already uses the pair-sum identity in O(M*N), avoiding O(M²*N). Do not propose an all-pairs loop as a correctness fix without preserving this algebraic optimization.
- Output columns are contiguous N-by-saves arrays. Memory O(N*saves) is appropriate when every snapshot is requested.

### 2.2 Allocation/type hotspots

| Priority | Location | Finding and action |
|---|---|---|
| High | `nonlinearity.jl:81` | `aux_data::NamedTuple` does not preserve the concrete NamedTuple type in the struct. Amplifier/carrier/vector hot paths extract unknown fields and lose inference. Parameterize the aux type (or use specialized models/workspaces); measure inference and warmed allocation for each kernel. |
| High | `types.jl:633,952`; `grid::Grid` | Pulse/VectorialPulse erase the grid parameter. This can impair inference through analysis, construction and solver setup. Store the concrete grid type as a type parameter; distinguish this from unannotated function arguments, which Julia can specialize normally. |
| Medium | `types.jl:262,353,913–914`; `fibers.jl:94`; `types.jl:661`; `elements.jl:82` | Abstract dispersion/Raman/RNG fields cause dynamic access. Medium and SimParams otherwise already parameterize important fields; avoid inaccurately calling all media unstable. PMDElement RNG dispatch occurs per draw/apply, not once per FFT bin. |
| Medium | `solver.jl:128,241` | `Any[]` cascade and `Vector{Any}` sweep results erase result types. Heterogeneous cascades may justify an abstract/union result API; homogeneous sweeps can preserve concrete result element type. Avoid forcing heterogeneity into the per-step kernel. |
| Medium | `types.jl:199,209,223–224,290–295,382`; `dispersion.jl:155` | `Any` signatures allow unsupported inputs but do not by themselves establish type instability: constructors preserve gamma/loss concrete types. Improve validation and function barriers; inspect inferred return types before adding restrictive annotations. |
| Medium | `adaptive_ssfm.jl:70` | Allocates a fresh N-element exponential vector each step. Preallocate and broadcast in place; unlike fixed SSFM it cannot precompute one operator for all varying steps. |
| Medium | `analysis.jl:434,469` | Spectrogram/FROG call unplanned `ifft` and allocate shifted results for every delay. Preallocate FFT output and reuse one plan plus fftshift!; O(N*delays*log N) stays, but planning and temporary allocation disappear. |
| Medium | `analysis.jl:38,130–131,147,161,496–501` | Vector peak power slices/copies two columns; centroid/photon/track routines allocate intensities and products. Use views, mapreduce or fused scalar reductions where semantics allow. `_fwhm`/bandwidth allocate findall even though only outer crossings are needed. |
| Medium | `dispersion.jl:42–45` | Taylor calculation allocates new B for each coefficient: O(K*N) temporary traffic. Comments explicitly justify rebinding for Enzyme activity. Preserve AD correctness; consider a separate fast in-place path or AD-compatible Horner evaluation rather than blindly changing to mutation. `factorial(n)` also overflows for sufficiently high n (e.g. n>=21 for Int64). |
| Medium | `fibers.jl:317` | Confinement-loss callback constructs FusedSilica and coefficient arrays on each frequency evaluation. Hoist fixed coefficients/model out of the per-bin callback. |
| Low/medium | builders in `nonlinearity.jl` | Several host/device copies and duplicate W/gamma_W/D_base arrays are made per solve, even on CPU. Plans are rebuilt across every sweep trial/stage, although grids often repeat. Consider reusable setup/workspace API with isolated mutable buffers and a safe FFT planning strategy; do not share one PhysicsModel scratch workspace concurrently. |
| Low | `elements.jl:128–132,194–195`; `types.jl:969–972,1014–1015` | Vector FFT construction, filtering, PMD, and display slice columns and allocate transforms. Some copies are required for the nonmutating API; replace avoidable slice staging with views or a dimension-1 plan. |
| Medium | `recipes.jl:35–61` | Multiple full N*saves power/dB matrices and cropped copies can substantially exceed solver output memory. Compute only displayed/cropped data where possible and handle zero fields/empty spectra. This is plotting memory, not propagation-loop allocation. |

GPU/Float32 claims are only partial scaffolding: public pulses/output matrices hard-code ComplexF64; Solution axes are Float64; noise signatures require ComplexF64 even on their no-op paths; the carrier array is CPU Float64. `similar(template)` and AbstractArray bounds alone do not establish GPU-safe or end-to-end generic propagation. Explicitly test supported precisions/devices or narrow documentation.

### 2.3 Complexity review

No avoidable O(N²) pattern was found in the propagation loops. Each step is O(N log N) plus O(N) algebra for a fixed number of stages/polarizations. Sellmeier and Taylor sampling are O(K*N), appropriate for K coefficients, although Horner can reduce exponentiation cost. Tabulated interpolation uses binary search per query, O(N log M); for sorted grids a monotonic cursor can achieve O(N+M). Spectrogram/FROG output itself needs O(N*delays) storage, so calling the total diagnostic cost “quadratic and reducible to O(N)” would be misleading when delays scales with N.

Performance priorities should follow correctness: repairing nonlinear SSFM integration changes its work per step. Benchmark cost to achieve a fixed physical error, not only raw steps/second. Add warmed allocation measurements and accuracy-versus-time benchmarks for passive Kerr, Raman, amplifier, semiconductor, vectorial, and repeated short sweeps; report setup time separately.

## Phase 3 — Test robustness (critical)

### 3.1 Classification key

**A:** known analytical/numerical identity with independent expected value. **P:** physical invariant, qualitative trend, or approximate physical scaling (useful but incomplete). **X:** external numerical reference, not an exact solution. **I:** internal consistency/shared implementation. **S:** structure, smoke, constructor/exception/metadata behavior. **D:** derivative of the implemented numerical map against finite differences (does not validate the physical model).

Assertions of type S are legitimate API tests. Their limitation is using them as evidence of physics correctness. A mixed row identifies the stronger assertions and the remaining weak ones.

### 3.2 Per-test inventory: core unit/API tests

| File / testset (line) | Classification and actual assertion coverage | What can pass incorrectly / improvement |
|---|---|---|
| `test_unit.jl` Grid (7) | S + A: lengths, monotonicity, positive dt, carrier formula, endpoint span, W=omega0+V | Wrong bin spacing or odd-grid handling can pass. Add exact FFT tones and invalid sizes. |
| Dispersion operator (28) | A + S: beta2/beta3 coefficients at one off-center bin, D(0), loss -alpha/2 | Good sign/unit tests; broaden bins/orders and add nonzero beta1, callback z. |
| Raman response (60) | P + S: all three silica models' fraction/type/shape, causality/nontriviality, unit area within 1% | A wrong causal, unit-area response passes. Add sample waveform and Raman-gain resonance/first-moment checks; refine dt/window. |
| Analysis (81) | A/P/S: Gaussian peak/FWHM/TBP; energy/bandwidth/photon merely positive; symmetric centroid near zero; bad domain rejected | Energy or photon normalization can be wrong; centered centroid cannot catch sign errors. Add exact energy and shifted spectra. |
| Wavelength grid & soliton metrics (99) | A + I: carrier wavelength, LD and LNL formulas; N computed via package LD/LNL | Shared mistakes in soliton metrics can cancel; use numerical expected dimensionless cases and degeneracies. |
| Noise & coherence (115) | S/I/P: perturbation, approximate energy, seeded reproducibility/independence, identical g=1, random g<0.3, shape errors | Wrong photon-per-mode amplitude or distribution can pass; g<0.3 for 300 realizations is very loose and unseeded. Test ensemble moments and known coherence mixtures. |
| `test_api.jl` Medium (5) | S: fields and negative scalar arguments rejected | Wrapped/callback gamma, nonfinite values and direct constructors need tests. |
| Medium keyword constructor (20) | S: defaults/promotion/exclusive dispersion arguments | Useful constructor contract, no physics. |
| Dispersion models (39) | I + S: tabulated samples copied from Taylor on identical nodes; invalid order/one sample; empty Taylor zero | No interpolation between nodes or extrapolation expected values; on-grid copying can pass. |
| Sellmeier dispersion (53) | S + weak A: coefficient unit conversion, length error, B(0)=0, zero loss | Wrong off-carrier dispersion/derivative can pass. Add independently tabulated index and finite-difference beta2 in supported band. |
| Raman models (80) | S: default constants and one override | No invalid fractions/times or model response shapes. |
| SimParams (91) | S/I/P: defaults/errors; ERK4IP empty-AW shape; zero loss/gain! buffers; both sweep forms' result lengths/type and power ordering | Does not compare serial/sweep results or exercise nonzero mutating loss/gain, all save modes, or stochastic reproducibility. |
| Pulses (139) | A + S: sech/Gaussian peak and FWHM, constant CW power; two invalid sech inputs | No lorentzian, pulse energies, CW noise statistics, invalid Gaussian/CW inputs, or FFT invariant. |
| build_physics_model (159) | S: D length, RW presence, fr, W constant/variable | Tests inspect W which kernels do not use; missing/misordered shock factor in gamma_W could pass. Check operator action. |
| Z-dependent gamma (180) | S/I: function property/sample values and output dimensions | Does not measure propagated phase; stronger independent tapered-SPM test exists in physics file. |
| Cascaded propagation & Lumped Elements (199) | A: amp/attenuator dB peak factors and amplifier energy ratio; P: filter lowers peak; S: stage types; I: piping equals same cascade | C2/C3 pass: no field continuity, transfer-bin check, or homogeneous long-fiber equivalence. |
| Frequency-dependent and Effective-area Nonlinearity (252) | I: constant wrapper/plain agreement, flat frequency gamma/plain no-shock agreement, constant area/shock-gamma agreement. A: Marcuse area near 6.675e-11 with 5%; S: Marcuse propagation endpoint | No nonflat gamma reference, shock toggle contract, amplifier equivalence, or AdaptiveSSFM controller check. |

### 3.3 Per-test inventory: solvers and physics

| File / testset (line) | Classification and actual assertion coverage | Weakness / next assertion |
|---|---|---|
| `test_solvers.jl` Solution structure (5) | S/I: dimensions, saved z endpoints/order, axes, initial At | A solver returning repeated initial field at all z passes. |
| Physics-flag combinations run (28) | S: finite At/AW for four Raman choices times two shock flags | Flags could be completely ignored. |
| Adaptive tolerance (45) | I: loose/tight final fields have squared relative norm <1e-3 | Two wrong solutions agree; tolerances ignored entirely could pass. No expected error improvement/order. |
| Tabulated dispersion matches Taylor (70) | I: table generated by package Taylor; propagated squared error <1e-6 | Shared operator/order bugs invisible. |
| Solver abstraction (101) | S: ERK4IP fields/default dispatch, negative tolerances, incompatible options; finite output | No ERK convergence or controller validation. |
| Symmetric SSFM (130) | S + I: dz errors, finite output, squared relative difference from ERK <1e-4 | Allows 1% L2 error and does not test order. First-order C1 passes at sufficiently small h. |
| AdaptiveSSFM (159) | S + I: negative phi, finite output, squared difference from ERK <1e-3 | Allows about 3.16% L2 error; no phase-bound/error-control or alternative gamma checks. |
| Spectrogram & FROG (183) | S: delays count, matrix shapes, finite/nonzero | Doubled/wrong frequency axis, normalization and delay symmetry all unchecked. |
| ERK4IP Operator Caching (200) | S: **only endpoint equals 0.05** | Recomputing all plans/operators at every stage still passes. Replace name or add instrumentation/allocation benchmark. |
| ERK4IP diverges cleanly (213) | S regression: extreme field throws ErrorException | Useful; generic error can satisfy it and a hang still hangs test process. Check message/type and wall-clock deadline. |
| z_saves >=2 (229) | S: 0 and 1 rejected | Good contract check, not physics. |
| `test_physics.jl` Gaussian sqrt(2) broadening (24) | A: FWHM at LD within 2% | Real benchmark. Only one z, beta2 sign, and ERK4IP; width does not validate chirp/phase. |
| Fundamental soliton peak (45) | A/P: peak remains P0 within 3% across saves spanning 2 soliton periods | Peak alone misses phase/tails/shifts. A no-op solver passes this test in isolation. |
| Energy conservation (67) | P: final/initial energy ratio within 1e-4 with Raman and no shock | Valuable invariant, but wrong unitary propagators or no-op pass. |
| Loss exponential (84) | A: energy ratio exp(-alpha L) within 1e-3 | Real independent benchmark; test all solvers, z-dependent and spectral loss, full field. |
| Raman red shift (104) | P: lower spectral centroid and later temporal centroid | Sign only, not shift rate; small wrong perturbation can pass. |
| SPM spectral broadening (132) | P: final RMS width exceeds initial for sech | No Gaussian quantitative broadening curve or exact phase; numerical noise/aliasing can broaden too. |
| Tapered gamma SPM phase (149) | A: Gaussian sample phasors match integral gamma(z) at significant samples, 1e-4 | Strong benchmark; only ERK4IP and intensity >1 W mask. Also assert amplitude and vary L/dt/window. |
| Fundamental soliton phase (192) | A: **one central sample** ratio equals exp(i*gamma*P0*L/2), 1e-3 | Stronger than peak preservation; still lacks full sech envelope/phase at all times and distances. |
| High-order fission distance (219) | P: global peak-compression snapshot within 30% of LD/N, with beta3 perturbation | Approximate scaling, not exact recurrence or proof of fission. Tracking returns compression even without separated solitons. |
| “Pure SPM Peak Phase Shift & Spectral Peak Count M” (242) | P: **only RMS width ratio >1.5** | Neither phase nor number of spectral peaks is asserted. Rename or add the named observables. |
| Vector XPM+FWM coupling (259) | A/P: ratio of short-distance phase shifts about 4 within 2%, deltaBeta0=0 | Good coefficient-ratio check; wrong common gamma scale cancels. Add absolute phase and relative-polarization-phase cases. |
| “FWM Peak Gain Detuning” (292) | P: **only spectral width increases** for normal-GVD Gaussian | No sideband seed, predicted detuning, gain curve, or measured peak location. Ordinary SPM satisfies it. |
| Molecular Raman impulse (311) | S/P: fraction, length, finite, negative-time zero | Wrong oscillation frequency/damping/area passes. |
| FCR blue shift (323) | P: final centroid increases | Useful sign check; no magnitude or carrier recurrence reference. |
| EDFA saturation energy limit (347) | P: 1 < energy gain <=10 | No exact saturation law. Comment says E_in>>Esat but input is about 53 nJ versus 1 microjoule Esat; initial regime is not heavily saturated. |

### 3.4 Per-test inventory: specialized physics, utilities, plotting

| File / testset (line) | Classification and actual assertions | Weakness / action |
|---|---|---|
| `test_vectorial.jl` Type validation (5) | S: some medium invalids, pulse shape invalids, fields | Add AW shape, component-length errors, unsupported solvers/medium pairing. |
| Comparison with scalar (32) | I + P: x agrees with scalar SSFM, y stays zero, result shape | Both methods share Euler update; cannot establish order or nonlinear accuracy. |
| Birefringent Walk-off (66) | A: beta1*L =1 ps peak separation within 2% | Good independent linear benchmark; add negative beta1 and complex profile/centroid. |
| Vectorial analysis/cascade (100) | I/A: energy/peak sums for aligned pulses, amp dB factor; S: cascade types/length, piping type | Unaligned peaks not checked; PMD can be identity and pass; no PMD DGD/Jones statistics or unitarity. |
| `test_edfa.jl` Constructors (10) | S/A: fields, 10 dB gain conversion | No invalid Esat/NF/gain controls. |
| Propagation & Gain Saturation (28) | A/P: low-energy gain 3.162 within 5%; high-energy gain <5 | Missing Kerr phase invisible. Highly lossy/wrong saturated model can satisfy the upper bound. Seed noise and use exact implicit energy law. |
| ASE injection (67) | P/I: nonzero distant field, larger NF increases noise, passive residual much smaller; explicit seeds | No absolute PSD/variance/normalization, accumulated gain scaling, or ensemble confidence intervals. Tail nonzero alone is weak; relative-NF assertion is more useful. Only SSFM ASE tested directly. |
| NonlinearityModel gamma types (121) | S: constant-wrapper and frequency-model amplifier return Solution | Does not even assert these two equivalent gammas agree or saturate; C4/H2 pass. |
| `test_hollowcore.jl` Gas index/pressure (5) | P: n>1, increasing, n-1 scales by five; S: unsupported gas throws | Arbitrarily wrong index magnitude/dispersion passes. No temperature or other gases. |
| HollowCore constructor/dispersion (16) | S: types/fields, gamma>0, propagation endpoint | beta2/beta3/beta4 and gas constants are unvalidated. |
| HollowCore with grid (37) | S regression: table type, result type/endpoint | Useful shadowed-length regression but no numerical table values/equivalence. |
| Molecular Gas Raman Models (57) | S: default fractions, response length/fraction, invalid gas | No actual vibrational response or N2 response tested. |
| Capillary confinement loss (72) | S/P: default zero, enabled finite/positive, extra loss increases values | No loss magnitude, radius^-3/wavelength² scaling, or exact additive offset. |
| `test_semiconductor.jl` Constructor (8) | S: fields/default FCA/FCR | No invalid physical parameter tests. |
| “TPA Loss & Free Carrier Blue Shift” (26) | S/P: type, endpoint, **energy decreases only** | Does not check blue shift here (separate physics test does); arbitrary damping passes TPA. |
| NonlinearityModel gamma types (48) | I/S: numeric/constant wrapper agree 1e-8; unsupported frequency gamma rejected | Good dispatch regression; shared physical errors remain. |
| `test_loss_gain.jl` Vector loss (8) | S: Solution and endpoint | Vector loss could be ignored and pass. |
| Vector gain (23) | P/S: Solution and increased energy | Missing spectral shape, saturation, ASE all pass. |
| Functional z loss/gain (43) | S: Solution/endpoint; **only loss configured** | Frozen D passes; no gain callback tested despite title. |
| Silica Loss Spectrum (55) | A/P/S: ~0.295 dB/km within 10%, OH >30 dB/km, D length and negative real part | One rough anchor; independent spectrum/reference and peak location needed. |
| `test_fibers.jl` Glass presets (5) | S: types/array lengths, Ge concentration errors | Wrong Sellmeier coefficients pass. |
| Commercial catalog (23) | S: keys, selected values, overrides, unknown key | Checks stored constants, not manufacturer accuracy; presets explicitly disclose representative values. |
| Commercial propagation (49) | S: result type/endpoint | Incorrect dispersion or gamma passes. |
| `test_conversions.jl` D/beta2 (5) | A: -21.67e-27 within 1%; I: inverse roundtrip | Good numerical anchor; rounded expected value supports modest tolerance, but add exact dimensional cases. |
| S/beta3 (19) | A: 0.1264e-39 within 2%, positive | Good anchor; negative slope/sign and limiting cases missing. |
| Wavelength/frequency (28) | A: 193.414 THz within 1e-3; I inverse | Add direct inverse anchor/invalid wavelength/frequency contract. |
| Soliton Tracking (35) | S: vector lengths and z_fiss inside simulation interval | Almost any argmax-based implementation passes; no actual soliton identities/trajectory. |
| Cherenkov analyzer (49) | P: wavelength between 450 and 820 nm | A hardcoded number in that interval passes. Assert analytic polynomial root and phase-matching residual/no-root behavior. |
| `test_recipes.jl` Plot Recipes (5) | S: Plot type and four subplots | Wrong axes/data, zero-field NaNs, empty-AW failure, missing units/Jacobian all unchecked. |

### 3.5 External reference tests and fixture quality

| `test_adversarial.jl` scenario | Classification and assertions | Limitations |
|---|---|---|
| 1 fundamental soliton (70; nested per-z checks at 95) | X/P: each peak vs CSV within 0.5%, Julia/reference drift each <0.5% | Scalar observable, not full complex solution; comment above scenario says 5% while code is 0.5%. |
| 2 SPM (113) | X/P: near-monotonic widths in both datasets; final/initial broadening ratio within 1% | Comment says 0.1%; scaling errors cancel in ratios. Does not compare whole width curve or Gaussian exact result. |
| 3 dispersion (150) | X + A: final FWHM vs CSV 1%, and analytic sqrt(2) width 2% | Real independent analytical check as well as reference; only final snapshot checked. |
| 4 Raman (184) | X/P: red-shift signs and total shift magnitude within 0.5% | Repeated sign assertions are not independent evidence. Full shape/rate not checked. |
| 5 output field (225) | X: normalized squared complex overlap >=0.999 | Invariant under arbitrary nonzero amplitude scaling and global phase. A fundamental soliton with completely missing evolution phase can pass. Loaded time grid unused. |
| 6 vector (257) | X/P/S: fixture exists; per-component overlap >=0.95; peaks within 5% | Overlap ignores each component's global phase, so relative polarization phase is untested. Time-grid construction differs from fixture and reference shock behavior differs (below). |

**Fixture generation defects needing correction before tightening numerical thresholds:**

1. `test/generate_reference_data.py` calls `DispersionFiberFromTaylor(BETA2_PS2M, [BETA2_PS2M])` in soliton, dispersion and Raman scenarios. The first argument is **loss**, so beta2=-0.01 in ps²/m becomes loss=-0.01 dB/m rather than zero. This means reference and Julia setups are not identical. The upstream documented signature is `(loss, betas)` with loss in dB/m. [gnlse-python dispersion documentation](https://gnlse.readthedocs.io/en/latest/dispersion.html).
2. The scalar generator mixes `self_steepning` with misspelled `self_steenpening`. Python permits setting unused attributes; the latter does not explicitly configure the documented option. Default behavior may currently mask this. Assert effective configuration and pin exact generator dependencies. The upstream documented field is `self_steepning`. [gnlse-python integration documentation](https://gnlse.readthedocs.io/en/latest/gnlse.html).
3. Vendored `test/cnlse_pkg/cnlse/cnlse.py` always uses the absolute frequency `self.W` in the nonlinear term and never reads `setup.self_steepning`. The vector generator sets this option false, but it has no effect on that implementation; Julia scenario 6 disables shock. Thus the systems compared are not exactly the same, even though the narrowband discrepancy may be small.
4. The vector test computes `t_span=t_ref[end]-t_ref[1]+dt`, then passes it to an inclusive-endpoint Julia grid. The generator itself uses inclusive linspace endpoints. Julia's resulting spacing/window differs from the fixture by one extra original dt. Check axis identity explicitly rather than compensating through a loose overlap threshold.
5. The vector generator intentionally enables a zero-strength Raman callback to include the reference coherent FWM term. This is explained and is better than ignoring the branch difference, but confirms that a peer implementation is not automatically a physics oracle. Record the exact vendored revision (README currently names a repository, not a commit), effective flags, versions and fixture hashes.
6. The generators are not run by the normal Julia tests. No environment lock/asserted version is enforced by the scalar script itself; its docstring states 2.0.0. Treat committed CSVs as reproducible numerical fixtures only after a clean regeneration audit. Do not regenerate them from Soliton output to make failures disappear.

### 3.6 Enzyme tests and runner wiring

`test/runtests.jl` includes all 13 non-Enzyme behavioral test files. `test_enzyme.jl` is deliberately excluded from the ordinary suite/test dependencies but **is explicitly run by a separate CI job** in `.github/workflows/CI.yml:43–71` on Julia 1.12 x64. It is not dead or completely unrun coverage.

| Enzyme test (line) | Classification | Assessment |
|---|---|---|
| Extension exists (130) | S | Useful packaging assertion. |
| Adjoint plan identities (134–164), four vector/matrix forward/inverse plans | A/I | Strong algebraic dot-product identity plus comparison with upstream adjoint operator. No propagation physics claim. |
| Isolated plan reverse gradient (166–193), two plans | D | One complex component vs forward finite differences, rtol=1e-4/atol=1e-6; expand locations/scales and check analytic quadratic gradient. |
| Isolated forward mode (195–225), two plans | D | Random direction vs central difference, same tolerance; seed randomness and compare with exact quadratic derivative. |
| Full SSFM beta2 forward (227–266) | D | Single-step energy derivative vs central difference, 1%; tests derivative of Euler-biased map, not true physical sensitivity. |
| `_spm` nonlinear reverse gradient (269–288) | D | One component vs finite difference, 1e-4/1e-6; useful boundary regression, not Kerr normalization validation. |
| Full SSFM betas reverse (290–349) | D | Single beta2, single-step output-energy gradient vs FD, 1%; same physical limitation. |
| Pulse-shape ladder (351–417), six cases | D | FFT into mutable field; copied spectrum; linear single-step; nonlinear single-step; linear five-step; stronger-dispersion shape mismatch. rtol 1e-4 or 1e-3, atol 1e-6. Useful localization and meaningful strengthened mismatch case, but only peak-index derivative checked. |

The beta2 loss is conserved total energy in the ideal lossless equation. Nonzero beta2 sensitivity can arise from the artificial Euler nonlinear energy drift. Accurate AD of that discretization is possible while the physical optimization objective is wrong. Add nonconserved shape/phase observables with independently verified finite differences and convergence as h decreases. The claim that an absolute FD step near 1e-30 automatically makes differences unreliable is incomplete: nondimensionalize beta2, examine dimensionless perturbations and sweep h to find a stable plateau. Enzyme tolerances have some written rationale, but most tests do not establish an FD error estimate.

### 3.7 Tolerances and false confidence

- Squared relative field errors are called `rel`/`rel_diff`: 1e-4 means 1% relative L2, not 0.01%; 1e-3 means 3.16%. Report the norm/error definition alongside thresholds.
- Physical-grid tolerances of 1–3% can be reasonable for noninterpolated FWHM/peak position. Existing comments justify some peak discretization; there is no systematic dt/window refinement budget. For a crossing-bin width estimate, derive an absolute error from dt and normalize by the expected width.
- 30% fission scaling and 450–820 nm dispersive-wave bounds are heuristic acceptance windows, not high-accuracy benchmarks. Name them accordingly and add stronger exact/reference checks.
- 5% unsaturated EDFA, 10% silica-loss and 5% Marcuse thresholds have approximate reference values but no measured numerical-error budget. Separate approximation uncertainty from solver error.
- Unseeded noise/coherence, EDFA default RNG and vector PMD tests impede reproducible diagnosis. Statistical tests need fixed seeds or reproducible ensembles plus sample-size-based confidence bounds; same-seed equality alone is not a distribution test.
- Identical input/output invariants, finiteness, shape, correlated implementations and normalized overlaps are individually insufficient. Mutation checks (deliberately disable Kerr, reverse dispersion, ignore Raman/shock, rotate FFT bins, remove z dependence) would reveal which claimed physics tests can actually detect each error.

### 3.8 Required analytical benchmark upgrades

These equations follow by reduction/substitution in the documented GNLSE; they are proposed independent test oracles, not empirical fits to Soliton output. The upstream equation explicitly includes dispersion, Kerr/Raman and shock terms. [gnlse-python equation and convention](https://gnlse.readthedocs.io/en/latest/gnlse.html).

1. **Fundamental soliton full field:** for beta2<0, `LD=T0²/abs(beta2)`, `P0=abs(beta2)/(gamma*T0²)`, compare every meaningful time sample at multiple z with `sqrt(P0)*sech(t/T0)*exp(i*z/(2*LD))`. Include absolute global phase, relative L2/Linf field error, envelope width/peak, tails, and conservation. Run all supported solvers at controlled error and refine time grid/window.
2. **Higher-order recurrence:** unperturbed N=2 or N=3 sech input, no Raman/shock/loss/higher dispersion, check recurrence of intensity/spectrum at `z0=pi*LD/2` and a known nontrivial intermediate profile (N=2 closed form is suitable). A fundamental soliton remains unchanged at every distance, so it cannot by itself prove a period. The existing fission test is not a recurrence test.
3. **Gaussian pure SPM:** `A0=sqrt(P0)*exp(-t²/(2*T0²))`; exact `A(z,t)=A0(t)*exp(i*gamma*z*|A0(t)|²)` without loss. Compare complex field and its independently transformed spectrum. With B=gamma*P0*z, the RMS broadening ratio is `sqrt(1+4*B²/(3*sqrt(3)))`, from the Fourier derivative/second-moment identity. Sweep B, including zero. This provides a quantitative Gaussian benchmark of the kind requested from Agrawal chapter 4 rather than just “width increased.”
4. **Gaussian dispersion:** `FWHM(z)=FWHM(0)*sqrt(1+(z/LD)²)`, `LD=T0²/abs(beta2)`. Check multiple z, both signs of beta2, amplitude and complex chirp; width alone cannot catch dispersion sign. Use the analytically solved complex Gaussian as a separate oracle.
5. **Convergence order:** three or more step sizes h,h/2,h/4 with endpoint/save spacing held consistent; compute `p=log2(e(h)/e(h/2))` against exact solutions. Establish the asymptotic range before selecting an order tolerance; exclude roundoff/spatial-error floors. Proper symmetric SSFM should approach two. For ERK4IP isolate fourth-order global integration behavior with a controlled step strategy; simply tightening rtol is not a fixed-step order test.
6. **Loss/gain/TPA/carriers:** exact integrated linear coefficients; scalar saturation energy identity `log(Eout/Ein)+(Eout-Ein)/Esat=g0*L` for the stated noise-free, loss-free gain law; pure TPA `P(z,t)=P(0,t)/(1+alpha2*P(0,t)*z/Aeff)`; prescribed carrier source with analytic exponential decay/integral. Add a supported way to disable ASE for deterministic gain benchmarks or separate deterministic operator tests from ensemble stochastic tests. NF=0 dB does **not** disable ASE in the current formula.
7. **Noise/PMD:** validate one-photon mode energy variance, RIN power variance, linewidth autocorrelation, ASE PSD and accumulation, PMD Jones unitarity and DGD mean/distribution, known shifted-tone frequency/time axes. These require quantitative ensemble checks, not just changes between random draws.

## Phase 4 — Public API coverage and missing edge cases

### 4.1 Exported-symbol matrix

All exported bindings in `src/Soliton.jl` are listed below. Abstract types/constants/catalogs are included separately from functions. “Dedicated” means assertions specifically target that symbol/behavior; “indirect” means construction/use alone. It does not mean all methods or physics have been covered.

| Public symbol(s) | Dedicated coverage | Important gap |
|---|---|---|
| `Medium` | Yes: API constructors, units, invalid scalars; physics integration | Nonfinite/wrapped/callback inputs and direct constructor consistency |
| `SimParams` | Yes: API/solver defaults, conflicts, z_saves, one save mode | Positional compatibility, all solver/save/flag combinations |
| `Grid` | Indirect via create_grid | Direct field consistency/validation |
| `Pulse` | Indirect creation/conversion across suite | Dedicated FFT invariant, malformed dimensions, scalar Solution conversion/cascade |
| `Solution` | Yes: solver structure | Malformed direct construction, empty AW consumer contract |
| `VectorialPulse` | Yes: vector shapes and analysis | Three-argument AW shape/invariant, ComplexF32, aliasing |
| `BirefringentMedium` | Yes: fields/negative length/gamma and walk-off | Invalid loss/wavelength, supported solvers, phase mismatch |
| `VectorialSolution` | Yes: shape and cascade result types | Direct conversion invariant, display, empty-spectrum behavior |
| `AmplifyingMedium` | Yes: constructors/gain/saturation trends | Kerr normalization, spectrum/callback gain saturation, negative/nonfinite values |
| `SemiconductorMedium` | Yes: fields, energy loss, gamma dispatch | Exact TPA/carriers/FCR magnitude and invalid values |
| `RamanModel` | Indirect subtype use | Extension contract and unsupported custom model errors |
| `BlowWood` | Yes: defaults, response properties, SSFS reference | Exact response spectrum/first moment and invalid parameters |
| `LinAgrawal` | Yes: defaults/response area and finite propagation | Quantitative Raman gain/propagation reference |
| `Hollenbeck` | Yes: defaults/response normalization, finite propagation | Independent oscillator response/propagation reference |
| `MolecularRamanGas` | Yes: H2 default fractions/rotational response, bad gas | N2, vibrational waveform, damping/fraction constraints |
| `DispersionModel` | Indirect | Custom model contract/fallback errors |
| `TaylorDispersion` | Yes: beta2/beta3/zero expansion, vector beta1 indirectly | High orders, factorial overflow, nonfinite coefficients |
| `TabulatedDispersion` | Yes: on-node equivalence, one sample/order errors | Midpoint interpolation, endpoints, flat extrapolation, duplicate/nonfinite x, unequal lengths |
| `SellmeierDispersion` | Yes: unit conversion, length error, zero carrier deviation | Off-carrier known values, resonance/cancellation errors |
| `GNLSESolver` | Indirect dispatch | Custom/unsupported solver behavior |
| `ERK4IP` | Yes: configuration/errors, physics, divergence | Order, initial-step validation, tolerance error scaling, step limits |
| `SSFM` | Yes: dz checks, comparisons, walk-off | Independent nonlinear accuracy/order/conservation |
| `AdaptiveSSFM` | Yes: phi rejection, cross-solver comparison | Bounds/callback gamma, nonfinite states, save_freq, independent order |
| `LumpedElement` | Indirect in cascade/piping | Custom stage error/return contracts |
| `Amplifier` | Yes: scalar/vector power gain | Field phase/invariant, extreme gain |
| `Attenuator` | Yes: scalar power factor | Vector dedicated test, negative/nonfinite loss policy |
| `Filter` | Only weak scalar peak-decrease assertion | Correct pass/stop bins and phase; vector path untested |
| `PMDElement` | Indirect vector cascade/type only | DGD distribution, unitarity, deterministic replay, scalar rejection, negative DGD |
| `apply` | Yes: amp/attenuator scalar, amp vector; weak filter | Vector attenuation/filter, PMD numerical action, unsupported pairings |
| `NonlinearityModel` | Indirect subtype use | Extension/unsupported input contract |
| `ConstantNonlinearity` | Yes: passive/semiconductor equivalence; amplifier smoke | Negative gamma validation, amplifier physical equivalence |
| `FrequencyDependentNonlinearity` | Yes: flat passive equivalence, amplifier smoke, semiconductor rejection | Nonflat oracle, shock contract, adaptive coefficient control |
| `NonlinearityFromEffectiveArea` | Yes: constant-area passive equivalence, Marcuse smoke | Varying-area oracle, zero/negative area, amplifier support |
| `FiberSpec` | Indirect catalog entries | Fields/constructor/units and independent data provenance |
| `FiberLibrary` | Yes: five key presence checks | Integrity/uncertainty metadata and mutation effects |
| `commercial_fiber` | Yes: values/overrides/errors; propagation smoke | Changing wavelength leaves Taylor coefficients unchanged; validate/document reference wavelength limits |
| `HollowCoreFiber` | Yes: two builder branches and loss trends | Known dispersion, radius/pressure/temp limits, derivative accuracy |
| `gas_refractive_index` | Yes: Ar pressure scaling and invalid gas | Absolute reference index and dispersion for all gases, temperature |
| `FusedSilica` | Yes: model type/coefficient lengths | Independent n(lambda)/beta values |
| `SF6` | Yes: type only | Coefficients/valid wavelength range/dispersion |
| `SF57` | Yes: type only | Same |
| `GeO2DopedSilica` | Yes: type/concentration bounds | Independent concentration-dependent index/dispersion |
| `SilicaLossSpectrum` | Yes: one rough anchor, OH bound, sign | Full spectrum, parameters, validity range |
| `step_index_aeff` | Yes: approximate SMF numerical area | Invalids, scaling, cutoff/multimode validity |
| `MarcuseAeff` | Indirect via mode-area solve | Dedicated call/value/invalid constructor assertions |
| `dispersion_D_to_beta2` | Yes: numeric anchor/sign | Invalid wavelength, parameter sweep |
| `beta2_to_dispersion_D` | Yes: roundtrip only | Independent expected value and limits |
| `dispersion_S_to_beta3` | Yes: numeric anchor/sign | Independent multiple cases/unit extremes |
| `wavelength_to_frequency` | Yes: 1550 nm anchor | Nonpositive/nonfinite values |
| `frequency_to_wavelength` | Yes: roundtrip only | Independent anchor and invalids |
| `create_grid` | Yes: common grid structure | N=0/1/odd/2, invalid window/wavelength, FFT-bin alignment |
| `wavelength_grid` | Yes: Grid method carrier/length | Solution method dedicated test; W<=0 |
| `sech_pulse` | Yes: width/peak/invalids/soliton physics | Exact energy, zero field, FFT invariants |
| `gaussian_pulse` | Yes: width/peak/analytical propagation | Invalids/energy/window truncation |
| `lorentzian_pulse` | **No test calls** | Full width/peak/energy/tails and invalid arguments |
| `cw_pulse` | Yes: noiseless constant intensity only | Pn/rng statistics, negative inputs, spectrum DC |
| `dispersion_operator` | Yes: Taylor/loss anchors, Sellmeier shape | Dynamic z use, unsupported models, scalar/vector semantics |
| `propagation_constant` | Yes: Taylor and table equivalence, Sellmeier B(0) | Between-node/extrapolation/high-order/known Sellmeier dispersion |
| `loss_vector` | Yes: confinement and silica-related values/trends; otherwise indirect | Exact vector/callback mapping and z integral |
| `loss_vector!` | Yes: zero buffer only | Nonzero equality, wrong buffer/spectrum length, callback overloads |
| `gain_vector` | Indirect through builders | Dedicated nonzero scalar/vector/callback expected values |
| `gain_vector!` | Yes: no-gain medium zeros only | Positive-gain and mismatch cases |
| `raman_response` | Yes: all silica response properties, rotational gas | Exact model values, vectors' time-grid validation |
| `solve` | Yes: many scalar/vector/cascade paths | Homogeneous cascade equivalence, functional stages, all save/solver/model combinations |
| `solve_sweep` | Yes: both forms, lengths and weak power ordering | Serial equality, multidimensional/empty inputs, mismatched lists, RNG/thread reproducibility |
| `build_physics_model` | Yes: passive structure; other builders indirect | Kernel action/normalization, template compatibility, plan/workspace ownership |
| `pulse_energy` | Yes: positivity/vector sum and gain/loss ratios | Absolute analytical Gaussian/sech energy and normalization |
| `peak_power` | Yes: pulse peaks/vector aligned sum | Separated polarization maxima, empty/zero fields |
| `fwhm` | Yes: Gaussian/sech widths, bad domain, frequency via TBP | Single-above-half sample, multimodal, zero/empty fields, edge clipping |
| `spectral_bandwidth` | Yes: positivity only | Quantitative width, level range/interpolation |
| `time_bandwidth_product` | Yes: Gaussian ~0.441 | Sech ~0.315, chirped reference, refinement |
| `photon_number` | Pulse positivity; Solution indirectly used by solve warning | Dedicated exact count-proxy/invariance, no-AW equivalence, W<=0 |
| `spectral_centroid` | Yes: centered symmetry only | Known signed detuning and zero field |
| `dispersion_length` | Yes: direct formula | beta2=0 and invalid width conventions |
| `nonlinear_length` | Yes: direct formula | zero/negative gamma and power |
| `soliton_number` | Yes: shared LD/LNL formula | Independent N=1/N=2 values and degeneracies |
| `add_noise` | Yes: basic seed/perturbation/energy | Quantum mode energy, phase_only/RIN/phase/linewidth options and invalids |
| `rin_rms` | **No test calls** | Exact PSD-bandwidth relation, invalid bandwidth |
| `spectral_coherence` | Yes: raw spectra identical/random and malformed ensembles | Pulse/Solution overloads, known partial coherence, zero spectra, absent AW |
| `spectrogram` | Yes: smoke/shape/nonzero | Analytic trace, axis/sign/normalization and gate/delay validation |
| `shg_frog_trace` | Yes: smoke/shape/nonzero | Analytic axes/trace/delay symmetry and interpolation |
| `track_solitons` | Yes: dimensions/range, heuristic compression distance | Known trajectory/multiple peaks, no-fission interpretation, absent AW |
| `dispersive_wave_wavelength` | Yes: broad wavelength interval only | Analytic roots/residual/no-root/negative frequency and gamma variants |
| `c` | Indirect formula use; no dedicated exact-value assertion | Tiny SI-constant assertion optional |

Unexported but reachable extension-facing surface: `PhysicsModel`, `propagate`, abstract media/pulse/component types, and callable configuration/element methods. These deserve documented support boundaries if qualified access is expected. Base.show methods and the recipe are framework API; only the recipe has dedicated tests. Lack of `export` alone is not proof a symbol is dead in Julia.

### 4.2 Edge-case backlog

| Area | Missing edge cases and consequences |
|---|---|
| Grids/fields | N=0,1,odd; even non-power-of-two; empty/mismatched At/AW/grid axes; direct Grid/Pulse/Solution bypass validation; nonfinite axes/window/wavelength; zero/negative absolute frequencies. Some downstream loops use @inbounds, so validate shape before kernel entry. |
| Physical signs/finite values | Inf passes many `>0` checks. Negative wrapped gamma bypasses bare-scalar Medium checks; semiconductor gamma/FCA and amplifier gamma are less constrained. Loss vector/callback values can be negative although scalar passive loss is rejected. Define legitimate signed physics separately from unsupported nonphysical values. |
| Solvers | dz_init=0/negative/NaN/Inf; dz_min<=0 or >dz_max; nonfinite tolerances; tiny L; extreme gain/Kerr/carrier rates; no-progress/rejection timeout; step count overflow; fixed SSFM `round(dz_save/dz)` may choose h_eff larger than requested and changes h with z_saves. |
| Empty/zero analysis | `_fwhm` calls maximum on empty input and returns 0 for one above-half sample even when crossings are interpolable; zero centroid divides by zero; zero plot normalizes 0/0; empty saved spectrum breaks consumers. Define behavior, then test. |
| Raman | fr outside [0,1], nonpositive tau, fractions fb/fc, nonuniform/reversed/short time vectors, zero normalization integral for Hollenbeck, windows too short for long-lived molecular response. fr=0 still computes convolution under current enable logic. |
| Tabulated/Sellmeier | Unequal lengths, duplicate detunings (issorted permits equality), NaN/Inf samples, extrapolation, resonance poles, nonpositive absolute omega, cancellation of beta-beta0-beta1*V near carrier. |
| Hollow-core/gases | Negative/zero temperature and wavelength, zero pressure/vacuum, all gas branches, low-radius cutoff. `arg_clamped=max(1e-10,arg)` silently creates a real propagation constant below cutoff; nonpositive wk bins silently retain zero tabulated dispersion. Higher finite differences b3/b4 at dw=1e-4*w0 need a numerical-conditioning/refinement audit. |
| Elements | Negative DGD, invalid/NaN gains, passband/phase filters, unsupported scalar PMD, vector attenuation/filter, zero-transfer output. Transfer functions and dispersion models should not silently use mismatched frequency frames. |
| Noise/statistics | Negative phase_rms currently silently behaves as zero; wrong quantum_model, negative strengths, zero input, abs(W) noise treatment for W<=0, seed replay across solver/stage/sweep; bandwidth/window dependence. |
| Sweeps/cascades | Empty/multidimensional spaces; vector length mismatch; custom function stage return validation; unsupported stage types; nested threaded sweeps (static scheduling constraints); callback thread safety; medium/grid mutation/aliasing; returned grid vectors are shared. |
| Diagnostics | Spectral level outside (0,1], n_delay<=0, negative gate width silently clamped, multiple spectral peaks, shifted carriers, no dispersive-wave root, invalid root frequency, clipping/window wraparound. |

## Action plan and acceptance criteria

1. **Repair C1–C4 and H1–H5 before relying on the affected features in quantitative studies.** Each fix should have an independent exact-field/operator regression that fails on this revision. Preserve these probes as evidence; do not loosen existing tolerances to hide discrepancies.
2. **Establish an accuracy baseline:** full N=1 field, N=2 recurrence/intermediate evolution, Gaussian SPM/dispersive broadening, exact loss, convergence order. Report error against h, dt and window, with all solver parameters recorded.
3. **Separate deterministic and stochastic physics verification:** exact gain/TPA/carrier laws; ASE/RIN/PMD ensemble moments with explicit confidence intervals. Add the missing amplifier Kerr and spectral-gain cases.
4. **Repair fixtures and their provenance:** zero loss where intended, verified shock flags, matching grids, pinned source revisions/configuration, automated validation of generated metadata.
5. **Make contracts explicit:** spectral ordering/scaling, save_freq consumer behavior, supported medium/solver combinations, domain validation, bounded solver failure behavior, and approximate diagnostic/preset validity.
6. **Then optimize measured hotspots:** concrete aux/grid types, adaptive exponential buffer, planned diagnostics, reusable per-solve workspace, confinement coefficient reuse. Require equal physical accuracy in performance comparisons and preserve AD behavior with dedicated validation.
7. **Improve test naming and reporting:** rename smoke/qualitative tests honestly; maintain an API/method coverage map; distinguish independent physics validation, numerical-reference comparison, consistency, performance, and AD checks in CI summaries.

## Runtime verification and audit limitations

### Existing suite

Ran the existing runner without modifying dependencies or test files:

```sh
JULIA_LOAD_PATH='@:@v#.#:@stdlib' GKSwstype=100 julia --project=. test/runtests.jl
```

**Result: 371/371 assertions passed in 1m54.2s** on Julia 1.12.7, Linux aarch64. The run included plotting and all normal-runner test files. It emitted a grid/medium wavelength mismatch warning and a 4.61% photon-drift warning; warnings did not fail tests. This was direct execution in the available environment, **not** a fresh isolated `Pkg.test()` environment or the full CI platform/version matrix. The separate Enzyme suite was read and its CI invocation verified but was not run locally. No claim is made that the AD suite passes on this machine.

The all-green normal run **coexists with the measured defects below**. This directly demonstrates that the current suite does not protect the affected contracts.

### Independent targeted probes

Probe script: `/tmp/soliton_audit_probes.jl`. Successful complete probe log: `/tmp/soliton_audit_probes_fixed.log`. Full test log: `/tmp/soliton_audit_tests.log`.

Common probe grid: N=256, time_window=10 ps, carrier=1550 nm. Unless noted, Gaussian peak=1 W, FWHM=1 ps, Raman disabled and two z saves. Values are observed outputs, not expected tolerances for future tests.

| Probe | Observed result | Expected/interpretation |
|---|---|---|
| Identity medium, then Pulse(Solution), relative AW-vs-ifft(At) residual | 1.414213562373095 | Near numerical roundoff for a consistent Pulse |
| Centered passband abs(omega-omega0)<1e13 rad/s, retained Gaussian energy | 5.4850909617848005e-33 | Near one for this pulse/passband |
| create_grid(1,...) | ArgumentError from construction | Explicit documented minimum-size rejection desired |
| Odd grids N=3 / N=5 | t/V lengths 3/2 and 5/4 | Both lengths must equal N |
| CW Kerr, L=0.5 m, gamma=1, P=1, SSFM h=0.05/0.025/0.0125 m | Relative field errors 0.0139213868 / 0.0069772750 / 0.0034916395 | Errors compared to A0*exp(0.5i) |
| Corresponding step-halving error ratios | 1.99524697 / 1.99828046 | Confirms first-order behavior |
| g0=0 amplifier vs passive Kerr derivative norm | 8.228698061258025e-16 | Should be one; equals 1/omega0 |
| Loss=(omega,z)->1+2z dB/m, L=0.5 m | Energy ratio 0.8912509381 | Exact integrated-loss ratio 0.8413951416 |
| AdaptiveSSFM save_freq=false | AW shape (256,2) | Empty spectrum under the other solvers' documented behavior |
| Lorentzian FWHM/input FWHM | 1.0015547661 | Nominal formula appears consistent; lack of tests is not evidence this pulse formula is wrong |
| Warmed passive `_spm` RHS allocation | 0 bytes | Supports allocation-free claim for this tested CPU path |
| Warmed `_amplifying_spm` RHS allocation | 256 bytes | Contradicts universal zero-allocation interpretation; consistent with abstract auxiliary-field concern |
| Export binding doc-metadata lookup | No missing exported bindings | Does not establish correct docstring attachment, accuracy, or per-method documentation |

Allocation measurements used `@allocated` after a warm-up call inside a Julia function. These are one-configuration measurements, not a throughput benchmark or proof about all types/models. No `@code_warntype`/JET-wide analysis, GPU testing, stochastic variance experiment, Enzyme execution, or fixture regeneration was performed. Potential type-instability findings beyond the measured amplifier path remain source-based risks to verify with inference tools.

An initial probe attempt stopped at an audit-script-only documentation lookup error after printing its numerical results. That lookup was corrected in `/tmp`, and the probe was rerun successfully to completion. The successful log above is the evidence used here; no package code was changed to make any probe pass.

### Source-file review checklist

| File | Main audit focus/result |
|---|---|
| `src/Soliton.jl` | Includes/exports/constants/precompile workload; single-module organization and mixed-unit introduction |
| `src/types.jl` | Constructor validation, concrete/abstract fields, doc attachment, Pulse/Solution conventions |
| `src/grid.jl` | Odd/single-point failure, inclusive endpoints, frequency/wavelength conventions |
| `src/pulses.jl` | Four envelope formulas, validation, allocations, untested Lorentzian/CW noise |
| `src/dispersion.jl` | Taylor/table/Sellmeier sampling, loss/gain callback semantics, static versus dynamic D |
| `src/raman.jl` | All four response methods, normalization, causality, invalid/undersampled time grids |
| `src/nonlinearity.jl` | All four builders, gamma normalization, buffers/plans, gain/carrier/vector effects |
| `src/solvers.jl` | Solver declarations and missing numerical-control validation |
| `src/solvers/erk4ip.jl` | Stages/FSAL/error control, cached buffers, unused wrapper, no-progress risk |
| `src/solvers/ssfm.jl` | Euler nonlinear update, step/save coupling, storage/noise |
| `src/solvers/ssfm_vectorial.jl` | Same order issue; vector buffer/output handling |
| `src/solvers/adaptive_ssfm.jl` | Phase control normalization, allocation, save flag, endpoint/failure behavior |
| `src/solver.jl` | Public dispatch, cascade conversion, RNG/sweep/result types, photon warning |
| `src/elements.jl` | Amp/attenuator factors, filter ordering, PMD/Jones construction and RNG field |
| `src/analysis.jl` | All metrics/noise/coherence/diagnostics; photon fallback, FROG axis, tracking/root semantics |
| `src/fibers.jl` | All catalog/preset/gas/capillary branches, provenance/validity and derivative conditioning |
| `src/conversions.jl` | Five conversion functions, units and domain edge cases |
| `src/recipes.jl` | Four-panel recipe, array copies, missing spectrum and zero normalization |

Repository `git status --short` was clean both before and after the audit execution. Changes are limited to audit artifacts under `/tmp`.
