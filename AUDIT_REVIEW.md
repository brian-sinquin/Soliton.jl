# Review of bugfix/audit-critical-fixes

Reviewed 2026-09-06: branch `3e381e5` versus master `4e734a1`. Read the branch diff, modified source files, and the constructors, analysis functions, dispersion builders and propagation paths needed to trace the conventions. This review concerns C2, C3, C4 and H4; the original AUDIT.md describes a broader backlog and is historical, not a statement of current test/CI configuration.

## Findings and corrections

### Critical: C4 introduces incorrect Raman-amplifier saturation

Removing the extra `inv_w0` is correct for the Kerr/Raman polarization. However, `_amplifying_spm_raman` still included `delta_g*u` inside the transform multiplied by `gamma_W`. For numeric gamma with shock disabled this multiplies the amplitude saturation correction by omega0, approximately 1.22e15 at 1550 nm. With shock it also introduces spurious frequency dependence into gain. With a frequency-dependent nonlinear coefficient it incorrectly makes saturation depend on Kerr strength. The original zero-gain, no-Raman regression cannot detect any of these failures.

Fixed by transforming the Kerr/Raman polarization and gain correction separately, weighting only the former by `gamma_W`. Reusing `buf_t2` is safe: the Raman convolution has already been consumed before this buffer receives the gain spectrum. This adds one inverse FFT to the Raman RHS, matching the non-Raman branch's separation; it adds no workspace array.

The contract is the product, not a universal unit for each factor:

| gamma input | eval_gamma result | gamma_W |
| --- | --- | --- |
| Number / ConstantNonlinearity | gamma / omega0 | omega0 without shock; omega with shock |
| Function of z | gamma(z) / omega0 | same |
| FrequencyDependentNonlinearity | 1 | supplied gamma(omega) |
| NonlinearityFromEffectiveArea | 1 | n2 * omega / (c * Aeff(omega)) |

The product has units 1/(W m). Explicit spectral models supply their full weighting and ignore the shock flag by the existing contract. The effective-area implementation is the package's simple frequency-dependent coefficient approximation, not the full transformed-envelope mode-profile theory.

For the optics transform `F = ifft`, the required nonlinear RHS is

```
i * gamma_z * gamma_W .* F(u .* ((1-fr)*abs2(u) + fr*(hR convolved with abs2(u))))
    + F(delta_g * u)
```

Here `delta_g = -g0*E/(2*(Esat+E))` has units 1/m. Together with the linear operator's `g0/2`, the net amplitude gain is `g0*Esat/(2*(Esat+E))`. Since delta_g is a scalar at each RHS evaluation, `F(delta_g*u) = delta_g*F(u)`. Adding untransformed `delta_g*u` to a spectral RHS is incorrect. The reviewed branch's **non-Raman** path already performed the correct transform; the remaining defect was in its Raman counterpart.

This placement of the shock factor on nonlinear polarization follows the standard GNLSE equation documented by [gnlse-python](https://gnlse.readthedocs.io/en/latest/gnlse.html), whose implementation derives from Dudley/Taylor. The distinction from full mode-profile dispersion is described in its [nonlinear-coefficient documentation](https://gnlse.readthedocs.io/en/latest/nonlinearity.html). Exact SPM and gain-energy test oracles follow directly by reducing the equation and integrating the package's stated gain law.

### High: C2 leaves no-spectrum cascades broken

The `ifftshift` itself is correct. All scalar propagation implementations save `fftshift(U)`; pulse generators use `ifft(At)`; dispersion/nonlinear operators use FFT order; pulse spectral analysis shifts to match monotonic `grid.V`. Vectorial propagation saves shifted spectra per polarization, while `VectorialPulse(sol)` reconstructs spectra from At along dimension 1.

But `Pulse(sol)` still indexed an empty AW when ERK4IP or SSFM used `save_freq=false`. Fixed by reconstructing `ifft(At)` when saved spectra are absent, retaining the unshift for saved spectra. This repairs an existing advertised cascade path. Corrected the contradictory Pulse docstring that said `N*ifft(At)` and assigned continuous-transform units to the unscaled discrete spectrum.

The original identity test was useful but did not test dispersion, multiple stages, alternate solvers or missing spectra. Added complex-field cascade equivalence and transform invariants for all three scalar solvers and both save settings. These are linear dispersive cases so unrelated nonlinear integration errors cannot conceal an ordering defect.

### C3: correct scalar and vectorial filter fix

Transfer callbacks receive absolute angular frequency in rad/s, sampled on monotonic grid.W. `ifftshift` aligns these samples with Pulse.AW. Julia broadcasts an N-vector across the columns of an N-by-2 matrix, so both polarizations receive the correct same scalar transfer; each is transformed back separately. No polarization mixing is introduced.

Repository usages specify ordinary carrier-centered passbands/Gaussians and are compatible. Constant transmission is unchanged. External callbacks deliberately compensating for the old bin permutation would need correction; that compensation was not the documented contract. The previous C2 and C3 ordering errors could partially mask each other, making separate tests essential.

The original >99% energy check detects the gross permutation but misses signed-frequency and phase errors. Added exact DC/positive/negative DFT-bin tones, asymmetric complex transmission, distinct polarization frequencies, field/energy transmission, and the vector FFT invariant. Documented the callback's units and field-transmission semantics.

### H4: justified implementation restriction

Odd FFT lengths are mathematically legitimate; even N is not a GNLSE requirement. However, this constructor previously generated N-1 frequency entries for odd N, and dispersion builders use fftshift where the inverse mapping is needed (equivalent only for even N). Thus the rejection removes malformed outputs, not a previously correct odd-grid propagation path. Supporting odd grids would require coordinated frequency-grid and shift changes beyond this validation fix. Direct manually constructed Grid objects remain outside this constructor's validation.

Retained even N >= 2, and documented it. Existing validation tests cover failures and array lengths; added comparison against FFTW frequency bins and exact dispersive phase for every bin of N=2,6,10, including Nyquist. Even non-power-of-two grids remain supported. The inclusive time grid retains FFT period N*dt, not time_window; no grid-convention migration was made.

## Regression evidence and scope

New checks cover all five scalar gamma representations, zero/nonzero scalar gain, Raman on/off and shock on/off; include an independent Kerr spectral-coefficient oracle, exact SPM phase, and integrated saturated gain at multiple z values. Deterministic gain propagation uses the production integrator with a test-local PhysicsModel lacking the ASE metadata key; this makes the existing noise hook a no-op without adding a public feature. A 0 dB noise figure alone would not disable ASE.

The first targeted run against the reviewed branch reproduced the missing-spectrum BoundsError and ten Raman/gain normalization assertion failures. An initial test-only attempt to disable ASE with a negative noise figure was rejected by the constructor and was replaced by the test-local deterministic model described above. Tests do not weaken existing tolerances or regenerate reference fixtures.

Pre-existing issues outside these four repairs remain, including spectral/function g0 bypassing saturation and ASE (AUDIT H2), AdaptiveSSFM ignoring save_freq=false (H5), and first-order nonlinear SSFM updates (C1). The cascade tests validate conversion with either actual save behavior; they do not claim to fix AdaptiveSSFM's storage flag. Passing this suite is not a blanket validation of those paths.

## Full-suite results

- Original branch: `julia --project -e 'using Pkg; Pkg.test()'`: **384/384 passed**, 2m59.3s.
- Final branch: `GKSwstype=100 julia --project -e 'using Pkg; Pkg.test()'`: **503/503 passed** (119 added assertions), 3m19.8s.
- Environment: Julia 1.12.7, Linux aarch64. Includes every test file in the current runner, including numerical reference fixtures and plot recipes. No Julia-version/architecture CI matrix or documentation build was run.
- Both runs emitted the existing grid/carrier mismatch and 4.61% photon-drift warnings; these are not made acceptable by the passing assertion count.
