# PR: JuGNLSE v0.2.0 Major Feature Release & Precompiled Pipeline

## 🚀 Summary of Changes

This PR introduces **JuGNLSE v0.2.0**, bringing comprehensive physical models for telecom/specialty fibers, active amplifiers, gas-filled hollow-core PCFs, silicon photonics, coupled vectorial propagation, solver abstractions, cascaded multi-stage propagation, automated per-commit benchmarking, executable documentation with inline figures, and tag-triggered precompiled binary releases.

---

## 📦 Key New Features

### 1. 🏗️ Solver Abstraction & SSFM Solver ([`src/solvers/`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/solvers/))
- **`GNLSESolver` Abstract Type**: Standardized solver interface supporting modular solver backends.
- **Adaptive `ERK4IP`**: Refactored Interaction Picture Runge-Kutta 4(3) with adaptive local error control.
- **Fixed-Step `SSFM`**: Symmetric Split-Step Fourier Method solver with constant step size control.

### 2. ⚡ Cascaded Propagation & Lumped Elements ([`src/elements.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/elements.jl))
- **Multi-Stage Cascading**: `solve(pulse, stages)` propagating across cascaded fiber sections, amplifiers, and filters.
- **Lumped Elements**: Instantaneous optical components:
  - `Amplifier(gain_db)` (Lumped optical amplifier)
  - `Attenuator(loss_db)` (Lumped optical attenuator)
  - `Filter(filter_type, center_wavelength, bandwidth)` (Spectral filter)
- **Piping Operator (`|>`)**: Expressive pipeline syntax:
  ```julia
  sol = pulse |> fiber1 |> Amplifier(10.0) |> Filter(:gaussian, 1550e-9, 10e-9) |> fiber2
  ```

### 3. 📐 Sellmeier & Frequency-Dependent Nonlinearity ([`src/dispersion.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/dispersion.jl), [`src/nonlinearity.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/nonlinearity.jl))
- **`SellmeierDispersion`**: Glass Sellmeier expansion with analytical derivatives $d^n \beta / d\omega^n$.
- **`FrequencyDependentNonlinearity`**: Frequency-dependent nonlinear coefficient $\gamma(\omega)$.
- **`NonlinearityFromEffectiveArea`**: Modal area-derived nonlinearity $\gamma(\omega) = \frac{n_2 \omega}{c A_{\text{eff}}(\omega)}$.

### 4. 🌾 Commercial Fiber Catalog & Glass Presets ([`src/fibers.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/fibers.jl))
- **Commercial Fiber Preset Library**: Preset instantiation via `commercial_fiber(name; length, lambda0, loss)` for standard telecom and specialty fibers:
  - `"Corning_SMF28"` (Standard single-mode telecom fiber)
  - `"Corning_LEAF"` (Large effective area fiber)
  - `"NKT_NL_PM_750"` (Highly-nonlinear polarization-maintaining PCF)
  - `"Thorlabs_PM780"` & `"Thorlabs_PM1550"`
  - `"NKT_LMA10"` (Large mode area silica PCF)
- **RefractiveIndex.io Glass Sellmeier Presets**:
  - `FusedSilica()` (Malitson 1965 expansion)
  - `SF6()` (Schott flint glass)
  - `SF57()` (Ultra-high nonlinearity lead-silicate flint glass)
  - `GeO2DopedSilica(mole_fraction)` (Concentration-dependent germanosilicate core scaling)

### 5. ⚡ EDFA Dynamics & ASE Quantum Noise ([`src/types.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/types.jl#L180), [`src/nonlinearity.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/nonlinearity.jl#L280))
- **`AmplifyingMedium`**: Active rare-earth-doped amplifying fibers (EDFA at 1550 nm, YDFA at 1064 nm, TDFA at 2000 nm).
- **Gain Saturation Model**:
  $$g(z, \omega) = \frac{g_0(\omega)}{1 + E_{\text{pulse}}(z) / E_{\text{sat}}}$$
- **ASE Noise Model**: Spontaneous emission quantum noise seeding derived from Noise Figure $F_{\text{dB}}$:
  $$S_{\text{ASE}}(\omega) = n_{\text{sp}} \cdot \hbar \omega_0 \cdot \left(e^{g \Delta z} - 1\right), \quad n_{\text{sp}} = \frac{10^{F_{\text{dB}}/10}}{2}$$

### 6. 🎈 Gas-Filled Hollow-Core PCF & Molecular Raman ([`src/fibers.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/fibers.jl#L180), [`src/raman.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/raman.jl#L175))
- **`HollowCoreFiber`**: Marcatili-Schmeltzer / Zeisberger capillary guidance model for gas-filled HC-PCF with pressure-dependent dispersion $\beta_n(P)$ and nonlinearity $\gamma(P)$.
- **`gas_refractive_index`**: Börzsönyi et al. (2013) Sellmeier expansions for noble gases (`:Ar`, `:Ne`, `:Kr`, `:Xe`) and molecular gases (`:H2`, `:N2`).
- **`MolecularRamanGas`**: Rotational ($S(1)$, $17.6\text{ THz}$) and vibrational ($Q(1)$, $124.6\text{ THz}$) Raman response lines for $\text{H}_2$ and $\text{N}_2$.

### 7. 🔬 Silicon Photonics & Semiconductor Waveguides ([`src/types.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/types.jl#L240), [`src/nonlinearity.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/nonlinearity.jl#L360))
- **`SemiconductorMedium`**: SOI nanowires, Germanium, and GaAs PIC waveguide physics.
- **Two-Photon Absorption (TPA)**: Non-linear attenuation scaling with $\alpha_2 |A|^2 / A_{\text{eff}}$.
- **Free-Carrier Dynamics**: Carrier generation rate equation $\frac{d N_c}{dt}$, Free-Carrier Absorption (FCA $\sigma_{\text{FCA}}$), and Free-Carrier Refraction (FCR plasma blue-shifting $k_{\text{FCR}}$).

### 8. 🌊 Vectorial Solver & Birefringent Coupling ([`src/solvers/ssfm_vectorial.jl`](file:///c:/Users/brian/Documents/GitHub/JuGNLSE/src/solvers/ssfm_vectorial.jl))
- **`BirefringentMedium` & `VectorialPulse`**: Fully coupled vector GNLSE solver supporting XPM, SPM, and FWM terms across fast and slow polarization axes.
- **Multi-Stage Vectorial Cascading**: `VectorialPulse` support for `solve(pulse, stages)` and lumped elements (`Amplifier`, `Attenuator`, `Filter`).

### 9. 📊 Benchmarking, Executable Documentation & JLL Release Pipeline
- **Per-Commit Benchmarks**: `BenchmarkTools.jl` suite in `benchmark/` with automated CI runner (`.github/workflows/benchmark.yml`).
- **Executable Examples**: Examples 1–9 run live during `Documenter.jl` builds with inline SVG plot rendering.
- **PR Documentation Artifacts**: Uploads `docs/build` as a GitHub Action artifact (`documentation-preview`) on PR builds.
- **Precompiled Release Pipeline**: Added `.github/workflows/compile-release.yml` for tag-triggered (`v*`) or manual (`workflow_dispatch`) precompiled JLL sysimage compilation via `PackageCompiler.jl`.

---

## 📑 Exhaustive Commit Log (All 19 Commits)

| Commit | Description |
| :--- | :--- |
| `a9d7e5a` | `docs: update PR_DESCRIPTION.md` |
| `6458810` | `docs: critical documentation review, expanded examples index, and math equations` |
| `4532f4d` | `docs: add PR_DESCRIPTION.md tracking all changelogs` |
| `8db4b9c` | `docs: list Examples 7, 8, 9 in examples index` |
| `65bac9c` | `docs & ci: executable example plot rendering, PR doc artifact upload, and precompiled JLL release workflow` |
| `c507e0d` | `docs & test: add examples 7-9 and expanded physical test coverage` |
| `4ceb780` | `feat: implement gas-filled Hollow-Core PCF (HC-PCF) and Molecular Gas Raman model` |
| `a47bb53` | `feat: implement EDFA dynamics and ASE quantum noise model` |
| `8a122b3` | `build: untrack benchmark venv` |
| `ae29341` | `feat: add Commercial Fiber Library and RefractiveIndex.io Glass Presets` |
| `c0cd972` | `feat: implement Birefringent Vectorial solver (Coupled GNLSE) support` |
| `0682742` | `feat: implement fixed-step Symmetric Split-Step Fourier Method (SSFM) solver` |
| `b2dacdd` | `docs: attach docstrings to correct bindings in solvers/erk4ip.jl to fix Documenter build` |
| `d6a0263` | `feat: implement Frequency-dependent and Effective Mode Area-based Nonlinearity models` |
| `e06f6e7` | `feat: add piping operator \|> support for SimParams and LumpedElements` |
| `855136f` | `feat: implement Cascaded Propagation API with lumped Amplifiers, Attenuators, and Filters` |
| `9760e7c` | `feat: abstract solvers into GNLSESolver interface and refactor SimParams/ERK4IP` |
| `d820fe0` | `test: add physical test verifying tapered fiber SPM phase shift against analytical solution` |
| `8ea06bb` | `feat: implement SellmeierDispersion model with analytical derivative and unit tests` |

---

## 🧪 Test Verification Results

All **253/253 unit and physical tests pass 100%**:

```
Test Summary:                                                | Pass  Total   Time
JuGNLSE.jl                                                   |  253    253  32.4s
  Unit                                                       |   56     56   3.3s
  API                                                        |   74     74  12.6s
    Medium                                                   |   10     10   0.0s
    Medium keyword constructor                               |    6      6   0.0s
    Dispersion models                                        |    4      4   0.0s
    Sellmeier dispersion                                     |    7      7   0.1s
    Raman models                                             |    6      6   0.0s
    SimParams                                                |    7      7   0.0s
    Pulses                                                   |    9      9   0.3s
    build_physics_model                                      |    6      6   0.2s
    Z-dependent gamma                                        |    5      5   2.1s
    Cascaded propagation & Lumped Elements                   |   11     11   3.4s
    Frequency-dependent and Effective-area Nonlinearity      |    3      3   6.3s
  Solvers                                                    |   42     42   0.9s
  Physics                                                    |   11     11  10.6s
  Vectorial/Birefringent Coupled Solver                      |   18     18   3.3s
  Commercial Fiber Library and Glass Presets                 |   23     23   0.4s
  Active Amplifying Fiber Dynamics (EDFA/YDFA/TDFA)          |    6      6   0.0s
  Hollow-Core PCF (HC-PCF) & Molecular Gas Raman             |   15     15   0.2s
  Semiconductor Waveguides (SOI, TPA, Free-Carrier Dynamics) |    8      8   0.2s
Testing JuGNLSE tests passed
```
