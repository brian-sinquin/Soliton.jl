# 🚀 Feature Proposals & Future Roadmap for JuGNLSE.jl

A comparative study of leading optical pulse propagation packages (**`laserfun`**, **`gnlse-python`**, **`PyNLSE`**, **`NLSE.py`**, and commercial tools like **Optiwave** / **RP Fiber Power**) was conducted to identify high-value feature enhancements for **JuGNLSE.jl**.

This document outlines key proposed features categorized into **Physics Models**, **Solvers & GPU Acceleration**, **Analysis & Signal Diagnostics**, **Pulse Utilities & Presets**, and **Visualization**.

---

## 📋 Summary Matrix of Proposed Features

| Category | Feature Name | Description | Reference / Target Package | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Physics** | **Amplifying Fiber (Gain Dynamics)** | Rate-equation active fibers (EDFA/YDFA/TDFA) with gain saturation $g(z, \omega)$ & ASE noise | `laserfun`, Agrawal Ch. 5 | Medium |
| **Physics** | **Silicon / Semiconductor Physics** | Two-Photon Absorption (TPA) & Free-Carrier Dynamics (FCA, FCR, carrier lifetime $\tau_c$) | `PyNLSE`, Agrawal Ch. 12 | Medium |
| **Physics** | **Gas-Filled HC-PCF Media** | Gas pressure $P$-dependent dispersion & molecular Raman response ($H_2$, $N_2$, noble gases) | `laserfun`, Dudley et al. | Medium |
| **Physics** | **Longitudinal Fiber Tapering** | $z$-dependent core geometry $d(z) \to \beta_n(z), A_{\text{eff}}(z)$ for tapered fibers | `gnlse-python` | Low |
| **Solvers** | **Adaptive Step SSFM** | Non-linear phase error control ($\Delta \phi_{\text{max}} = \gamma P_{\text{max}} \Delta z \le \phi_{\text{tol}}$) | Standard SSFM literature | Low |
| **Solvers** | **GPU Acceleration (`CUDA.jl`)** | Zero-copy GPU array propagation using `CUDA.jl` for $N = 2^{16} - 2^{18}$ grids | Julia Ecosystem | Medium |
| **Analysis** | **Spectrogram & Wigner Distribution** | Short-Time Fourier Transform (STFT) & Wigner-Ville time-frequency representations | `laserfun`, Dudley | Low |
| **Analysis** | **FROG Trace Simulator** | SHG-FROG, PG-FROG, and THG-FROG spectrogram map generation | Experimental Optics | Low |
| **Analysis** | **Soliton Fission & SSFS Engine** | Automated tracking of soliton fission, order $N(z)$, and shift rate $d\Omega/dz$ | `gnlse-python` | Medium |
| **Analysis** | **Distance-Resolved 2D Coherence** | Spatial-spectral coherence map $g_{12}^{(1)}(z, \omega)$ evolution | Dudley & Coen (2002) | Low |
| **Utilities**| **Commercial Fiber Library** | Built-in presets for standard fibers (SMF-28, NL-PM-750, LMA-8, Thorlabs PM-780) | `laserfun` | Low |
| **Utilities**| **OSA / FROG Data Importer** | Import measured optical spectrum and phase profiles (`.csv`, `.dat`, `.txt`) | `laserfun` | Low |
| **Utilities**| **Units & Optical Conversion Toolkit**| $D [\text{ps/(nm·km)}] \leftrightarrow \beta_2 [\text{ps}^2/\text{km}]$ and Stokes vector parameters | Optics Standard | Low |
| **Plots**    | **`Plots.jl` Plot Recipes** | `@recipe` for automatic 4-panel dashboard of `Solution` & `VectorialSolution` | Julia Ecosystem | Low |

---

## 🔍 Detailed Proposal Specifications

### 1. 🏛️ Physics & Medium Enhancements

#### A. Active Fiber Model (Amplifiers & Gain Saturation)
Implement an active fiber medium (`AmplifyingMedium`) with gain saturation and frequency dependence:
$$g(z, \omega) = \frac{g_0(\omega)}{1 + E_{\text{pulse}}(z) / E_{\text{sat}}}$$
- Includes parabolic / Lorentzian gain spectrum $g_0(\omega)$.
- Injects Amplified Spontaneous Emission (ASE) noise along propagation.

#### B. Semiconductor Waveguide Model (TPA & Free Carriers)
For silicon, germanium, or GaAs photonic integrated circuits (PICs):
$$\frac{\partial A}{\partial z} = -\frac{\alpha_0 + \alpha_2 |A|^2}{2} A + i \gamma |A|^2 A - \frac{\sigma_{\text{FCA}} + 2 i k_0 k_{\text{FCR}}}{2} N_c A$$
$$\frac{d N_c(t)}{dt} = \frac{\alpha_2}{2 \hbar \omega_0 A_{\text{eff}}^2} |A(t)|^4 - \frac{N_c(t)}{\tau_c}$$

#### C. Gas-Filled Hollow-Core Fiber (HC-PCF)
Support gas pressure scaling $P$ [bar] for noble gases (Ar, Xe, Kr) and molecular gases ($H_2$, $N_2$):
- Pressure-dependent Sellmeier refraction index $n(\lambda, P)$.
- Rotational and vibrational Raman response functions for molecular gases.

#### D. Tapered Fiber Geometry
Support $z$-dependent medium functions:
$$\beta_n(z) = \beta_n^0 \cdot f_{\text{taper}}(z), \quad \gamma(z) = \frac{\gamma_0}{f_{\text{taper}}(z)}$$

---

### 2. ⚡ Solvers & Computation

#### A. Adaptive SSFM (Phase-Controlled Step Size)
Implement variable step-size SSFM:
$$\Delta z_{k+1} = \Delta z_k \cdot \min\left(1.5, \frac{\phi_{\text{tol}}}{\gamma P_{\text{max}} \Delta z_k}\right)$$
Provides significant speedups when pulse power decays due to loss or pulse broadening.

#### B. GPU Acceleration (`CUDA.jl` Array Extensions)
Leverage JuGNLSE's template-based `_to_device` abstraction in `PhysicsModel`:
```julia
# Move pulse to GPU
vpulse_gpu = VectorialPulse(CuArray(vpulse.At), grid)
sol_gpu = solve(vpulse_gpu, params)
```

---

### 3. 📊 Analysis & Diagnostic Tools

#### A. Spectrogram (STFT) & Wigner-Ville Distribution
Compute time-frequency intensity maps:
$$S(t, \omega) = \left| \int_{-\infty}^{\infty} A(t') g(t' - t) e^{-i \omega t'} dt' \right|^2$$
Using Gaussian gate function $g(t) = \exp(-t^2 / 2 \tau_g^2)$.

#### B. FROG Trace Simulation
Generate synthetic Frequency-Resolved Optical Gating (FROG) traces:
$$I_{\text{FROG}}(\omega, \tau) = \left| \int_{-\infty}^{\infty} A(t) A(t - \tau) e^{-i \omega t} dt \right|^2 \quad (\text{SHG-FROG})$$

#### C. Automated Soliton Analysis & Fission Tracking
- Extract individual fundamental solitons during fission.
- Calculate soliton order $N(z)$ along propagation.
- Measure Soliton Self-Frequency Shift (SSFS) drift rate $d\Omega_{\text{soliton}} / dz$ [THz/m].

#### D. Distance-Resolved Coherence $g_{12}^{(1)}(z, \omega)$
Evaluate ensemble coherence matrix at every saved distance $z$ to map the exact location of coherence degradation.

---

### 4. 🛠️ Utilities, Presets & Plotting

#### A. Pre-packaged Commercial Fiber Library (`FiberLibrary`)
Provide instant access to common commercial fibers:
```julia
medium = FiberLibrary["Thorlabs_NL_PM_750"](length=0.2)
medium_smf = FiberLibrary["Corning_SMF28"](length=100.0)
```

#### B. Experimental Data Importer
Load FROG or Optical Spectrum Analyzer (OSA) text / CSV exports:
```julia
pulse = load_pulse_from_file("experimental_frog_trace.csv", grid)
```

#### C. Units & Conversions
```julia
beta2 = dispersion_D_to_beta2(D=15.0, lambda0=1550e-9) # ps/(nm*km) -> s^2/m
```

#### D. Native Plot Recipes (`Plots.jl`)
```julia
using Plots
plot(solution) # Automatically creates 4-panel dashboard: At heat map, AW heat map, Temporal slice, Spectral slice
```
