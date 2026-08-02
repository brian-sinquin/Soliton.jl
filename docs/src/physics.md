```@meta
CurrentModule = GNLSE
```

# Physics Background

This page summarizes the physical models implemented in GNLSE.jl.

## The Generalized Nonlinear Schrödinger Equation

The GNLSE governing the evolution of the pulse envelope ``A(z, t)`` in the retarded time frame is:

```math
\frac{\partial A}{\partial z} = \underbrace{\sum_{n \geq 2} \frac{i^{n+1}}{n!} \beta_n \frac{\partial^n A}{\partial t^n} - \frac{\alpha}{2} A}_{\text{linear: dispersion + loss}}
+ \underbrace{i \gamma \left(1 + \frac{i}{\omega_0} \frac{\partial}{\partial t}\right) \left[ A(z,t) \int_{-\infty}^{\infty} R(t') |A(z,t-t')|^2 \, dt' \right]}_{\text{nonlinear: Kerr + Raman + self-steepening}}
```

where the nonlinear response function is:

```math
R(t) = (1 - f_R) \delta(t) + f_R h_R(t)
```

and ``f_R \approx 0.18`` is the fractional Raman contribution.

## Dispersion

In the frequency domain (interaction picture), the linear operator is:

```math
D(\omega) = i B(\omega) - \frac{\alpha}{2}
```

where ``B(\omega - \omega_0)`` is the propagation-constant deviation from the reference frame, computed from one of the available dispersion models:

- **Taylor expansion** — `TaylorDispersion(betas)`:
  ``B(V) = \beta_1 V + \sum_{n \geq 2} \frac{\beta_n}{n!} V^n``

- **Tabulated dispersion** — `TabulatedDispersion(detuning, beta)`:
  Linearly interpolated from measured data.

- **Sellmeier model** — `SellmeierDispersion(B, C)`:
  ``n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}``

## Kerr Nonlinearity

The instantaneous Kerr contribution gives self-phase modulation (SPM). In the frequency domain (after ``\mathcal{F}^{-1}`` of the time-domain product):

```math
N_{\rm Kerr} = i \gamma \frac{\omega}{\omega_0} \mathcal{F}^{-1}\!\left[ |A(t)|^2 A(t) \right]
```

The ratio ``\omega/\omega_0`` is the **self-steepening shock term** (enabled via `self_steepening=true`).

The nonlinear coefficient ``\gamma`` can be:
- **Constant** — a scalar value `γ` [1/(W·m)]
- **z-dependent** — a function `γ(z)` for tapered or graded fibers
- **Frequency-dependent** — `FrequencyDependentNonlinearity` or derived from effective mode area via `NonlinearityFromEffectiveArea`

## Raman Scattering

Three models for the Raman response ``h_R(t)`` are available:

| Model | Reference |
|:---|:---|
| `BlowWood()` | Blow & Wood (1989) — two-component exponential |
| `LinAgrawal()` | Lin & Agrawal (2006) — fits measured fused-silica data |
| `Hollenbeck()` | Hollenbeck & Cantrell (2002) — most accurate for silica |

## Self-Steepening

Enabling `self_steepening=true` replaces the constant ``\gamma`` prefactor with the frequency-dependent ``\gamma \omega/\omega_0``, which accounts for the intensity-dependent group velocity and produces the optical shock effect important for sub-100 fs pulses.

## Birefringent Media (Coupled GNLSE)

For birefringent fibers or waveguides, two orthogonal polarization envelopes ``A_x`` and ``A_y`` are coupled:

```math
\frac{\partial A_x}{\partial z} = D_x A_x + i \gamma \left(|A_x|^2 + \frac{2}{3}|A_y|^2 \right) A_x + \frac{i \gamma}{3} A_y^2 A_x^* e^{-2i\Delta\beta_0 z}
```
```math
\frac{\partial A_y}{\partial z} = D_y A_y + i \gamma \left(|A_y|^2 + \frac{2}{3}|A_x|^2 \right) A_y + \frac{i \gamma}{3} A_x^2 A_y^* e^{+2i\Delta\beta_0 z}
```

where ``\Delta\beta_0 = \beta_{0x} - \beta_{0y}`` is the birefringence phase mismatch, and ``D_x``, ``D_y`` are independent dispersion operators including any group-velocity mismatch (walk-off) via ``\beta_1``.

## Active Amplifying Fibers (EDFA / YDFA)

Active rare-earth doped fibers experience single-pass amplification with dynamic gain saturation:

```math
g(z, \omega) = \frac{g_0(\omega)}{1 + E_{\text{pulse}}(z) / E_{\text{sat}}}
```

where $E_{\text{pulse}}(z) = \int |A(z,t)|^2 dt$ is the local pulse energy and $E_{\text{sat}}$ is the saturation energy. Spontaneous emission quantum noise (ASE) is seeded at each step:

```math
S_{\text{ASE}}(\omega) = n_{\text{sp}} \cdot \hbar \omega_0 \cdot \left(e^{g(z) \Delta z} - 1\right), \qquad n_{\text{sp}} = \frac{10^{F_{\text{dB}}/10}}{2}
```

where $F_{\text{dB}}$ is the amplifier Noise Figure in dB.

## Gas-Filled Hollow-Core PCF

Gas guidance in hollow-core anti-resonant / photonic crystal fibers (HC-PCF) combines:
1. **Marcatili-Schmeltzer Capillary Loss** (opt-in via `confinement_loss=true` in
   [`HollowCoreFiber`](@ref); `false` by default):
   ```math
   \alpha(\lambda) = \left( \frac{u_{01}}{2\pi} \right)^2 \frac{\lambda^2}{a^3} \frac{\nu^2 + 1}{\sqrt{\nu^2 - 1}}
   ```
   where $a$ is the core radius and $\nu$ is the cladding index ratio. This is the
   bare single-wall thick-capillary bound — it excludes the anti-resonant
   wall-thickness transmission-window term that real negative-curvature HC-PCF
   use to suppress loss by orders of magnitude, so at typical HC-PCF core radii
   (tens of μm) it substantially *overestimates* loss relative to measured
   fibers. Treat it as a conservative bound, not a quantitative prediction.
2. **Pressure-Dependent Gas Sellmeier**:
   ```math
   n(\lambda, P) = 1 + P \cdot \frac{C_1}{C_2 - \lambda^{-2}}
   ```
   with coefficients from Börzsönyi et al. (2013) for noble gases (`:Ar`, `:Ne`, `:Kr`, `:Xe`) and molecular gases (`:H2`, `:N2`).
3. **Molecular Gas Raman Response**: Rotational ($S(1)$ shift $17.6\text{ THz}$) and vibrational ($Q(1)$ shift $124.6\text{ THz}$) Raman lines for $\text{H}_2$ and $\text{N}_2$.

## Semiconductor Photonics & TPA / Free Carriers

In silicon nanowires, Germanium, and GaAs PIC waveguides, two-photon absorption (TPA) and free-carrier dynamics modify pulse propagation:

1. **Two-Photon Absorption (TPA)**:
   ```math
   \frac{\partial A}{\partial z} \Big|_{\text{TPA}} = -\frac{\alpha_2}{2 A_{\text{eff}}} |A|^2 A
   ```
2. **Free-Carrier Dynamics**:
   Carrier density $N_c(t)$ evolves via the TPA rate equation:
   ```math
   \frac{d N_c}{dt} = \frac{\alpha_2}{2 \hbar \omega_0 A_{\text{eff}}^2} |A(t)|^4 - \frac{N_c(t)}{\tau_c}
   ```
   producing Free-Carrier Absorption (FCA loss $\sigma_{\text{FCA}} N_c$) and Free-Carrier Refraction (FCR index blue-shifting $-k_{\text{FCR}} N_c$).

## Interaction Picture

All solvers work in the **interaction picture** — the simulation variable ``U = e^{-D z} A`` eliminates the linear dispersion operator from the equation of motion. This allows the solver to take larger steps in weakly nonlinear regimes and enables high-accuracy adaptive stepping.

## References

- G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019)
- J. M. Dudley, G. Genty & S. Coen, Rev. Mod. Phys. **78**, 1135 (2006)
- P. St.J. Russell et al., Nat. Photonics **8**, 278 (2014)
- L. Yin & G. P. Agrawal, Opt. Lett. **32**, 2031-2033 (2007)
- A. Börzsönyi et al., Opt. Express **21**, 21086 (2013)
- Q. Lin & G. P. Agrawal, Opt. Lett. **31**, 3086 (2006)
- D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B **19**, 2886 (2002)
- W. H. Blow & D. Wood, IEEE J. Quantum Electron. **25**, 2665 (1989)
