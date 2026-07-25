```@meta
CurrentModule = JuGNLSE
```

# Physics Background

This page summarizes the physical models implemented in JuGNLSE.jl.

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

## Interaction Picture

All solvers work in the **interaction picture** — the simulation variable ``U = e^{-D z} A`` eliminates the linear dispersion operator from the equation of motion. This allows the solver to take larger steps in weakly nonlinear regimes and enables high-accuracy adaptive stepping.

## References

- G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019)
- J. M. Dudley, G. Genty & S. Coen, Rev. Mod. Phys. **78**, 1135 (2006)
- Q. Lin & G. P. Agrawal, Opt. Lett. **31**, 3086 (2006)
- D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B **19**, 2886 (2002)
- W. H. Blow & D. Wood, IEEE J. Quantum Electron. **25**, 2665 (1989)
