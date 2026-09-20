# Semiconductor Waveguides & Silicon Photonics

Soliton supports semiconductor optical waveguides ([`SemiconductorMedium`](@ref)) with Two-Photon Absorption (TPA), Three-Photon Absorption (3PA), and Free-Carrier Dynamics (FCA, FCR, carrier lifetime $\tau_c$).

---

## ⚡ Physics Model

In semiconductor photonic integrated circuits (SOI nanowires, Germanium, GaAs), high peak intensity generates free electron-hole pairs via Two-Photon Absorption ($\alpha_2$) and, at longer (mid-infrared) wavelengths where the photon energy drops below half the bandgap and TPA is forbidden, via Three-Photon Absorption ($\alpha_3$) instead. The two generation channels add:

$$\frac{\partial N_c(t)}{\partial t} = \frac{\alpha_2}{2 \hbar \omega_0 A_{\text{eff}}^2} |A(t)|^4 + \frac{\alpha_3}{3 \hbar \omega_0 A_{\text{eff}}^3} |A(t)|^6 - \frac{N_c(t)}{\tau_c}$$

The total optical non-linear envelope equation includes:
- **Kerr Non-linearity**: $i \gamma |A(t)|^2 A(t)$
- **Two-Photon Absorption (TPA)**: $-\frac{\alpha_2}{2 A_{\text{eff}}} |A(t)|^2 A(t)$
- **Three-Photon Absorption (3PA)**: $-\frac{\alpha_3}{2 A_{\text{eff}}^2} |A(t)|^4 A(t)$
- **Free-Carrier Absorption (FCA)**: $-\frac{\sigma_{\text{FCA}} N_c(t)}{2} A(t)$
- **Free-Carrier Refraction (FCR / Plasma Effect)**: $-i \frac{\omega_0}{c} k_{\text{FCR}} N_c(t) A(t)$

`alpha3` defaults to `0.0`, recovering pure-TPA behavior; enable it for mid-infrared silicon/germanium waveguides (roughly 2.2–4 μm) where 3PA dominates the nonlinear loss.

!!! warning "Known limitations"
    This is the standard 1D lumped-parameter TPA/3PA/free-carrier formalism (Lin, Painter & Agrawal, *Opt. Express* **15**, 16604 (2007)) and shares its usual simplifications — see the [`SemiconductorMedium`](@ref) docstring for the full list. In short: Kerr/TPA/3PA all reuse a single `Aeff` rather than distinct mode-overlap integrals per order; carrier recombination is linear only (no Auger `∝N_c³`, no diffusion); the 3PA-associated nonlinear-refraction term (Kramers-Kronig partner of `alpha3`) is omitted, consistent with TPA's own omitted refractive term; and self-steepening is applied to Kerr/TPA/3PA but *not* to FCA/FCR (their physical origins differ).

    **Validation**: TPA is cross-checked against an external package — `gnlse-python`'s Kerr-only step accepts a complex nonlinearity (`γ_Kerr + i·α₂/(2·Aeff)`), the standard literature trick for encoding TPA, letting its own independent solver validate ours (`test/test_adversarial.jl`, Scenario 7). 3PA and FCA/FCR have no such trick available (3PA needs a field term beyond cubic; FCA/FCR need an auxiliary carrier-density ODE) and no other suitable open package was found — see the docstring for what was checked (PyNLO, MEEP, Tidy3D). Both remain validated only via a dedicated ODE ground-truth integrator (`test/test_semiconductor.jl`).

---

## 💻 Usage Example

```julia
using Soliton

grid = create_grid(2^12, 20e-12, 1550e-9)
pulse = gaussian_pulse(grid, 50.0, 1.0e-12) # 50 W peak power input

# Silicon-on-Insulator (SOI) Nanowire Waveguide
soi = SemiconductorMedium(
    length = 0.01,        # 1 cm waveguide
    gamma = 300.0,        # 300 /W/m Kerr parameter
    alpha2 = 5.0e-12,     # 5 cm/GW TPA parameter at 1550 nm
    Aeff = 0.1e-12,       # 0.1 μm² modal area
    tau_c = 1.0e-9,       # 1 ns carrier recombination lifetime
    betas = [-1000e-27],  # anomalous dispersion
    lambda0 = 1550e-9
)

params = SimParams(; medium=soi, z_saves=50)
sol = solve(pulse, params)
```
