# Semiconductor Waveguides & Silicon Photonics

GNLSE supports semiconductor optical waveguides ([`SemiconductorMedium`](@ref)) with Two-Photon Absorption (TPA) and Free-Carrier Dynamics (FCA, FCR, carrier lifetime $\tau_c$).

---

## ⚡ Physics Model

In semiconductor photonic integrated circuits (SOI nanowires, Germanium, GaAs), high peak intensity generates free electron-hole pairs via Two-Photon Absorption ($\alpha_2$):

$$\frac{\partial N_c(t)}{\partial t} = \frac{\alpha_2}{2 \hbar \omega_0 A_{\text{eff}}^2} |A(t)|^4 - \frac{N_c(t)}{\tau_c}$$

The total optical non-linear envelope equation includes:
- **Kerr Non-linearity**: $i \gamma |A(t)|^2 A(t)$
- **Two-Photon Absorption (TPA)**: $-\frac{\alpha_2}{2 A_{\text{eff}}} |A(t)|^2 A(t)$
- **Free-Carrier Absorption (FCA)**: $-\frac{\sigma_{\text{FCA}} N_c(t)}{2} A(t)$
- **Free-Carrier Refraction (FCR / Plasma Effect)**: $-i \frac{\omega_0}{c} k_{\text{FCR}} N_c(t) A(t)$

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
