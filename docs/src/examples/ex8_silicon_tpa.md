# Example 8: Silicon Photonics (TPA & Free-Carrier Blue-Shift)

This example models non-linear pulse propagation in a Silicon-on-Insulator (SOI) nanowire, demonstrating Two-Photon Absorption (TPA) and Free-Carrier-induced spectral blue-shifting.

---

## 🔬 Literature Comparison & Physics

In sub-micron silicon photonic integrated circuits (PICs), intense optical pulses at 1550 nm experience strong Kerr self-phase modulation alongside Two-Photon Absorption ($\alpha_2 \approx 5\text{ cm/GW}$) and TPA-generated free carrier dynamics (L. Yin et al., *Opt. Express* 15, 13833 (2007)):

1. **Two-Photon Absorption (TPA)**: Non-linear attenuation scaling with $|A|^4 / A_{\text{eff}}$.
2. **Free-Carrier Refraction (FCR)**: Generated free electron-hole pairs decrease the refractive index ($n_{FC} = -k_{\text{FCR}} N_c$), shifting the pulse spectrum towards shorter wavelengths (blue-shift).

---

## 💻 Julia Code

```julia
using JuGNLSE
using Plots

# 1. Setup grid and input pulse (1550 nm, 2.0 ps, 30 W peak power)
grid = create_grid(2^12, 40e-12, 1550e-9)
pulse = gaussian_pulse(grid, 30.0, 2.0e-12)

# 2. Silicon-on-Insulator (SOI) Nanowire Waveguide
soi = SemiconductorMedium(
    length = 0.01,        # 1 cm waveguide length
    gamma = 300.0,        # 300 /W/m Kerr parameter
    alpha2 = 5.0e-12,     # 5 cm/GW TPA parameter
    Aeff = 0.1e-12,       # 0.1 μm² effective modal area
    tau_c = 1.0e-9,       # 1 ns carrier recombination lifetime
    betas = [-1000e-27],  # anomalous dispersion
    lambda0 = 1550e-9
)

# 3. Simulate propagation
params = SimParams(; medium=soi, raman_model=nothing, z_saves=100)
sol = solve(pulse, params)

# 4. Energy attenuation check
E_in = pulse_energy(pulse)
E_out = pulse_energy(Pulse(sol))

println("TPA Non-linear Transmission: ", round(E_out / E_in * 100, digits=1), "%")
```
