# Example 9: Femtosecond EDFA Pulse Amplification

This example models pulse amplification, gain saturation, and Amplified Spontaneous Emission (ASE) noise build-up in an Erbium-Doped Fiber Amplifier (EDFA).

---

## 🔬 Physics Background

In active rare-earth fiber amplifiers (EDFA at 1550 nm, YDFA at 1064 nm), energy extraction from inverted Erbium ions leads to dynamic gain saturation along the fiber length (G. P. Agrawal, *Nonlinear Fiber Optics*, Ch. 11):

$$g(z) = \frac{g_0}{1 + E_{\text{pulse}}(z) / E_{\text{sat}}}$$

Simultaneously, spontaneous decay introduces quantum ASE noise:

$$S_{\text{ASE}}(\omega) = n_{\text{sp}} \cdot \hbar \omega_0 \cdot \left(e^{g \Delta z} - 1\right), \quad n_{\text{sp}} = \frac{10^{F_{\text{dB}}/10}}{2}$$

---

## 💻 Julia Code

```julia
using JuGNLSE
using Plots

# 1. Setup grid and input pulse (1550 nm, 100 fs, 50 W peak power)
grid = create_grid(2^13, 10e-12, 1550e-9)
pulse = gaussian_pulse(grid, 50.0, 100e-15)

# 2. Define Erbium-Doped Active Fiber Amplifier (EDFA)
edfa = AmplifyingMedium(
    length = 2.0,           # 2 m active fiber
    gamma = 0.0012,         # 1.2 /W/km nonlinearity
    g0_db = 12.0,           # +12 dB/m small-signal gain
    Esat = 2.0e-6,          # 2 μJ saturation energy
    noise_figure_db = 4.5,  # 4.5 dB Noise Figure
    betas = [-22.0e-27],    # anomalous dispersion
    lambda0 = 1550e-9
)

# 3. Simulate propagation
params = SimParams(; medium=edfa, raman_model=nothing, z_saves=100)
sol = solve(pulse, params)

# 4. Analyze gain
E_in = pulse_energy(pulse)
E_out = pulse_energy(Pulse(sol))
gain_db = 10 * log10(E_out / E_in)

println("Input Energy:   ", round(E_in * 1e12, digits=2), " pJ")
println("Output Energy:  ", round(E_out * 1e9, digits=3), " nJ")
println("Amplifier Gain: ", round(gain_db, digits=2), " dB")
```
