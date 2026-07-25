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

```@example ex9
using JuGNLSE

grid = create_grid(2^13, 10e-12, 1550e-9)
pulse = gaussian_pulse(grid, 50.0, 100e-15)

edfa = AmplifyingMedium(
    length = 2.0,           # 2 m active fiber
    gamma = 0.0012,         # 1.2 /W/km nonlinearity
    g0_db = 12.0,           # +12 dB/m small-signal gain
    Esat = 2.0e-6,          # 2 μJ saturation energy
    noise_figure_db = 4.5,  # 4.5 dB Noise Figure
    betas = [-22.0e-27],    # anomalous dispersion
    lambda0 = 1550e-9
)

params = SimParams(; medium=edfa, raman_model=nothing, z_saves=100)
sol = solve(pulse, params; progress=false)

E_in = pulse_energy(pulse)
E_out = pulse_energy(Pulse(sol))
gain_db = 10 * log10(E_out / E_in)
println("Input Energy:   ", round(E_in * 1e12, digits=2), " pJ")
println("Output Energy:  ", round(E_out * 1e9, digits=3), " nJ")
println("Amplifier Gain: ", round(gain_db, digits=2), " dB")
```

```@example ex9; hide = true
using Plots
gr()

energies_nJ = [pulse_energy(Pulse(sol.At[:, i], sol.AW[:, i], grid)) * 1e9 for i in 1:length(sol.Z)]
p1 = plot(sol.Z, energies_nJ, label="Pulse Energy (nJ)", xlabel="Amplifier Length z (m)", ylabel="Energy (nJ)", color=:magenta, lw=2.0)
p2 = plot(grid.t .* 1e12, abs2.(Pulse(sol).At), label="Output Amplified Pulse", xlabel="Time (ps)", ylabel="Power (W)", color=:darkgreen, lw=1.5)
plot(p1, p2, layout=(2, 1), size=(800, 500), plot_title="EDFA Femtosecond Pulse Amplification")
```
