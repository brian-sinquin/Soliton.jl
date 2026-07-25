# Example 7: Gas-Filled Hollow-Core PCF & Pressure-Tuned Solitons

This example demonstrates pulse propagation and pressure-tuned zero-dispersion wavelength ($\lambda_{\text{ZD}}$) shifting in an Argon-filled Hollow-Core Photonic Crystal Fiber (HC-PCF).

---

## 🔬 Literature Comparison & Physics

Hollow-core photonic crystal fibers (HC-PCF) guide optical pulses through a central gas-filled core (P. St.J. Russell et al., *Nat. Photonics* 8, 278 (2014)). Unlike solid silica fibers, gas guidance provides:
1. **Extremely Low Kerr Nonlinearity**: $n_2 \approx 10^{-23}\text{ m}^2/\text{W/bar}$ ($>1000\times$ smaller than silica).
2. **Pressure-Tunable Dispersion**: Varying gas pressure $P$ [bar] shifts the zero-dispersion wavelength $\lambda_{\text{ZD}}$ across the visible and near-infrared spectral range (Börzsönyi et al., *Opt. Express* 21, 21086 (2013)).

---

## 💻 Julia Code

```@example ex7
using JuGNLSE

grid = create_grid(2^13, 15e-12, 800e-9)
pulse = sech_pulse(grid, 50.0e3, 30e-15)

hcf = HollowCoreFiber(
    radius = 15e-6,      # 15 μm core radius (30 μm core diameter)
    gas = :Ar,           # Argon gas
    pressure = 3.0,      # 3 bar
    length = 0.5,        # 0.5 m length
    lambda0 = 800e-9
)

params = SimParams(; medium=hcf, raman_model=nothing, z_saves=100)
sol = solve(pulse, params; progress=false)

E_in = pulse_energy(pulse)
E_out = pulse_energy(Pulse(sol))
println("Input Energy:  ", round(E_in * 1e9, digits=3), " nJ")
println("Output Energy: ", round(E_out * 1e9, digits=3), " nJ")
```

```@example ex7; hide = true
using Plots
gr()

wl_nm = 2π * 2.99792458e8 ./ grid.W .* 1e9
psd_db = 10 .* log10.(max.(1e-10, abs2.(Pulse(sol).AW)))

p1 = plot(grid.t .* 1e12, abs2.(Pulse(sol).At), label="Output Pulse (Argon 3 bar)", xlabel="Time (ps)", ylabel="Power (W)", color=:dodgerblue, lw=1.5)
p2 = plot(wl_nm, psd_db, label="Output Spectrum", xlabel="Wavelength (nm)", ylabel="PSD (dB)", xlims=(500, 1100), color=:teal, lw=1.5)
plot(p1, p2, layout=(2, 1), size=(800, 500), plot_title="Hollow-Core PCF Soliton Dynamics")
```
