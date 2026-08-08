# Example 8: Two-Photon Absorption (TPA) — Silicon Optical Limiter

This example demonstrates the signature use of Two-Photon Absorption (TPA) in silicon photonics: **passive optical limiting**. Instead of a single fixed-power run, we sweep the input peak power across two orders of magnitude and show the resulting limiter curve — nonlinear transmission that collapses as $P_0$ grows, because TPA loss scales with intensity itself ($\propto |A|^4$) rather than being fixed.

---

## 🔬 Physics

For a silicon nanowire (SOI), the field obeys (Q. Lin, O. Painter & G. P. Agrawal, *Opt. Express* **15**, 16604 (2007); L. Yin & G. P. Agrawal, *Opt. Lett.* **32**, 2031 (2007)):

```math
\frac{\partial A}{\partial z}\Big|_{\text{TPA}} = -\frac{\alpha_2}{2 A_{\text{eff}}} |A|^2 A
```

Integrating the corresponding intensity equation $dI/dz = -\alpha_2 I^2$ for a lossless, dispersionless beam gives the classic saturable-limiter form:

```math
I_{\text{out}} = \frac{I_{\text{in}}}{1 + \alpha_2 I_{\text{in}} L / A_{\text{eff}}}
```

At low power the transmission is ≈100%; as $P_0 \to \infty$ the *transmitted* power saturates towards a constant ($\propto A_{\text{eff}}/(\alpha_2 L)$) instead of growing linearly with input — silicon TPA acts as a passive power limiter, useful for protecting downstream detectors from optical damage.

---

## 💻 Julia Code

```@example ex8
using Soliton

grid = create_grid(2^12, 40e-12, 1550e-9)

soi = SemiconductorMedium(
    length = 0.01,        # 1 cm SOI nanowire
    gamma = 300.0,        # 300 /W/m Kerr parameter
    alpha2 = 5.0e-12,     # 0.5 cm/GW TPA parameter
    Aeff = 0.1e-12,       # 0.1 μm² effective modal area
    tau_c = 1.0e-9,       # 1 ns carrier recombination lifetime
    betas = [-1000e-27],  # anomalous dispersion
    lambda0 = 1550e-9
)

powers = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]  # W

sols = solve_sweep(powers; progress=false) do P0
    return (gaussian_pulse(grid, P0, 2.0e-12), SimParams(; medium=soi, raman_model=nothing, z_saves=2))
end

transmission = Float64[]
P_out = Float64[]
for (P0, sol) in zip(powers, sols)
    Ein = pulse_energy(gaussian_pulse(grid, P0, 2.0e-12))
    Eout = pulse_energy(Pulse(sol))
    push!(transmission, 100 * Eout / Ein)
    push!(P_out, P0 * Eout / Ein)
end
nothing # hide
```

```@example ex8
using Plots # hide

p1 = plot(powers, transmission; xaxis=:log10, marker=:circle, lw=2, # hide
    xlabel="Input Peak Power P₀ [W]", ylabel="Nonlinear Transmission [%]", # hide
    label="TPA transmission", legend=:topright, title="Silicon TPA Optical Limiter") # hide

p2 = plot(powers, P_out; xaxis=:log10, marker=:square, lw=2, color=:orangered, # hide
    xlabel="Input Peak Power P₀ [W]", ylabel="Output Peak Power [W]", # hide
    label="Output power (limited)", legend=:topleft) # hide
plot!(p2, powers, powers; ls=:dash, color=:gray, label="Linear (no TPA)") # hide

plot(p1, p2; layout=(1,2), size=(950,400), dpi=300) # hide
```

---

## 📊 Expected Results

| $P_0$ [W] | Transmission | Output Power |
|:---:|:---:|:---:|
| 1 | ~74% | ~0.7 W |
| 10 | ~21% | ~2.1 W |
| 100 | ~2% | ~2.4 W |

The output power *saturates* around ~2–3 W despite a 100× increase in input — the hallmark optical-limiting behavior driven purely by TPA's quadratic intensity dependence.

!!! note "Silicon TPA is very strong"
    The TPA figure of merit for silicon, $\text{FOM} = n_2/(\alpha_2\lambda) \approx 0.4$ at 1550 nm, sits below the threshold of 1 needed for net parametric gain. This is what makes silicon waveguides poor optical amplifiers but excellent all-optical limiters — see [Example 11](ex11_silicon_3pa.md) for how this changes entirely in the mid-infrared, where 3PA (not TPA) sets the nonlinear loss.

## References

> L. Yin and G. P. Agrawal, "Impact of two-photon absorption on self-phase modulation in silicon waveguides,"
> *Opt. Lett.* **32**, 2031–2033 (2007). DOI: [10.1364/OL.32.002031](https://doi.org/10.1364/OL.32.002031)

> Q. Lin, O. J. Painter, and G. P. Agrawal, "Nonlinear optical phenomena in silicon waveguides,"
> *Opt. Express* **15**, 16604–16644 (2007). DOI: [10.1364/OE.15.016604](https://doi.org/10.1364/OE.15.016604)
