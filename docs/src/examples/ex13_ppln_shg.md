# Example 13: PPLN Waveguide — Second-Harmonic Generation & Quasi-Phase-Matching

This example simulates second-harmonic generation (1550 nm → 775 nm) in a **periodically poled lithium niobate (PPLN)** waveguide using [`SecondOrderMedium`](@ref). It demonstrates the three defining features of a PPLN device in one script: (1) why periodic poling is needed at all — lithium niobate's *natural* birefringent phase mismatch is far too large for efficient SHG over a centimeter-scale waveguide, (2) the classic depleted-pump SHG conversion curve, and (3) the `sinc²` phase-matching acceptance bandwidth that sets a PPLN device's wavelength/temperature tolerance.

---

## 🔬 Physics & Device Parameters

Lithium niobate's natural (unpoled) phase mismatch for 1550 nm → 775 nm type-0 SHG corresponds to a coherence length of only a few microns — hopeless for a waveguide device more than a few micrometers long. Periodic poling flips the sign of $\chi^{(2)}$ every coherence length (period $\Lambda \approx 19$–$20\,\mu\text{m}$ for this wavelength in congruent LN near room temperature) so the nonlinear coupling stays in phase over the whole device length — this is exactly [`SecondOrderMedium`](@ref)'s `poling_period` field.

We use a normalized SHG efficiency $\eta_{\text{norm}} = 150\%/(\text{W}\cdot\text{cm}^2)$, representative of high-quality PPLN waveguides (Parameswaran et al., *Opt. Lett.* **27**, 43 (2002)), converted to the package's `kappa` [1/(√W·m)] via $\kappa^2\,[\text{W}^{-1}\text{m}^{-2}] = 100\,\eta_{\text{norm}}\,[\%/(\text{W}\cdot\text{cm}^2)]$.

---

## 💻 Julia Code

```@example ex13
using Soliton

grid = create_grid(2^10, 20e-12, 1550e-9)

L = 0.01                      # 1 cm PPLN waveguide
kappa = sqrt(150.0 * 100)     # from 150 %/(W*cm^2) normalized efficiency
poling_period = 19.2e-6       # typical LN 1550 -> 775 nm QPM period
deltak0_natural = 2π / poling_period   # LN's natural (unpoled) phase mismatch

cw_pulse(P0) = SecondOrderPulse(
    fill(ComplexF64(sqrt(P0)), grid.N), zeros(ComplexF64, grid.N), grid
)

function shg_efficiency(P0; deltak0=0.0, poling=0.0)
    medium = SecondOrderMedium(;
        length=L, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0],
        deltak0=deltak0, poling_period=poling, lambda0=1550e-9,
    )
    params = SimParams(; medium=medium, z_saves=2, raman_model=nothing, solver=SSFM(1e-7))
    sol = solve(cw_pulse(P0), params; progress=false)
    return abs2(sol.At[1, 2, end]) / P0
end

# --- 1. Why poling is needed: unpoled vs QPM-poled at LN's natural mismatch ---
P_demo = 0.3  # W
eta_unpoled = shg_efficiency(P_demo; deltak0=deltak0_natural, poling=0.0)
eta_poled = shg_efficiency(P_demo; deltak0=deltak0_natural, poling=poling_period)
eta_ideal = shg_efficiency(P_demo; deltak0=0.0, poling=0.0)

println("At P=$(P_demo) W: unpoled η=$(round(100eta_unpoled, sigdigits=3))%, ",
        "QPM-poled η=$(round(100eta_poled, sigdigits=2))%, ",
        "ideal phase-matched η=$(round(100eta_ideal, sigdigits=3))%")

# --- 2. Depleted-pump SHG conversion curve (designed PPLN, deltak0=0 by construction) ---
powers = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # W
eta_vs_power = [shg_efficiency(P0) for P0 in powers]
eta_tanh2 = [tanh(kappa * sqrt(P0) * L)^2 for P0 in powers]

# --- 3. QPM acceptance bandwidth (sinc^2), well below depletion ---
P_weak = 1e-3  # W, safely in the undepleted regime
dk_range = range(-3000.0, 3000.0; length=61)
eta_vs_dk = [shg_efficiency(P_weak; deltak0=dk) for dk in dk_range]
nothing # hide
```

```@example ex13
using Plots # hide

p1 = bar(["Unpoled", "QPM-poled", "Ideal\n(Δk=0)"], 100 .* [eta_unpoled, eta_poled, eta_ideal]; # hide
    legend=false, ylabel="SHG Efficiency [%]", # hide
    title="Why PPLN Needs Poling\n(P=$(P_demo) W, L=1 cm)", color=[:gray :steelblue :seagreen]) # hide

p2 = plot(powers, 100 .* eta_vs_power; marker=:circle, lw=2, label="Soliton.jl", # hide
    xlabel="Input Power P₁ [W]", ylabel="SHG Efficiency [%]", # hide
    title="Depleted-Pump SHG Conversion", legend=:bottomright) # hide
plot!(p2, powers, 100 .* eta_tanh2; ls=:dash, color=:black, label="tanh²(κ√P₁L)") # hide

p3 = plot(collect(dk_range), 100 .* eta_vs_dk; lw=2, color=:orangered, label=false, # hide
    xlabel="Residual Phase Mismatch Δk [1/m]", ylabel="SHG Efficiency [%]", # hide
    title="QPM Phase-Matching Acceptance") # hide

plot(p1, p2, p3; layout=(1, 3), size=(1300, 400), dpi=300, bottom_margin=8Plots.mm) # hide
```

---

## 📊 Expected Results

| Quantity | Value |
|:---|:---|
| Unpoled SHG efficiency (natural LN Δk, 1 cm, 0.3 W) | ≈ 0% (coherence length ≈ 9.6 μm ≪ 1 cm) |
| QPM-poled SHG efficiency (same conditions) | ≈ 16% |
| Ideal phase-matched efficiency (Δk=0, same conditions) | ≈ 34% |
| SHG conversion at 1 W, 1 cm (Δk=0) | ≈ 71%, matching `tanh²(κ√P₁L)` exactly |
| First `sinc²` acceptance null | at `Δk = ±2π/L` (≈ ±628 rad/m for L=1 cm) |

The gap between the "unpoled" and "ideal phase-matched" bars is the whole reason PPLN exists; the gap between "QPM-poled" and "ideal" is the well-known **first-order QPM efficiency factor** — an idealized square-wave poling profile only recovers its fundamental Fourier component of the coupling, i.e. an effective coupling $\kappa_{\text{eff}} = (2/\pi)\kappa$, so efficiency scales as $(2/\pi)^2 \approx 0.405$ relative to a hypothetical unmismatched crystal with the same $\kappa$ — consistent with what's observed here (`SecondOrderMedium` intentionally does not pre-apply this factor; see the `guide/second_order.md` "Known Limitations" note).

!!! note "Reading the acceptance bandwidth in real units"
    The `Δk` sweep here stands in for whatever physical knob detunes phase matching in a real device — pump wavelength or crystal temperature — via the material's `dΔk/dλ` or `dΔk/dT`. Real PPLN devices are commonly quoted with a temperature acceptance bandwidth of a few °C·cm and a wavelength acceptance of order 0.1–1 nm·cm; converting this example's `Δk` axis into those units requires the LN Sellmeier/thermo-optic data, which `SecondOrderMedium`'s Phase-1 model doesn't compute internally (`deltak0` is a direct input, not derived from a material model — see `ROADMAP.md`).

## References

> M. M. Fejer, G. A. Magel, D. H. Jundt & R. L. Byer, "Quasi-phase-matched second harmonic generation: tuning and tolerances,"
> *IEEE J. Quantum Electron.* **28**, 2631–2654 (1992). DOI: [10.1109/3.161322](https://doi.org/10.1109/3.161322)

> K. R. Parameswaran, R. K. Route, J. R. Kurz, R. V. Roussev, M. M. Fejer & M. Fujimura, "Highly efficient second-harmonic generation in buried waveguides formed by annealed and reverse proton exchange in periodically poled lithium niobate,"
> *Opt. Lett.* **27**, 43–45 (2002). DOI: [10.1364/OL.27.000043](https://doi.org/10.1364/OL.27.000043)
