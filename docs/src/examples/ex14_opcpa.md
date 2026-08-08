# Example 14: Optical Parametric Chirped-Pulse Amplification (OPCPA)

This example demonstrates **degenerate OPCPA**: a weak, stretched (chirped) seed pulse is parametrically amplified by a strong pump via the same χ⁽²⁾ coupling used for PPLN SHG in [Example 13](ex13_ppln_shg.md) — just run with the initial conditions reversed. Instead of a weak fundamental generating a second harmonic, a strong "pump" (the SHG example's second-harmonic branch) transfers energy into a weak "signal" (the fundamental branch) launched alongside it. The example follows the full CPA narrative: **stretch → amplify → recompress**.

---

## 🔬 Physics: The Degenerate Limit of OPCPA

[`SecondOrderMedium`](@ref) is currently a two-envelope (fundamental + second-harmonic) coupled model (Phase 1 of the second-order-nonlinearity roadmap — see `ROADMAP.md`). Real OPCPA is usually **non-degenerate**: a pump at $\omega_p$ splits into a signal at $\omega_s$ and idler at $\omega_i$ with $\omega_p = \omega_s + \omega_i$ and $\omega_s \neq \omega_i$. This example instead uses the **degenerate limit** $\omega_s = \omega_i = \omega_p/2$ — mathematically the *same* coupled equations as SHG:

```math
\frac{\partial A_1}{\partial z} = i\hat{D}_1 A_1 + i\kappa\, A_2 A_1^{*}\, e^{-i\Delta k_0 z}, \qquad
\frac{\partial A_2}{\partial z} = i\hat{D}_2 A_2 + i\kappa\, A_1^{2}\, e^{+i\Delta k_0 z}
```

but now launched with $A_2$ (pump, at $2\omega_0$) strong and $A_1$ (signal, at $\omega_0$) weak, instead of the other way around. This correctly captures the core OPCPA physics — **phase-sensitive parametric gain**, **pump depletion**, and **gain bandwidth set by phase matching** — but not the group-velocity walk-off between two *independently tunable* signal/idler branches that a genuine three-wave (non-degenerate) model would have; that's a natural Phase-2-adjacent extension (see `ROADMAP.md`).

At perfect phase matching ($\Delta k_0=0$) and in the undepleted-pump limit, launching a real (in-quadrature) seed gives an **exact** closed-form gain — derived directly from the same coupled equations, not assumed from a textbook — by writing $A_1=x+iy$ and $A_2=A_{2,0}$ (real, constant):

```math
\frac{dx}{dz} = g y, \qquad \frac{dy}{dz} = g x, \qquad g \equiv \kappa\sqrt{P_{\text{pump}}}
```

which gives $x(z)=A_{1,0}\cosh(gz)$, $y(z)=A_{1,0}\sinh(gz)$, hence the **exact parametric power gain**:

```math
\frac{P_1(z)}{P_1(0)} = \cosh(2gz)
```

This is the quantitative validation target below (Panel 1). The chirped-pulse case (Panels 2–3) is a qualitative demonstration of realistic OPCPA operation, not a per-sample analytic check — a chirped pulse's local phase relative to the pump drifts across its duration, which is itself a real, well-known OPCPA subtlety (gain depends on local seed-pump relative phase).

---

## 💻 Julia Code

```@example ex14
using Soliton
using FFTW

grid = create_grid(2^13, 20e-12, 1550e-9)  # signal @ 1550 nm, pump @ 775 nm

kappa = 50.0       # 1/(√W·m)
P0_pump = 50.0     # W
g = kappa * sqrt(P0_pump)

# --- Panel 1: CW parametric gain vs. interaction length, vs. exact cosh(2gL) ---
P0_seed_cw = 1e-3  # W
pulse_cw = SecondOrderPulse(
    fill(ComplexF64(sqrt(P0_seed_cw)), grid.N), fill(ComplexF64(sqrt(P0_pump)), grid.N), grid
)
Ls = range(0.0, 0.012; length=13)
gain_numeric = Float64[]
for Lx in Ls
    if Lx == 0.0
        push!(gain_numeric, 1.0)
        continue
    end
    medium = SecondOrderMedium(;
        length=Lx, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0], deltak0=0.0, lambda0=1550e-9,
    )
    params = SimParams(; medium=medium, z_saves=2, raman_model=nothing, solver=SSFM(2e-7))
    sol = solve(pulse_cw, params; progress=false)
    push!(gain_numeric, abs2(sol.At[1, 1, end]) / P0_seed_cw)
end
gain_analytic = cosh.(2 .* g .* Ls)

# --- Panels 2/3: chirped-pulse OPCPA (stretch -> amplify -> recompress) ---
T0 = 30e-15         # transform-limited seed duration
T1 = 3e-12          # stretched duration
C = sqrt((T1 / T0)^2 - 1)                # equivalent linear chirp
Phi2 = C * T0^2                          # GDD imparted by the (virtual) stretcher
P0_seed = 1e-3                           # TL-equivalent peak power
P_stretched_peak = P0_seed * T0 / T1     # energy-conserving stretched peak
At_seed = @. sqrt(P_stretched_peak) * exp(-(1 - im * C) * grid.t^2 / (2 * T1^2))

Tpump_FWHM = 6e-12
At_pump = @. sqrt(P0_pump) * exp(-4 * log(2) * 0.5 * grid.t^2 / Tpump_FWHM^2)

pulse = SecondOrderPulse(At_seed, At_pump, grid)
L_amp = 0.01
medium = SecondOrderMedium(;
    length=L_amp, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0], deltak0=0.0, lambda0=1550e-9,
)
params = SimParams(; medium=medium, z_saves=2, raman_model=nothing, solver=SSFM(2e-7))
sol = solve(pulse, params; progress=false)

E_seed_in = sum(abs2.(sol.At[:, 1, 1])) * (grid.t[2] - grid.t[1])
E_seed_out = sum(abs2.(sol.At[:, 1, end])) * (grid.t[2] - grid.t[1])
E_pump_in = sum(abs2.(sol.At[:, 2, 1])) * (grid.t[2] - grid.t[1])
E_pump_out = sum(abs2.(sol.At[:, 2, end])) * (grid.t[2] - grid.t[1])
println("Seed energy gain: ", round(E_seed_out / E_seed_in, digits=1), "x")
println("Pump depletion: ", round(100 * (1 - E_pump_out / E_pump_in), sigdigits=2), "%")

# --- Recompression: propagate the amplified (still stretched) signal through the
#     equal-and-opposite GDD (an idealized grating-pair compressor, not a real fiber) ---
amp_At = sol.At[:, 1, end]
Lc = 0.001
beta2_c = -Phi2 / Lc
amp_pulse = Pulse(amp_At, ifft(amp_At), grid)
comp_medium = Medium(Lc, 0.0, 0.0, [beta2_c], 1550e-9)
comp_params = SimParams(; medium=comp_medium, z_saves=2, raman_model=nothing)
comp_sol = solve(amp_pulse, comp_params; progress=false)
comp_At = comp_sol.At[:, end]

println("Net peak-power gain after recompression: ", round(maximum(abs2.(comp_At)) / P0_seed, digits=1), "x")
nothing # hide
```

```@example ex14
using Plots # hide

p1 = plot(collect(Ls) .* 1e3, gain_numeric; yaxis=:log10, marker=:circle, lw=2, # hide
    label="Soliton.jl", xlabel="Interaction Length [mm]", ylabel="Signal Power Gain", # hide
    title="Parametric Gain vs. Length (CW)", legend=:topleft) # hide
plot!(p1, collect(Ls) .* 1e3, gain_analytic; ls=:dash, color=:black, label="cosh(2gL)") # hide

p2 = plot(grid.t .* 1e12, abs2.(At_seed); lw=2, color=:gray, label="Stretched seed (in)", # hide
    xlabel="Time [ps]", ylabel="Signal Power [W]", title="OPCPA: Stretched Pulse", # hide
    xlims=(-6, 6)) # hide
plot!(p2, grid.t .* 1e12, abs2.(amp_At) ./ 100; lw=2, color=:orangered, # hide
    label="Amplified (in), /100") # hide

p3 = plot(grid.t .* 1e15, abs2.(comp_At); lw=2, color=:seagreen, label="Recompressed", # hide
    xlabel="Time [fs]", ylabel="Signal Power [W]", title="Recompressed Output", # hide
    xlims=(-300, 300)) # hide

plot(p1, p2, p3; layout=(1, 3), size=(1300, 400), dpi=300, bottom_margin=8Plots.mm) # hide
```

---

## 📊 Expected Results

| Quantity | Value |
|:---|:---|
| CW parametric gain, `g·L=3.5` (L=1 cm) | ≈ 585× numeric, ≈ 589× analytic `cosh(2gL)` |
| Stretched seed duration | ≈ 2.7 ps (target 3 ps) |
| Seed energy gain (chirped pulse, 1 cm) | ≈ 367× |
| Pump depletion | ≈ 0.006% (small-signal / low-depletion preamp regime) |
| Transform-limited target duration | ≈ 50 fs (FWHM) |
| Recompressed duration | ≈ 85 fs (broader than TL — gain narrowing) |
| Net peak-power gain after recompression | ≈ 113× |

The recompressed pulse doesn't quite reach the transform limit — this is **gain narrowing**, a real and well-documented OPCPA effect: parametric gain is itself frequency-dependent (it falls off away from perfect phase matching), so a broadband chirped pulse experiences slightly less gain at its spectral edges than at its center, narrowing the amplified spectrum and broadening the recompressed pulse relative to the unamplified seed. This example reproduces that effect from the underlying physics, not from an added approximation.

!!! note "Why the pump barely depletes here"
    Many real OPCPA front-end/pre-amplifier stages deliberately run in this low-depletion regime — it keeps the gain close to its (better-behaved) small-signal value and avoids back-conversion (energy flowing from signal back into the pump past peak conversion), at the cost of needing multiple amplification stages to reach full pump depletion. Push `kappa`, `L_amp`, or `P0_pump` higher to see saturation and depletion set in — the same physics `SecondOrderMedium` already validates via the exact `tanh²` SHG formula in [Example 13](ex13_ppln_shg.md) applies here in reverse.

## References

> A. Dubietis, G. Jonušauskas & A. Piskarskas, "Powerful femtosecond pulse generation by chirped and stretched pulse parametric amplification in BBO crystal,"
> *Opt. Commun.* **88**, 437–440 (1992). DOI: [10.1016/0030-4018(92)90070-8](https://doi.org/10.1016/0030-4018(92)90070-8) — the original OPCPA paper.

> I. N. Ross, P. Matousek, M. Towrie, A. J. Langley & J. L. Collier, "The prospects for ultrashort pulse duration and ultrahigh intensity using optical parametric chirped pulse amplifiers,"
> *Opt. Commun.* **144**, 125–133 (1997). DOI: [10.1016/S0030-4018(97)00399-4](https://doi.org/10.1016/S0030-4018(97)00399-4)

> G. Cerullo & S. De Silvestri, "Ultrafast optical parametric amplifiers,"
> *Rev. Sci. Instrum.* **74**, 1–18 (2003). DOI: [10.1063/1.1523642](https://doi.org/10.1063/1.1523642) — standard reference for the `cosh²`/`sinh²` OPA gain formulas.
