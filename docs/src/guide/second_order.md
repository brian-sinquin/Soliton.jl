# Second-Order (χ⁽²⁾) Nonlinearity — SHG / SFG / OPA

Soliton supports second-order (χ⁽²⁾) nonlinear propagation ([`SecondOrderMedium`](@ref)) — second-harmonic generation (SHG), sum-/difference-frequency generation, and degenerate optical parametric amplification — via a coupled fundamental + second-harmonic envelope model. This is **Phase 1** of the second-order-nonlinearity roadmap (see `ROADMAP.md`); the unified broadband χ⁽²⁾+χ⁽³⁾ cascaded-nonlinearity model (T. Voumard et al., *APL Photonics* **8**, 036114 (2023)) is planned as Phase 2.

See [Example 13](../examples/ex13_ppln_shg.md) for a full worked PPLN waveguide simulation (quasi-phase-matching, depleted-pump conversion, and acceptance bandwidth), and [Example 14](../examples/ex14_opcpa.md) for degenerate optical parametric chirped-pulse amplification (the same coupled equations run with the roles of the two branches reversed — strong pump, weak amplified signal).

---

## ⚡ Physics Model

Two envelopes propagate on the same time grid: the fundamental $A_1$ at $\omega_0$ and the second harmonic $A_2$ at $2\omega_0$:

$$\frac{\partial A_1}{\partial z} = i\hat{D}_1 A_1 + i\kappa(z)\, A_2 A_1^{*}\, e^{-i\Delta k_0 z}$$
$$\frac{\partial A_2}{\partial z} = i\hat{D}_2 A_2 + i\kappa(z)\, A_1^{2}\, e^{+i\Delta k_0 z}$$

where $\hat{D}_1, \hat{D}_2$ are each branch's own linear dispersion/loss operator (`dispersion_fund`/`dispersion_sh`, each expanded around its *own* carrier — $\omega_0$ for the fundamental, $2\omega_0$ for the second harmonic), and $\Delta k_0 = \beta_0(2\omega_0) - 2\beta_0(\omega_0)$ is the phase mismatch, supplied directly via `deltak0` (it is intentionally **not** derivable from `dispersion_fund`/`dispersion_sh`, which — like every other dispersion model in Soliton — omit the removable $\beta_0$ term; this mirrors how `BirefringentMedium`'s `deltabeta0` is supplied directly).

Using the identical real $\kappa$ in both equations makes the system exactly power-conserving in the lossless case ($d(P_1+P_2)/dz = 0$): two photons at $\omega_0$ convert to one photon at $2\omega_0$, so energy conservation requires no extra factor of 2 between the equations — this is verified numerically in `test/test_second_order.jl`.

Two closed-form limits are used for validation:
- **Undepleted pump** ($|A_2|\ll|A_1|$): $|A_2(L)|^2 \approx \kappa^2 P_{1,0} L^2 \,\mathrm{sinc}^2(\Delta k_0 L/2)$
- **Depleted pump, perfect phase matching** ($\Delta k_0=0$, exact): $P_2(z)/P_1(0) = \tanh^2\!\big(\kappa\sqrt{P_1(0)}\,z\big)$

Quasi-phase-matching (periodic poling) is modeled as an idealized period-`poling_period` sign-flipping square wave on $\kappa(z)$ (`poling_period=0.0`, the default, disables poling for naturally- or birefringent-phase-matched devices).

---

## 💻 Usage Example

```julia
using Soliton

grid = create_grid(2^12, 20e-12, 1550e-9)

medium = SecondOrderMedium(;
    length      = 0.01,       # 1 cm waveguide
    kappa       = 5.0,        # 1/(√W·m) normalized SHG coupling
    betas_fund  = [-1000e-27],
    betas_sh    = [-500e-27],
    deltak0     = 0.0,        # perfect phase matching
    poling_period = 0.0,      # no QPM needed (already phase matched)
    lambda0     = 1550e-9,
)

pulse = SecondOrderPulse(sech_pulse(grid, 50.0, 200e-15).At, zeros(ComplexF64, grid.N), grid)

# Coupled χ⁽²⁾ propagation currently requires the fixed-step SSFM solver
# (the same limitation BirefringentMedium has — see "Known Limitations" below).
params = SimParams(; medium=medium, z_saves=50, raman_model=nothing, solver=SSFM(1e-6))
sol = solve(pulse, params)

# sol.At[:, 1, :] = fundamental, sol.At[:, 2, :] = second harmonic
```

---

## Known Limitations (Phase 1)

- **No simultaneous χ⁽³⁾**: Kerr/Raman are not modeled on either branch. Combined χ⁽²⁾+χ⁽³⁾ (needed for cascaded-nonlinearity supercontinuum) is Phase 2's whole point — see `ROADMAP.md`.
- **No self-steepening** on the χ⁽²⁾ coupling term.
- **SSFM only**: like `BirefringentMedium`, coupled two-field propagation currently only supports the fixed-step `SSFM` solver, not the adaptive `ERK4IP`.
- **Idealized QPM**: the poling square wave is not first-order-Fourier-derated (no `2/π` efficiency factor) — pass an already-derated `kappa` if that correction matters for your device.

## References

- R. W. Boyd, *Nonlinear Optics*, 4th ed., Ch. 2 (three-wave mixing, SHG, QPM)
- T. Voumard, M. Ludwig, T. Wildi, F. Ayhan, V. Brasch, L. G. Villanueva & T. Herr, "Simulating supercontinua from mixed and cascaded nonlinearities," *APL Photonics* **8**, 036114 (2023). DOI: [10.1063/5.0135252](https://doi.org/10.1063/5.0135252)
