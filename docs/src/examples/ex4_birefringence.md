```@meta
CurrentModule = JuGNLSE
```

# Example 4: Soliton Trapping in a Birefringent Fiber

**Reproducing the polarization-locking phenomenon of Menyuk (1988)**

DOI: [10.1364/JOSAB.5.000392](https://doi.org/10.1364/JOSAB.5.000392)

---

## Physical Background

In a linearly birefringent fiber, the two polarization axes (x and y) have different group
velocities. A pulse launched at 45° will normally split into two temporally separated
sub-pulses after a "walk-off length" ``L_W = T_0 / |\delta|``, where ``\delta = \beta_{1x} - \beta_{1y}``
is the group-velocity mismatch (walk-off).

Menyuk (1988) showed that above a critical power, **cross-phase modulation (XPM)** can
overcome the walk-off: the faster and slower solitons trap each other through the XPM
potential and travel together as a **bound polarization pair**, with the same group velocity.

The trapping condition is approximately:
```math
\frac{L_D}{L_W} < \frac{\sqrt{3}}{2}
\quad \Longleftrightarrow \quad
|\delta| < \frac{\sqrt{3}}{2} \frac{T_0}{L_D}
```

## Simulation Setup

We reproduce the trapping effect using `BirefringentMedium` with equal energy on both axes
and a group-velocity mismatch that is just inside/outside the trapping threshold.

```@example ex4
using JuGNLSE

lambda0 = 1550e-9
beta2   = -21.5e-27
gamma   = 0.0011

T0   = 1e-12
P0   = abs(beta2) / (gamma * T0^2)
LD   = T0^2 / abs(beta2)
FWHM = 2 * log(1 + sqrt(2)) * T0

grid = create_grid(2^12, 80e-12, lambda0)
delta_beta1 = 1e-12

disp_x = TaylorDispersion([beta2],  0.0)
disp_y = TaylorDispersion([beta2], +delta_beta1)

med_A = BirefringentMedium(5.0, gamma, 0.0, disp_x, disp_y, 0.0, lambda0)

Ax = sech_pulse(grid, P0, FWHM).At
Ay = copy(Ax)
vpulse = VectorialPulse(Ax, Ay, grid)

params = SimParams(; medium=med_A, z_saves=200, solver=SSFM(1e-3), raman_model=nothing)
vsol_trap = solve(vpulse, params; progress=false)
```

```@example ex4; hide = true
using Plots
gr()

t_ps = vsol_trap.t .* 1e12
px = abs2.(vsol_trap.At[:, 1, end])
py = abs2.(vsol_trap.At[:, 2, end])

plot(t_ps, px, label="x-polarization (trapped)", xlabel="Time (ps)", ylabel="Power (W)", color=:blue, lw=1.5)
plot!(t_ps, py, label="y-polarization (trapped)", color=:red, ls=:dash, lw=1.5, plot_title="Soliton Trapping in Birefringent Fiber")
```

## Expected Results

| Case | Walk-off ``\delta`` | Temporal separation at ``z = 5`` m |
|:---|:---|:---|
| Trapped | 1 ps/m (< threshold) | < 1 ps — solitons co-propagate |
| Separated | 5 ps/m (> threshold) | ≈ 25 ps — linear walk-off |

## Effect of Δβ₀ (Phase Mismatch)

A non-zero birefringence phase mismatch `deltabeta0` suppresses the coherent FWM coupling term.
For ``\Delta\beta_0 \gg 1/L`` the FWM term averages out and only SPM/XPM survive:

```julia
# High phase mismatch: FWM averages out
med_highbire = BirefringentMedium(
    5.0, gamma, 0.0, disp_x, disp_y,
    1e6,       # Δβ₀ = 10⁶ 1/m  (beat length ~ 6 µm — very high birefringence)
    lambda0,
)
vsol_hb = solve(vpulse, SimParams(; medium=med_highbire, z_saves=200,
                                    solver=SSFM(1e-3), raman_model=nothing))
```

## References

> C. R. Menyuk, "Stability of solitons in birefringent optical fibers. II. Arbitrary
> amplitudes," *J. Opt. Soc. Am. B* **5**, 392–402 (1988).
> DOI: [10.1364/JOSAB.5.000392](https://doi.org/10.1364/JOSAB.5.000392)

> C. R. Menyuk, "Nonlinear pulse propagation in birefringent optical fibers,"
> *IEEE J. Quantum Electron.* **23**, 174–176 (1987).
> DOI: [10.1109/JQE.1987.1073308](https://doi.org/10.1109/JQE.1987.1073308)
