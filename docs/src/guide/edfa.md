# Active Amplifying Fibers (EDFA / YDFA / TDFA)

GNLSE supports rare-earth active amplifying fibers ([`AmplifyingMedium`](@ref)) with dynamic gain saturation and Amplified Spontaneous Emission (ASE) quantum noise.

---

## ⚡ Physics Model

Active fibers (such as Erbium-doped EDFA at 1550 nm, Ytterbium-doped YDFA at 1064 nm, or Thulium TDFA at 2000 nm) provide localized optical power gain $g(z, \omega)$ that saturates as the pulse accumulates energy along distance $z$:

$$g(z, \omega) = \frac{g_0(\omega)}{1 + \frac{E_{\text{pulse}}(z)}{E_{\text{sat}}}}$$

where:
- $g_0(\omega)$ is the small-signal gain coefficient [1/m] (or $g_{\text{dB}}$ in dB/m).
- $E_{\text{pulse}}(z) = \int |A(t, z)|^2 dt$ is the local pulse energy [J].
- $E_{\text{sat}}$ is the saturation energy of the active medium [J] (typically $0.1 - 10\,\mu\text{J}$).

### Amplified Spontaneous Emission (ASE) Noise

The spontaneous emission factor $n_{\text{sp}}$ is determined by the amplifier Noise Figure $F_{\text{dB}}$ (typically $3 - 6\text{ dB}$):

$$n_{\text{sp}} = \frac{10^{F_{\text{dB}}/10}}{2}$$

---

## 💻 Usage Example

```julia
using Soliton

grid = create_grid(2^13, 10e-12, 1550e-9)
pulse = gaussian_pulse(grid, 10.0, 100e-15) # 10 W peak power input

# Define an Erbium-doped active fiber amplifier (EDFA)
edfa = AmplifyingMedium(
    length = 1.5,           # 1.5 m active fiber
    gamma = 0.0012,         # 1.2 /W/km
    g0_db = 3.0,            # 3 dB/m small-signal gain (typ. 1–5 dB/m for EDF)
                            # Note: 15 dB/m would imply 22.5 dB total gain — unrealistic
    Esat = 1.0e-6,          # 1 μJ saturation energy
    noise_figure_db = 4.5,  # 4.5 dB noise figure
    betas = [-22.0e-27],    # anomalous dispersion
    lambda0 = 1550e-9
)

params = SimParams(; medium=edfa, z_saves=100)
sol = solve(pulse, params)
```
