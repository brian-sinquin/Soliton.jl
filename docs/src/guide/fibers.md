# Commercial Fiber Library & Glass Presets

GNLSE provides a built-in catalog of commercial optical fibers (`FiberLibrary`) and standard optical glass Sellmeier dispersion models (`RefractiveIndex.io` database parameters).

---

## 🏬 Commercial Fiber Catalog (`commercial_fiber`)

Instead of manually constructing Taylor dispersion vectors and nonlinear coefficients, you can instantiate standard commercial fibers with a single call to [`commercial_fiber`](@ref):

```julia
using Soliton

# 1. Corning SMF-28e+ (Standard Telecom Fiber at 1550 nm)
smf28 = commercial_fiber("Corning_SMF28", length=100.0)

# 2. NKT Photonics NL-PM-750 (Supercontinuum Photonic Crystal Fiber)
pcf = commercial_fiber("NKT_NL_PM_750", length=0.15)

# 3. Thorlabs PM780-HP (Polarization Maintaining 780 nm Fiber)
pm780 = commercial_fiber("Thorlabs_PM780", length=2.0)
```

### Available Commercial Fibers

| Fiber Key | Manufacturer | Description | Default $\lambda_0$ | $\gamma$ [1/(W·m)] | Loss [dB/m] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `"Corning_SMF28"` | Corning | Standard single-mode telecom fiber for C-band | 1550 nm | 0.00127 | 0.0002 |
| `"NKT_NL_PM_750"` | NKT Photonics | Highly nonlinear PM photonic crystal fiber | 835 nm | 0.11 | 0.0010 |
| `"Thorlabs_PM780"`| Thorlabs | PM fiber for 780 nm delivery | 780 nm | 0.0055 | 0.0030 |
| `"Thorlabs_PM1550"`| Thorlabs | PM fiber for 1550 nm | 1550 nm | 0.0012 | 0.0002 |
| `"NKT_LMA10"` | NKT Photonics | Large Mode Area photonic crystal fiber | 1064 nm | 0.0015 | 0.0005 |

### Overriding Wavelength or Loss

You can override default parameters when instantiating:

```julia
custom_smf = commercial_fiber("Corning_SMF28", length=50.0, lambda0=1310e-9, loss=0.0004)
```

---

## 🧪 Glass Refractive Index Presets (`RefractiveIndex.io`)

GNLSE provides standard Sellmeier dispersion models derived from the `RefractiveIndex.io` database:

- [`FusedSilica()`](@ref): Pure fused silica glass ($0.21 - 3.71\,\mu\text{m}$, Malitson 1965).
- [`SF6()`](@ref): Schott SF6 heavy flint glass.
- [`SF57()`](@ref): Schott SF57 lead-silicate glass.
- [`GeO2DopedSilica(x)`](@ref): Germania-doped silica core ($x \in [0, 15]\,\%\text{ GeO}_2$, Fleming 1978).

### Example: Custom Fiber with Fused Silica Core

```julia
grid = create_grid(2^13, 10e-12, 1064e-9)
pulse = sech_pulse(grid, 5000.0, 100e-15)

# Fiber with Fused Silica Sellmeier dispersion
medium = Medium(
    length = 0.5,
    gamma = 0.01,
    loss = 0.0,
    dispersion = FusedSilica(),
    lambda0 = 1064e-9
)

params = SimParams(; medium=medium, z_saves=100)
sol = solve(pulse, params)
```
