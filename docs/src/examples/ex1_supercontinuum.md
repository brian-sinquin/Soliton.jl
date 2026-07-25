```@meta
CurrentModule = JuGNLSE
```

# Example 1: Supercontinuum Generation in a PCF

**Reproducing Fig. 3 of Dudley, Genty & Coen, Rev. Mod. Phys. 78, 1135 (2006)**

DOI: [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)

---

## Physical Setup

A 50 fs, 10 kW sech² pulse is launched into a 15 cm photonic crystal fiber (PCF) near the zero-dispersion wavelength (ZDW) at 835 nm. The high nonlinearity (γ = 0.11 /W/m) and anomalous dispersion combine with Raman scattering and self-steepening to drive **soliton fission** followed by **dispersive wave (Cherenkov radiation) emission**, producing a supercontinuum spanning more than one octave.

The soliton order is:
```math
N = \sqrt{\frac{\gamma P_0 T_0^2}{|\beta_2|}} \approx 6.7
```
placing the pulse firmly in the soliton fission regime.

## Parameters

All dispersion coefficients from Dudley et al. (2006) Table 1:

| Parameter | Value |
|:---|:---|
| λ₀ | 835 nm |
| L | 15 cm |
| γ | 0.11 1/(W·m) |
| β₂ | −11.83 × 10⁻²⁷ s²/m |
| β₃ | +8.1076 × 10⁻⁴¹ s³/m |
| ... | (up to β₁₀) |
| T_FWHM | 50 fs |
| P₀ | 10,000 W |

## Julia Code

```julia
using JuGNLSE

# ─── Grid ──────────────────────────────────────────────────────────────────
# 8192-point grid, 12.5 ps window at 835 nm — matches gnlse-python defaults
grid = create_grid(2^13, 12.5e-12, 835e-9)

# ─── PCF fiber (Dudley et al. 2006, Table 1) ────────────────────────────────
# Higher-order dispersion coefficients [s^n/m] up to β₁₀
betas = [
    -11.83e-27,   #  β₂ [s²/m]
     8.1076e-41,  #  β₃ [s³/m]
    -9.5229e-56,  #  β₄ [s⁴/m]
     2.0737e-70,  #  β₅
    -5.3943e-85,  #  β₆
     1.3486e-99,  #  β₇
    -2.5495e-114, #  β₈
     3.0524e-129, #  β₉
    -1.7140e-144, # β₁₀
]

medium = Medium(;
    length  = 0.15,       # 15 cm
    gamma   = 0.11,       # [1/(W·m)]
    loss    = 0.0,
    betas   = betas,
    lambda0 = 835e-9,
)

# ─── Input pulse: 50 fs FWHM sech², 10 kW peak power ───────────────────────
# sech_pulse expects FWHM; T₀ = FWHM / (2 ln(1+√2)) ≈ FWHM / 1.763
pulse = sech_pulse(grid, 10_000.0, 50e-15)

# ─── Simulation parameters ──────────────────────────────────────────────────
params = SimParams(;
    medium          = medium,
    z_saves         = 200,
    raman_model     = Hollenbeck(),   # most accurate silica Raman model
    self_steepening = true,
)

# ─── Solve ──────────────────────────────────────────────────────────────────
sol = solve(pulse, params)

# ─── Diagnostics ────────────────────────────────────────────────────────────
using FFTW

# Wavelength grid from the frequency solution (for plotting)
lambda = 2π * c ./ sol.W   # [m]

# Output spectrum at z = 15 cm (last save)
spectrum_out = abs2.(sol.AW[:, end])

# Soliton number
N = soliton_number(betas[1], 0.11, 50e-15 / 1.763, 10_000.0)
println("Soliton number N = ", round(N; digits=2))   # expect ≈ 6.7
```

## Expected Results

- **Soliton fission** occurring near z ≈ 5–7 cm (first fission length L_fiss ≈ L_D/N)
- **Dispersive wave** emitted to the blue side (< 600 nm)
- **Raman-shifted solitons** separating to the red (> 1000 nm)
- Output spectrum spanning roughly 400–1600 nm

## Reference

> J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in photonic crystal fiber,"
> *Rev. Mod. Phys.* **78**, 1135–1184 (2006).
> DOI: [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)
