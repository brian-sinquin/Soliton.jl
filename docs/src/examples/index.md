```@meta
CurrentModule = JuGNLSE
```

# Examples

Each example reproduces a key result from the nonlinear fiber optics literature,
providing the exact parameters needed to match published figures.
All quantities are in natural SI units.

| # | Title | Paper | Key Feature |
|:---|:---|:---|:---|
| [1](ex1_supercontinuum.md) | Supercontinuum in PCF | Dudley et al. 2006 | Soliton fission, dispersive waves |
| [2](ex2_ssfs.md) | Soliton Self-Frequency Shift | Mitschke & Gordon 1986 | Raman red-shift ``\propto T_0^{-4}`` |
| [3](ex3_coherence.md) | Supercontinuum Coherence | Dudley & Coen 2002 | MI noise, ensemble coherence |
| [4](ex4_birefringence.md) | Soliton Trapping | Menyuk 1988 | XPM polarization locking |
| [5](ex5_soliton_compression.md) | Higher-Order Solitons | Mollenauer et al. 1980 | Periodic compression |
| [6](ex6_stable_n3_soliton.md) | Stable N=3 Soliton | Zakharov & Shabat 1972 | FPUT recurrence, perturbation stability |

## Common Setup Pattern

All examples follow the same basic workflow:

```julia
using JuGNLSE

# 1. Grid
grid = create_grid(N_points, time_window, lambda0)

# 2. Medium
medium = Medium(; length, gamma, loss, betas, lambda0)

# 3. Pulse
pulse = sech_pulse(grid, P0, FWHM)

# 4. Parameters
params = SimParams(; medium, raman_model=BlowWood(), self_steepening=true)

# 5. Solve
sol = solve(pulse, params)
```

## Key Dimensionless Numbers

Before running a simulation, compute these to understand the physics:

```julia
# Soliton number N (determines regime)
N = soliton_number(beta2, gamma, T0, P0)
# N ≈ 1: fundamental soliton
# N > 1: higher-order soliton / fission

# Lengths
LD  = dispersion_length(beta2, T0)     # dispersion dominates at L >> LD
LNL = nonlinear_length(gamma, P0)      # nonlinearity dominates at L >> LNL

# Fission length (where soliton breaks up)
L_fiss = LD / N   # approx, without SS/Raman
```
