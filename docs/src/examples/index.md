# Examples Overview & Literature Benchmarks

Each example in **JuGNLSE.jl** reproduces a key result from the nonlinear optics literature, providing exact parameters to match published figures and experimental data. All quantities are specified in natural SI units.

---

## 📚 Benchmark Summary Table

| # | Title | Reference Paper | Key Physical Phenomena | Primary Modules |
|:---|:---|:---|:---|:---|
| **1** | [Supercontinuum in PCF](ex1_supercontinuum.md) | Dudley et al., *Rev. Mod. Phys.* **78**, 1135 (2006) | Soliton fission, Cherenkov dispersive waves | `Medium`, `Hollenbeck`, `sech_pulse` |
| **2** | [Soliton Self-Frequency Shift](ex2_ssfs.md) | Mitschke & Mollenauer (1986); Gordon (1986) | Raman red-shift ($\propto T_0^{-4}$) | `BlowWood`, `centroid_wavelength` |
| **3** | [Supercontinuum Coherence](ex3_coherence.md) | Dudley & Coen, *Opt. Lett.* **27**, 1180 (2002) | MI noise seeding, ensemble coherence | `add_noise`, `spectral_coherence` |
| **4** | [Soliton Trapping in Birefringent Fiber](ex4_birefringence.md) | Menyuk, *J. Opt. Soc. Am. B* **5**, 392 (1988) | XPM polarization locking, vector GNLSE | `BirefringentMedium`, `VectorialPulse` |
| **5** | [Higher-Order Soliton Compression](ex5_soliton_compression.md) | Mollenauer et al., *Phys. Rev. Lett.* **45**, 1095 (1980) | Periodic temporal compression ($N=3$) | `soliton_number`, `Medium` |
| **6** | [Stable N=3 Soliton Recurrence](ex6_stable_n3_soliton.md) | Zakharov & Shabat (1972); Akhmediev (1987) | FPUT recurrence & perturbation stability | `solve`, `ERK4IP` |
| **7** | [Gas-Filled Hollow-Core PCF](ex7_hollowcore_gas.md) | Russell et al., *Nat. Photonics* **8**, 278 (2014) | Pressure-tuned dispersion ($\beta_n(P)$) & gas Raman | `HollowCoreFiber`, `MolecularRamanGas` |
| **8** | [Silicon Photonics (TPA)](ex8_silicon_tpa.md) | Yin et al., *Opt. Express* **15**, 13833 (2007) | Two-photon absorption & free-carrier blue-shift | `SemiconductorMedium` |
| **9** | [Femtosecond EDFA Amplifier](ex9_edfa_amplifier.md) | Agrawal, *Nonlinear Fiber Optics*, Ch. 11 | Gain saturation & quantum ASE noise | `AmplifyingMedium` |
| **10** | [Multithreaded Parameter Sweep](ex10_parallel_sweep.md) | Parallel Soliton Fission & SSFS vs Peak Power | Multi-core scaling & parameter sweeps | `solve_sweep`, `Threads` |

---

## 🛠️ Common Workflow Patterns

### 1. Standard Scalar GNLSE (Photonic Crystal Fiber)
```julia
using JuGNLSE, Plots

medium = commercial_fiber("NKT_NL_PM_750", length=0.15) # 15 cm fiber
grid   = create_grid(2^13, 12.5e-12, medium.lambda0)
pulse  = sech_pulse(grid, 10_000.0, 50e-15)
sol    = solve(pulse, SimParams(; medium=medium, raman_model=Hollenbeck(), self_steepening=true))
plot(sol) # Dashboard visualization
```

### 2. Birefringent Coupled Vectorial GNLSE
```julia
using JuGNLSE

grid   = create_grid(2^12, 50e-12, 1550e-9)
disp_x = TaylorDispersion([-21.5e-27], 0.0)
disp_y = TaylorDispersion([-21.5e-27], 1e-12) # group-velocity mismatch
medium = BirefringentMedium(5.0, 0.0011, 0.0, disp_x, disp_y, 0.0, 1550e-9)

Ax     = sech_pulse(grid, 100.0, 1e-12).At
vpulse = VectorialPulse(Ax, Ax, grid) # 45° launch
vsol   = solve(vpulse, SimParams(; medium=medium, solver=SSFM(1e-3), raman_model=nothing))
```

### 3. Active EDFA Fiber Amplifier
```julia
using JuGNLSE

grid   = create_grid(2^13, 10e-12, 1550e-9)
pulse  = gaussian_pulse(grid, 50.0, 100e-15)
edfa   = AmplifyingMedium(; length=2.0, gamma=0.0012, g0_db=12.0, Esat=2.0e-6, noise_figure_db=4.5, betas=[-22.0e-27], lambda0=1550e-9)
sol    = solve(pulse, SimParams(; medium=edfa, raman_model=nothing))
```

### 4. Gas-Filled Hollow-Core PCF
```julia
using JuGNLSE

grid   = create_grid(2^13, 15e-12, 800e-9)
pulse  = sech_pulse(grid, 50e3, 30e-15)
hcf    = HollowCoreFiber(; radius=15e-6, gas=:Ar, pressure=3.0, length=0.5, lambda0=800e-9)
sol    = solve(pulse, SimParams(; medium=hcf, raman_model=nothing))
```

### 5. Silicon Nanowire (TPA & Free Carriers)
```julia
using JuGNLSE

grid   = create_grid(2^12, 40e-12, 1550e-9)
pulse  = gaussian_pulse(grid, 30.0, 2.0e-12)
soi    = SemiconductorMedium(; length=0.01, gamma=300.0, alpha2=5.0e-12, Aeff=0.1e-12, tau_c=1.0e-9, betas=[-1000e-27], lambda0=1550e-9)
sol    = solve(pulse, SimParams(; medium=soi, raman_model=nothing))
```

---

## 📏 Key Characteristic Quantities

Before running simulations, compute these parameters to understand the dominant regime:

```julia
# 1. Soliton Order N
N = soliton_number(beta2, gamma, T0, P0)

# 2. Dispersion & Nonlinear Lengths
LD  = dispersion_length(beta2, T0)   # L >> LD -> dispersion dominant
LNL = nonlinear_length(gamma, P0)    # L >> LNL -> nonlinearity dominant

# 3. Soliton Fission Length
L_fiss = LD / N

# 4. Soliton Self-Frequency Shift Rate (Gordon 1986)
dΩdz = -8 * 3e-15 * abs(beta2) / (15 * T0^4)
```
