using JuGNLSE

# Warm-up precompile workload for C/C++/JLL sysimage compilation
grid = create_grid(2^10, 10e-12, 1550e-9)
pulse = sech_pulse(grid, 100.0, 100e-15)

# 1. Standard medium
med = Medium(0.1, 0.001, 0.0, [-22.0e-27], 1550e-9)
sol = solve(pulse, SimParams(; medium=med, raman_model=BlowWood(), z_saves=2); progress=false)

# 2. Vectorial medium
med_vec = BirefringentMedium(0.1, 0.001, 0.0, TaylorDispersion([-22.0e-27]), TaylorDispersion([-22.0e-27], 1e-12), 0.0, 1550e-9)
vpulse = VectorialPulse(pulse.At, pulse.At, grid)
vsol = solve(vpulse, SimParams(; medium=med_vec, raman_model=nothing, z_saves=2); progress=false)

# 3. Active EDFA amplifier
edfa = AmplifyingMedium(length=0.1, gamma=0.001, g0_db=5.0, Esat=1e-6, betas=[-22.0e-27], lambda0=1550e-9)
sol_edfa = solve(pulse, SimParams(; medium=edfa, raman_model=nothing, z_saves=2); progress=false)

# 4. Hollow-Core PCF
hcf = HollowCoreFiber(radius=15e-6, gas=:Ar, pressure=3.0, length=0.1, lambda0=1550e-9)
sol_hcf = solve(pulse, SimParams(; medium=hcf, raman_model=nothing, z_saves=2); progress=false)

# 5. Semiconductor Waveguide
soi = SemiconductorMedium(length=0.01, gamma=100.0, alpha2=5e-12, Aeff=0.1e-12, betas=[-1000e-27], lambda0=1550e-9)
sol_soi = solve(pulse, SimParams(; medium=soi, raman_model=nothing, z_saves=2); progress=false)
