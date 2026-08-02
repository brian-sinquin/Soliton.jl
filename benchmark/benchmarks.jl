using BenchmarkTools
using JuGNLSE

const SUITE = BenchmarkGroup()

# ---------------------------------------------------------------------------
# 1. Solvers Benchmark Group
# ---------------------------------------------------------------------------
SUITE["solvers"] = BenchmarkGroup()

# Scenario: Supercontinuum generation grid setup
grid = create_grid(2^12, 10e-12, 835e-9)
medium = Medium(0.05, 0.11, 0.0, [-11.83e-27, 8.10e-41], 835e-9)
pulse = sech_pulse(grid, 10000.0, 50e-15)

params_erk4ip = SimParams(;
    medium=medium,
    z_saves=20,
    raman_model=BlowWood(),
    self_steepening=true,
    solver=ERK4IP(; rtol=1e-5, atol=1e-7),
)
params_ssfm = SimParams(;
    medium=medium,
    z_saves=20,
    raman_model=BlowWood(),
    self_steepening=true,
    solver=SSFM(5e-5),
)

SUITE["solvers"]["erk4ip_scalar"] = @benchmarkable solve(
    $pulse, $params_erk4ip; progress=false
)
SUITE["solvers"]["ssfm_scalar"] = @benchmarkable solve($pulse, $params_ssfm; progress=false)

# Vectorial SSFM
dis_x = TaylorDispersion([-11.83e-27])
dis_y = TaylorDispersion([-11.83e-27])
b_medium = BirefringentMedium(0.05, 0.11, 0.0, dis_x, dis_y, 1e-3, 835e-9)
vpulse = VectorialPulse(pulse.At, pulse.At .* 0.5, grid)
params_vssfm = SimParams(; medium=b_medium, z_saves=20, solver=SSFM(5e-5))

SUITE["solvers"]["ssfm_vectorial"] = @benchmarkable solve(
    $vpulse, $params_vssfm; progress=false
)

# ---------------------------------------------------------------------------
# 2. Physics & Operators Benchmark Group
# ---------------------------------------------------------------------------
SUITE["operators"] = BenchmarkGroup()

model_scalar = build_physics_model(grid, params_erk4ip, pulse.At)
model_vec = build_physics_model(grid, params_vssfm, vpulse.At)

SUITE["operators"]["build_physics_model"] = @benchmarkable build_physics_model(
    $grid, $params_erk4ip, $(pulse.At)
)
SUITE["operators"]["spm_raman_eval"] = @benchmarkable JuGNLSE._spm_raman(
    $(pulse.AW), $model_scalar, 0.025
)
SUITE["operators"]["vectorial_spm_fwm_eval"] = @benchmarkable JuGNLSE._vectorial_spm_fwm(
    $(vpulse.At), $model_vec, 0.025
)

# ---------------------------------------------------------------------------
# 3. Analysis Metrics Benchmark Group
# ---------------------------------------------------------------------------
SUITE["analysis"] = BenchmarkGroup()

ensemble = [add_noise(pulse; photons_per_mode=1.0) for _ in 1:10]

SUITE["analysis"]["pulse_energy"] = @benchmarkable pulse_energy($pulse)
SUITE["analysis"]["add_noise"] = @benchmarkable add_noise($pulse; photons_per_mode=1.0)
SUITE["analysis"]["spectral_coherence"] = @benchmarkable spectral_coherence($ensemble)
