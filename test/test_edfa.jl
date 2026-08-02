using Test
using GNLSE
using Random

@testset "Active Amplifying Fiber Dynamics (EDFA/YDFA/TDFA)" begin
    grid = create_grid(2^10, 10e-12, 1550e-9)
    pulse = gaussian_pulse(grid, 100.0, 100e-15)
    E_in = pulse_energy(pulse)

    @testset "AmplifyingMedium Constructors" begin
        med = AmplifyingMedium(
            length=2.0,
            gamma=0.0012,
            g0_db=10.0, # +10 dB/m
            Esat=1e-6,   # 1 μJ
            noise_figure_db=4.5,
            loss=0.0,
            betas=[-22.0e-27],
            lambda0=1550e-9,
        )

        @test med.length == 2.0
        @test med.Esat == 1e-6
        @test med.noise_figure_db == 4.5
        @test med.g0 ≈ log(10.0) # 10 dB/m = 2.3026 Np/m
    end

    @testset "Propagation & Gain Saturation" begin
        # Low energy pulse -> unsaturated gain boost
        low_pulse = gaussian_pulse(grid, 1.0, 100e-15)
        med = AmplifyingMedium(
            length=1.0,
            gamma=0.0012,
            g0_db=5.0, # +5 dB/m
            Esat=1.0,  # Huge Esat -> linear gain regime
            noise_figure_db=0.0,
            betas=[-22.0e-27],
            lambda0=1550e-9,
        )
        params = SimParams(; medium=med, z_saves=5, raman_model=nothing)

        sol = solve(low_pulse, params; progress=false)
        @test sol isa Solution
        E_out = pulse_energy(Pulse(sol))

        # Expected power gain: +5 dB/m over 1 m = 10^(5/10) = 3.162x energy
        @test E_out / pulse_energy(low_pulse) ≈ 10.0^(5.0/10.0) rtol=0.05

        # Saturated gain regime
        high_pulse = gaussian_pulse(grid, 1e5, 100e-15) # ~10.6 nJ pulse
        med_sat = AmplifyingMedium(
            length=1.0,
            gamma=0.0012,
            g0_db=10.0, # +10 dB/m (linear boost would be 10x)
            Esat=1e-9,  # 1 nJ Esat -> heavily saturated
            noise_figure_db=0.0,
            betas=[-22.0e-27],
            lambda0=1550e-9,
        )
        params_sat = SimParams(; medium=med_sat, z_saves=5, raman_model=nothing)
        sol_sat = solve(high_pulse, params_sat; progress=false)
        E_out_sat = pulse_energy(Pulse(sol_sat))
        # Net gain ratio E_out / E_in should be significantly smaller than linear 10x gain
        @test E_out_sat / pulse_energy(high_pulse) < 5.0
    end

    @testset "ASE noise injection" begin
        # Regression test: noise_figure_db was stored/threaded through
        # aux_data but never actually used by any solver — ASE was
        # documented (physics.md, guide/edfa.md) but pure vaporware.
        med = AmplifyingMedium(
            length=1.0,
            gamma=0.0012,
            g0_db=10.0,
            Esat=1.0,
            noise_figure_db=6.0,
            betas=[-22.0e-27],
            lambda0=1550e-9,
        )
        pulse = gaussian_pulse(grid, 1.0, 100e-15)
        params = SimParams(; medium=med, z_saves=5, raman_model=nothing, solver=SSFM(1e-3))

        sol = solve(pulse, params; rng=Random.MersenneTwister(1), progress=false)
        # Far from the pulse (many FWHM away) the field should now carry a
        # nonzero ASE noise floor instead of exact numerical zero.
        far_idx = findall(t -> abs(t) > 20 * 100e-15, grid.t)
        @test maximum(abs2, sol.At[far_idx, end]) > 0

        # A higher noise figure must inject more ASE power (same seed so the
        # only difference is n_sp).
        med_lowF = AmplifyingMedium(
            length=1.0,
            gamma=0.0012,
            g0_db=10.0,
            Esat=1.0,
            noise_figure_db=0.0,
            betas=[-22.0e-27],
            lambda0=1550e-9,
        )
        params_lowF = SimParams(;
            medium=med_lowF, z_saves=5, raman_model=nothing, solver=SSFM(1e-3)
        )
        sol_lowF = solve(pulse, params_lowF; rng=Random.MersenneTwister(1), progress=false)
        noise_power_highF = sum(abs2, sol.At[far_idx, end])
        noise_power_lowF = sum(abs2, sol_lowF.At[far_idx, end])
        @test noise_power_highF > noise_power_lowF

        # A passive medium (no gain) must never inject ASE noise — any residual
        # is only FFT/numerical round-off, many orders of magnitude below the
        # ASE-seeded noise floor measured above.
        passive = Medium(1.0, 0.0012, 0.0, [-22.0e-27], 1550e-9)
        params_passive = SimParams(;
            medium=passive, z_saves=5, raman_model=nothing, solver=SSFM(1e-3)
        )
        sol_passive = solve(
            pulse, params_passive; rng=Random.MersenneTwister(1), progress=false
        )
        @test maximum(abs2, sol_passive.At[far_idx, end]) < 1e-6 * noise_power_lowF
    end

    @testset "NonlinearityModel gamma types (dispatch-based gamma resolution)" begin
        # Regression test: build_physics_model's gamma resolution for
        # AmplifyingMedium/SemiconductorMedium used to only handle Number/Function
        # inline, silently mishandling ConstantNonlinearity/FrequencyDependentNonlinearity/
        # NonlinearityFromEffectiveArea (a MethodError inside eval_gamma at propagation
        # time). Now unified via the dispatched `_resolve_gamma` helper.
        med = AmplifyingMedium(
            length=0.5,
            gamma=ConstantNonlinearity(0.0012),
            g0_db=3.0,
            Esat=1e-6,
            noise_figure_db=0.0,
            betas=[-20e-27],
            lambda0=1550e-9,
        )
        params = SimParams(; medium=med, z_saves=5, raman_model=nothing, solver=SSFM(1e-3))
        @test solve(pulse, params; progress=false) isa Solution

        med_freq = AmplifyingMedium(
            length=0.5,
            gamma=FrequencyDependentNonlinearity(w -> 0.0012),
            g0_db=3.0,
            Esat=1e-6,
            noise_figure_db=0.0,
            betas=[-20e-27],
            lambda0=1550e-9,
        )
        params_freq = SimParams(;
            medium=med_freq, z_saves=5, raman_model=nothing, solver=SSFM(1e-3)
        )
        @test solve(pulse, params_freq; progress=false) isa Solution
    end
end
