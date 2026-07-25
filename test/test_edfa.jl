using Test
using JuGNLSE

@testset "Active Amplifying Fiber Dynamics (EDFA/YDFA/TDFA)" begin
    grid = create_grid(2^10, 10e-12, 1550e-9)
    pulse = gaussian_pulse(grid, 100.0, 100e-15)
    E_in = pulse_energy(pulse)

    @testset "AmplifyingMedium Constructors" begin
        med = AmplifyingMedium(
            length = 2.0,
            gamma = 0.0012,
            g0_db = 10.0, # +10 dB/m
            Esat = 1e-6,   # 1 μJ
            noise_figure_db = 4.5,
            loss = 0.0,
            betas = [-22.0e-27],
            lambda0 = 1550e-9
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
            length = 1.0,
            gamma = 0.0012,
            g0_db = 5.0, # +5 dB/m
            Esat = 1.0,  # Huge Esat -> linear gain regime
            noise_figure_db = 0.0,
            betas = [-22.0e-27],
            lambda0 = 1550e-9
        )
        params = SimParams(; medium=med, z_saves=5, raman_model=nothing)

        sol = solve(low_pulse, params; progress=false)
        @test sol isa Solution
        E_out = pulse_energy(Pulse(sol))

        # Expected power gain: +5 dB/m over 1 m = 10^(5/10) = 3.162x energy
        @test E_out / pulse_energy(low_pulse) ≈ 10.0^(5.0/10.0) rtol=0.05
    end
end
