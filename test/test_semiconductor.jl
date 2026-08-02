using Test
using JuGNLSE

@testset "Semiconductor Waveguides (SOI, TPA, Free-Carrier Dynamics)" begin
    grid = create_grid(2^10, 10e-12, 1550e-9)
    pulse = gaussian_pulse(grid, 50.0, 1e-12) # 50 W peak power pulse

    @testset "SemiconductorMedium Constructor" begin
        soi = SemiconductorMedium(
            length=0.01, # 1 cm silicon nanowire
            gamma=300.0, # 300 /W/m Kerr nonlinearity
            alpha2=5e-12, # 5 cm/GW TPA coefficient
            Aeff=0.1e-12, # 0.1 μm² effective area
            tau_c=1e-9,  # 1 ns carrier lifetime
            betas=[-1000e-27], # anomalous dispersion
            lambda0=1550e-9,
        )

        @test soi.length == 0.01
        @test soi.alpha2 == 5e-12
        @test soi.Aeff == 0.1e-12
        @test soi.sigma_fca == 1.45e-21
        @test soi.k_fcr == 5.3e-27
    end

    @testset "TPA Loss & Free Carrier Blue Shift" begin
        soi = SemiconductorMedium(
            length=0.005,
            gamma=100.0,
            alpha2=5e-12,
            Aeff=0.1e-12,
            tau_c=1e-9,
            betas=[-1000e-27],
            lambda0=1550e-9,
        )
        params = SimParams(; medium=soi, z_saves=5, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.005

        # TPA must reduce pulse energy below lossless Kerr propagation
        E_in = pulse_energy(pulse)
        E_out = pulse_energy(Pulse(sol))
        @test E_out < E_in
    end

    @testset "NonlinearityModel gamma types" begin
        # Regression test: SemiconductorMedium's build_physics_model/_semiconductor_spm
        # used to only handle Number/Function gamma inline, hitting a MethodError for
        # ConstantNonlinearity etc. Now dispatched via `_semiconductor_gamma_at_z`.
        common = (alpha2=5e-12, Aeff=0.1e-12, tau_c=1e-9, betas=[-1000e-27], lambda0=1550e-9)

        soi_num = SemiconductorMedium(; length=0.005, gamma=100.0, common...)
        soi_const = SemiconductorMedium(; length=0.005, gamma=ConstantNonlinearity(100.0), common...)
        params_num = SimParams(; medium=soi_num, z_saves=5, raman_model=nothing)
        params_const = SimParams(; medium=soi_const, z_saves=5, raman_model=nothing)

        sol_num = solve(pulse, params_num; progress=false)
        sol_const = solve(pulse, params_const; progress=false)
        @test sol_const isa Solution
        # ConstantNonlinearity(100.0) must behave identically to plain gamma=100.0
        @test sol_const.At ≈ sol_num.At rtol=1e-8

        soi_freq = SemiconductorMedium(; length=0.005, gamma=FrequencyDependentNonlinearity(w -> 100.0), common...)
        params_freq = SimParams(; medium=soi_freq, z_saves=5, raman_model=nothing)
        @test_throws ArgumentError solve(pulse, params_freq; progress=false)
    end
end
