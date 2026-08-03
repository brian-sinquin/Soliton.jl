using Test
using Soliton

@testset "Wavelength & z-Dependent Loss and Gain Models" begin
    grid = create_grid(2^10, 10e-12, 1550e-9)
    pulse = gaussian_pulse(grid, 10.0, 1e-12) # 10 W pulse

    @testset "Vector Wavelength-Dependent Loss Spectrum α(ω)" begin
        # Define silica water peak absorption at 1383 nm
        loss_spectrum = [
            0.2 + 5.0 * exp(-((w - 2π*299792458/1383e-9) / (2π*5e12))^2) for w in grid.W
        ]
        medium = Medium(
            length=0.5, gamma=0.001, loss=loss_spectrum, betas=[-20e-27], lambda0=1550e-9
        )

        params = SimParams(; medium=medium, z_saves=5, raman_model=nothing)
        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.5
    end

    @testset "Vector EDFA Gain Spectrum g₀(ω)" begin
        # Parabolic EDFA gain spectrum centered at 1550 nm (1520-1570 nm band)
        g0_spectrum = [5.0 * exp(-((w - grid.omega0) / (2π*1.5e12))^2) for w in grid.W]
        edfa = AmplifyingMedium(
            length=2.0,
            gamma=0.001,
            g0=g0_spectrum,
            Esat=1.0e-6,
            noise_figure_db=4.0,
            loss=0.0,
            betas=[-20e-27],
            lambda0=1550e-9,
        )

        params = SimParams(; medium=edfa, z_saves=5, raman_model=nothing)
        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test pulse_energy(Pulse(sol)) > pulse_energy(pulse) # Pulse is amplified
    end

    @testset "Functional z-Dependent Loss α(z) and Gain g(z)" begin
        loss_func = (omega, z) -> 0.1 * (1.0 + z / 0.5) # Increasing loss along fiber
        medium = Medium(
            length=0.5, gamma=0.001, loss=loss_func, betas=[-20e-27], lambda0=1550e-9
        )

        params = SimParams(; medium=medium, z_saves=5, raman_model=nothing)
        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.5
    end

    @testset "Silica Loss Spectrum Preset" begin
        loss_model = SilicaLossSpectrum()

        # Loss at 1550 nm (~0.29 dB/km Rayleigh scattering)
        omega_1550 = 2π * c / 1550e-9
        loss_1550_db_m = loss_model(omega_1550)
        @test isapprox(loss_1550_db_m * 1000, 0.295; rtol=0.1)

        # OH absorption peak at 1383 nm
        omega_oh = 2π * c / 1383e-9
        loss_oh_db_m = loss_model(omega_oh)
        @test loss_oh_db_m * 1000 > 30.0 # > 30 dB/km OH peak

        # Test within Medium
        grid_loc = create_grid(2^10, 10e-12, 1550e-9)
        medium = Medium(1.0, 0.0011, loss_model, [-21.5e-27], 1550e-9)
        D_op = dispersion_operator(grid_loc, medium)
        @test length(D_op) == 1024
        @test all(real.(D_op) .< 0.0) # loss attenuation
    end
end
