using Test
using JuGNLSE

@testset "Optics Units Conversions & Soliton Tracker" begin
    @testset "Dispersion D <-> beta2 conversion" begin
        lambda0 = 1550e-9
        D = 17.0 # ps/(nm*km) for SMF-28
        b2 = dispersion_D_to_beta2(D, lambda0)

        # SMF-28 beta2 at 1550 nm is approx -21.6 ps^2/km = -21.6e-27 s^2/m
        @test b2 < 0 # anomalous dispersion
        @test isapprox(b2, -21.67e-27; rtol=0.01)

        # Round trip
        D_back = beta2_to_dispersion_D(b2, lambda0)
        @test D_back ≈ D
    end

    @testset "Dispersion slope S -> beta3 conversion" begin
        lambda0 = 1550e-9
        D = 17.0
        S = 0.056 # ps/(nm^2*km) for SMF-28
        b3 = dispersion_S_to_beta3(S, D, lambda0)
        @test b3 > 0 # positive beta3
        @test isapprox(b3, 0.1264e-39; rtol=0.02) # SMF-28 beta3 = 0.1264 ps^3/km = 1.264e-40 s^3/m
    end

    @testset "Wavelength <-> Frequency" begin
        lam = 1550e-9
        f = wavelength_to_frequency(lam)
        @test f ≈ 193.414e12 rtol=1e-3 # 193.4 THz
        @test frequency_to_wavelength(f) ≈ lam
    end

    @testset "Automated Soliton Tracking" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)
        params = SimParams(; medium=medium, z_saves=5, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        z_fiss, peak_power_z, centroid_w_z = track_solitons(sol)

        @test length(peak_power_z) == 5
        @test length(centroid_w_z) == 5
        @test 0.0 <= z_fiss <= 0.02
    end

    @testset "Resonant Cherenkov Dispersive Wave Analyzer" begin
        grid = create_grid(2^12, 12.5e-12, 835e-9)
        betas = [-11.830e-27, 8.1038e-40]
        medium = Medium(0.15, 0.11, 0.0, betas, 835e-9)
        pulse = sech_pulse(grid, 10000.0, 50e-15)

        lambda_dw = dispersive_wave_wavelength(medium, pulse)
        @test 450e-9 <= lambda_dw <= 820e-9
    end
end
