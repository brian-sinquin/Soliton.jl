using Test
using Soliton

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

    @testset "Wavelength <-> Angular Frequency" begin
        lam = 1550e-9
        omega = wavelength_to_omega(lam)
        @test omega ≈ 2π * wavelength_to_frequency(lam)
        @test omega_to_wavelength(omega) ≈ lam
        # Consistent with the internal Grid convention omega0 = 2*pi*c/lambda0
        grid = create_grid(2^8, 1e-12, lam)
        @test wavelength_to_omega(lam) ≈ grid.omega0
        @test omega_to_wavelength(grid.omega0) ≈ lam
    end

    @testset "Decibel / power-ratio conversions" begin
        # Power-ratio dB <-> linear
        @test db_to_linear_power(0.0) ≈ 1.0
        @test db_to_linear_power(10.0) ≈ 10.0
        @test db_to_linear_power(20.0) ≈ 100.0
        @test linear_power_to_db(db_to_linear_power(13.7)) ≈ 13.7
        @test_throws ArgumentError linear_power_to_db(0.0)
        @test_throws ArgumentError linear_power_to_db(-1.0)

        # Amplitude-ratio dB <-> linear (factor of 2 vs power dB)
        @test db_to_linear_amplitude(0.0) ≈ 1.0
        @test db_to_linear_amplitude(20.0) ≈ 10.0
        @test linear_amplitude_to_db(db_to_linear_amplitude(-4.2)) ≈ -4.2
        # A doubling of amplitude is a quadrupling of power: consistent cross-check
        @test db_to_linear_power(2 * linear_amplitude_to_db(2.0)) ≈ 4.0
        @test_throws ArgumentError linear_amplitude_to_db(0.0)

        # dB <-> Np (matches the loss/gain convention used internally for Medium.loss)
        @test db_to_np(10.0) ≈ log(10.0)
        @test np_to_db(db_to_np(0.2)) ≈ 0.2
        # A 10 dB/m loss over 1 m attenuates power by exactly a factor of 10
        @test exp(-db_to_np(10.0)) ≈ 0.1

        # dBm <-> W
        @test dbm_to_watt(0.0) ≈ 1e-3      # 0 dBm = 1 mW
        @test dbm_to_watt(30.0) ≈ 1.0      # 30 dBm = 1 W
        @test watt_to_dbm(1.0) ≈ 30.0
        @test watt_to_dbm(1e-3) ≈ 0.0
        @test watt_to_dbm(dbm_to_watt(17.3)) ≈ 17.3
        @test_throws ArgumentError watt_to_dbm(0.0)
        @test_throws ArgumentError watt_to_dbm(-1.0)
    end

    @testset "Loss/gain refactor: dB helpers match Medium's internal dB/m -> Np/m operator" begin
        # Regression guard: dispersion_operator's real part is (gain-loss)/2 [1/m],
        # so a scalar 10 dB/m loss must give real(D) = -db_to_np(10.0)/2 everywhere.
        grid = create_grid(2^8, 1e-12, 1550e-9)
        medium = Medium(; length=1.0, gamma=0.0, loss=10.0, betas=[0.0], lambda0=1550e-9)
        D = dispersion_operator(grid, medium)
        @test all(x -> isapprox(real(x), -db_to_np(10.0) / 2; atol=1e-12), D)
    end

    @testset "Photon energy" begin
        # Well-known reference values: hc/lambda in eV
        eV = 1.602176634e-19
        @test photon_energy(1550e-9) / eV ≈ 0.7999 rtol=1e-3
        @test photon_energy(1000e-9) / eV ≈ 1.2398 rtol=1e-3
        # Scales as 1/lambda
        @test photon_energy(500e-9) ≈ 2 * photon_energy(1000e-9)
    end

    @testset "n2/Aeff <-> gamma" begin
        n2 = 2.6e-20   # fused silica, m^2/W
        Aeff = 80e-12  # 80 um^2, typical SMF
        lambda0 = 1550e-9
        gamma = n2_aeff_to_gamma(n2, lambda0, Aeff)
        # Typical SMF-28-like gamma is ~1-1.5 /W/km
        @test 0.5e-3 < gamma < 3.0e-3
        @test gamma_aeff_to_n2(gamma, lambda0, Aeff) ≈ n2
        @test_throws ArgumentError n2_aeff_to_gamma(n2, -lambda0, Aeff)
        @test_throws ArgumentError n2_aeff_to_gamma(n2, lambda0, -Aeff)
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
