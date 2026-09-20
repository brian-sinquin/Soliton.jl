using Test
using Soliton
using FFTW
using Random
using Statistics

@testset "Public export coverage" begin
    grid = create_grid(2^12, 20e-12, 1550e-9)
    P0, width = 25.0, 500e-15
    pulse = gaussian_pulse(grid, P0, width)

    @testset "Lorentzian envelope" begin
        p = lorentzian_pulse(grid, P0, width)
        expected = sqrt(P0) ./ (1 .+ (2sqrt(sqrt(2) - 1) .* grid.t ./ width).^2)
        @test p.At ≈ expected
        @test peak_power(p) ≈ P0 rtol=1e-3
        @test fwhm(p) ≈ width rtol=1e-3
    end

    @testset "RIN and seeded noise" begin
        # Flat -100 dBc/Hz over 1 MHz gives 1% RMS power fluctuations.
        rin = rin_rms(-100, 1e6)
        @test rin ≈ 0.01
        rng = MersenneTwister(42)
        fluctuations = [pulse_energy(add_noise(pulse; photons_per_mode=0,
                            rin, rng)) / pulse_energy(pulse) - 1 for _ in 1:2000]
        @test abs(mean(fluctuations)) < 3rin / sqrt(length(fluctuations))
        @test std(fluctuations) ≈ rin rtol=0.06
        for model in (:gaussian, :phase_only)
            a = add_noise(pulse; rng=MersenneTwister(123), quantum_model=model)
            b = add_noise(pulse; rng=MersenneTwister(123), quantum_model=model)
            @test a.At == b.At
            @test a.AW == b.AW
            noise_energy = sum(abs2, a.At .- pulse.At) * grid.dt
            expected = 1.054571817e-34 * sum(abs, grid.W)
            @test noise_energy ≈ expected rtol=(model === :gaussian ? 0.06 : 1e-10)
        end
    end

    @testset "Photon invariant and spectral shift" begin
        # photon_number omits N*dt/ħ; restore it for an absolute count.
        count = photon_number(pulse) * grid.N * grid.dt / 1.054571817e-34
        @test count ≈ pulse_energy(pulse) / (1.054571817e-34 * grid.omega0) rtol=1e-4
        shift = 12 * (grid.V[2] - grid.V[1])
        # AW = ifft(At): exp(-i*shift*t) produces positive detuning.
        At = pulse.At .* cis.(-shift .* grid.t)
        shifted = Pulse(At, ifft(At), grid)
        @test spectral_centroid(shifted) ≈ shift rtol=1e-10
    end

    @testset "Characteristic lengths" begin
        beta2, T0, gamma, power = -20e-27, 100e-15, 0.01, 800.0
        @test dispersion_length(beta2, T0) ≈ 0.5
        @test nonlinear_length(gamma, power) ≈ 0.125
        @test soliton_number(beta2, gamma, T0, power) ≈ 2.0
    end

    @testset "Spectrogram" begin
        delays, frequencies, spectrum = spectrogram(pulse; n_delay=31, gate_fwhm=width)
        @test length(delays) == 31
        @test frequencies == grid.V
        @test size(spectrum) == (grid.N, 31)
        @test all(isfinite, spectrum)
        @test all(>=(0), spectrum)
        @test maximum(spectrum) > 0
    end

    @testset "Fundamental soliton tracking" begin
        T0, beta2, gamma = 100e-15, -20e-27, 0.01
        LD = T0^2 / abs(beta2)
        power = 1 / (gamma * LD)
        g = create_grid(2^11, 40T0, 1550e-9)
        p = sech_pulse(g, power, 2asinh(1) * T0)
        medium = Medium(LD, gamma, 0.0, [beta2], 1550e-9)
        sol = solve(p, SimParams(; medium, z_saves=11, raman_model=nothing,
                    self_steepening=false, solver=ERK4IP()); progress=false)
        z_peak, powers, centroids = track_solitons(sol)
        @test length(powers) == length(centroids) == 11
        @test all(x -> isapprox(x, power; rtol=1e-3), powers)
        @test maximum(abs, centroids) * T0 < 1e-8
        # N=1 does not fission; the diagnostic still selects a sampled maximum.
        @test z_peak == sol.Z[argmax(powers)]
    end

    @testset "Dispersive wave phase matching" begin
        # Choose β₃ so the cubic phase mismatch has a known positive root,
        # including the nonlinear γP/2 contribution (not the fallback formula).
        beta2, gamma, detuning = -20e-27, 0.01, 20e12
        beta3 = 6 * (gamma * P0 / 2 - beta2 * detuning^2 / 2) / detuning^3
        medium = Medium(0.1, gamma, 0.0, [beta2, beta3], 1550e-9)
        expected = 2π * c / (grid.omega0 + detuning)
        @test dispersive_wave_wavelength(medium, pulse; P0) ≈ expected rtol=1e-5
    end
end

@testset "Solver × medium × Raman × shock matrix" begin
    for solver in (ERK4IP(), SSFM(1e-5)), raman in (nothing, BlowWood()), shock in (true, false)
        @testset "$(typeof(solver)), $(typeof(raman)), shock=$shock" begin
            grid = create_grid(2^10, 5e-12, 1550e-9)
            medium = Medium(0.01, 0.01, 0.0, [-20e-27], 1550e-9)
            pulse = gaussian_pulse(grid, 100.0, 100e-15)
            params = SimParams(; medium, solver, raman_model=raman,
                               self_steepening=shock, z_saves=5)
            sol = solve(pulse, params; progress=false)
            @test all(isfinite, sol.At)
            @test all(isfinite, sol.AW)
            @test sol.Z[end] ≈ medium.length
            energies = vec(sum(abs2, sol.At; dims=1)) .* grid.dt
            # Short, lossless propagation: allow 1% drift for Euler and shock/Raman.
            @test all(e -> isapprox(e, pulse_energy(pulse); rtol=0.01), energies)
        end
    end
end
