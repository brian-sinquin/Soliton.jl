using Test
using FFTW
using Random
using GNLSE

@testset "Unit" begin
    @testset "Grid" begin
        N = 2^10
        grid = create_grid(N, 10e-12, 835e-9)
        @test grid.N == N
        @test length(grid.t) == N
        @test length(grid.V) == N
        @test length(grid.W) == N
        @test grid.dt > 0
        @test grid.omega0 ≈ 2π * GNLSE.c / 835e-9

        # V is monotonic and contains a zero element
        @test issorted(grid.V)
        @test minimum(abs.(grid.V)) < 1e-6 * maximum(abs.(grid.V))

        # Time window spans the requested duration
        @test grid.t[end] - grid.t[1] ≈ 10e-12

        # Absolute optical frequency is W = ω₀ + V
        @test grid.W ≈ grid.omega0 .+ grid.V
    end

    @testset "Dispersion operator" begin
        N = 2^10
        grid = create_grid(N, 10e-12, 835e-9)
        b2 = -1.0e-26
        medium = Medium(0.1, 0.0, 0.0, [b2], 835e-9)
        D = dispersion_operator(grid, medium)

        @test length(D) == N
        @test eltype(D) <: Complex

        # At V = 0: no dispersion, no loss
        iz = N ÷ 2 + 1
        @test abs(grid.V[iz]) < 1e-6
        @test abs(D[iz]) < 1e-12

        # β₂ term: B = β₂/2 · V²  ⇒  D = i·β₂/2·V²
        k = iz + 50
        @test imag(D[k]) ≈ 0.5 * b2 * grid.V[k]^2 rtol = 1e-10
        @test real(D[k]) ≈ 0.0 atol = 1e-12

        # β₃ term: B includes β₃/6 · V³
        b3 = 5.0e-41
        D3 = dispersion_operator(grid, Medium(0.1, 0.0, 0.0, [0.0, b3], 835e-9))
        @test imag(D3[k]) ≈ b3 / 6 * grid.V[k]^3 rtol = 1e-10

        # Loss enters as a real, negative damping -α/2
        lossy = Medium(0.1, 0.0, 2.0, [b2], 835e-9)
        Dl = dispersion_operator(grid, lossy)
        alpha = log(10.0^(2.0 / 10.0))
        @test real(Dl[iz]) ≈ -alpha / 2 rtol = 1e-12
    end

    @testset "Raman response" begin
        grid = create_grid(2^12, 10e-12, 835e-9)
        for model in (BlowWood(), LinAgrawal(), Hollenbeck())
            fr, RT = raman_response(grid, model)
            @test fr == model.fr
            @test length(RT) == grid.N
            @test eltype(RT) <: Real
            # Causality: response vanishes for t < 0
            @test all(abs.(RT[grid.t .< 0]) .< 1e-9 * maximum(abs.(RT)))
            # Non-trivial response for t > 0
            @test maximum(abs.(RT[grid.t .> 0])) > 0
            # Unit-area normalization: ∫h_R(t)dt = 1, so the delayed-Raman
            # convolution in `_spm_raman` (fr·dt·conv(|A|²,RT)) reduces to a
            # true weighted average. Regression test for the Hollenbeck model,
            # whose empirical oscillator amplitudes do NOT integrate to unity
            # on their own and previously were left unnormalized (off by a
            # factor of ~1e11).
            @test sum(RT) * grid.dt ≈ 1.0 rtol = 1e-2
        end
    end

    @testset "Analysis" begin
        grid = create_grid(2^12, 20e-12, 835e-9)
        FWHM = 200e-15
        pulse = gaussian_pulse(grid, 5.0, FWHM)

        @test peak_power(pulse) ≈ 5.0 rtol = 2e-3   # grid-discretization of the peak
        @test pulse_energy(pulse) > 0
        @test fwhm(pulse; domain=:time) ≈ FWHM rtol = 1e-3
        @test spectral_bandwidth(pulse) > 0
        # Transform-limited Gaussian: TBP ≈ 0.441
        @test time_bandwidth_product(pulse) ≈ 0.441 rtol = 0.05
        @test_throws ArgumentError fwhm(pulse; domain=:bogus)

        # A centered, symmetric pulse has its spectral centroid at 0
        @test abs(spectral_centroid(pulse)) < 1e-3 * maximum(abs.(grid.V))
        @test photon_number(pulse) > 0
    end

    @testset "Wavelength grid & soliton metrics" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        λ = wavelength_grid(grid)
        @test length(λ) == grid.N
        # at V = 0 the wavelength equals λ₀
        iz = grid.N ÷ 2 + 1
        @test λ[iz] ≈ 835e-9 rtol = 1e-9

        # Textbook scaling relations
        b2, γ, T0, P0 = -1.0e-26, 0.1, 100e-15, 50.0
        @test dispersion_length(b2, T0) ≈ T0^2 / abs(b2)
        @test nonlinear_length(γ, P0) ≈ 1 / (γ * P0)
        @test soliton_number(b2, γ, T0, P0) ≈
            sqrt(dispersion_length(b2, T0) / nonlinear_length(γ, P0))
    end

    @testset "Noise & coherence" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 1000.0, 100e-15)

        # add_noise perturbs the field but keeps structure and (nearly) energy
        np = add_noise(pulse; rng=MersenneTwister(1))
        @test np isa Pulse
        @test length(np.At) == grid.N
        @test np.At != pulse.At
        @test pulse_energy(np) ≈ pulse_energy(pulse) rtol = 1e-2
        # reproducible per seed, independent across seeds
        @test add_noise(pulse; rng=MersenneTwister(1)).At ==
            add_noise(pulse; rng=MersenneTwister(1)).At
        @test add_noise(pulse; rng=MersenneTwister(1)).At !=
            add_noise(pulse; rng=MersenneTwister(2)).At

        # spectral_coherence: identical realizations ⇒ g = 1
        base = randn(ComplexF64, 64)
        @test all(spectral_coherence([copy(base) for _ in 1:8]) .≈ 1.0)
        # fully independent realizations ⇒ g ≈ 0
        indep = [randn(ComplexF64, 64) for _ in 1:300]
        @test maximum(spectral_coherence(indep)) < 0.3
        @test_throws ArgumentError spectral_coherence([randn(ComplexF64, 64)])
        @test_throws ArgumentError spectral_coherence([
            randn(ComplexF64, 4), randn(ComplexF64, 5)
        ])
    end
end
