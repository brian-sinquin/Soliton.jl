using Test, Soliton, FFTW, LinearAlgebra

@testset "Audit Critical Fixes" begin

    @testset "C2 solution pulse spectral ordering" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        medium = Medium(length=0.1, gamma=0.0, betas=[0.0], lambda0=1550e-9)
        sol = solve(pulse, SimParams(medium=medium, z_saves=2,
            raman_model=nothing, self_steepening=false); progress=false)
        result = Pulse(sol)
        # A consistent Pulse must satisfy AW == ifft(At)
        @test norm(result.AW - ifft(result.At)) / norm(result.AW) < 1e-10
    end

    @testset "C3 filter spectral ordering" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        vpulse = VectorialPulse(pulse.At, 0.5im .* pulse.At, grid)
        # Broadband passband centered on carrier — should retain >99% energy
        filt = Filter(w -> abs(w - grid.omega0) < 1e13 ? 1.0 : 0.0)
        for input in (pulse, vpulse)
            @test pulse_energy(apply(input, filt)) / pulse_energy(input) > 0.99
        end
    end

    @testset "C4 zero-gain amplifier Kerr normalization" begin
        grid = create_grid(256, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1.0, 1e-12)
        # With g0=0, amplifying medium Kerr should match passive medium exactly
        passive = Medium(length=0.1, gamma=0.7, betas=[0.0], lambda0=1550e-9)
        active = AmplifyingMedium(length=0.1, gamma=0.7, g0=0.0,
            Esat=1e-9, betas=[0.0], lambda0=1550e-9)
        models = map((passive, active)) do medium
            build_physics_model(grid, SimParams(medium=medium, z_saves=2,
                raman_model=nothing, self_steepening=false))
        end
        rhs = map(m -> copy(m.nonlinear_function(pulse.At, m, 0.03)), models)
        @test norm(rhs[2] - rhs[1]) / norm(rhs[1]) < 1e-12
    end

    @testset "H4 grid validation" begin
        # N < 2 must be rejected
        @test_throws ArgumentError create_grid(0, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(1, 10e-12, 1550e-9)
        # Odd N must be rejected
        @test_throws ArgumentError create_grid(3, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(5, 10e-12, 1550e-9)
        @test_throws ArgumentError create_grid(127, 10e-12, 1550e-9)
        # Even N >= 2 should work
        grid2 = create_grid(2, 10e-12, 1550e-9)
        @test length(grid2.t) == 2
        @test length(grid2.V) == 2
        grid6 = create_grid(6, 10e-12, 1550e-9)
        @test length(grid6.t) == 6
        @test length(grid6.V) == 6
    end

end

@testset "Audit review regressions" begin
    grid = create_grid(256, 2e-12, 1550e-9)
    seed = gaussian_pulse(grid, 2.0, 150e-15)
    u = seed.At .* cis.(3 .* 2π .* grid.t ./ (grid.N * grid.dt))
    pulse = Pulse(u, ifft(u), grid)

    @testset "C2 cascades and optional spectra" begin
        for solver in (ERK4IP(), SSFM(0.002), AdaptiveSSFM()), save_freq in (true, false)
            medium = Medium(length=0.02, gamma=0.0, betas=[-20e-27], lambda0=grid.lambda0)
            params = SimParams(; medium, solver, save_freq, z_saves=2, raman_model=nothing,
                self_steepening=false)
            sol = solve(pulse, params; progress=false)
            restored = Pulse(sol)
            @test restored.AW ≈ ifft(restored.At) rtol=1e-12
            longer = Medium(length=0.04, gamma=0.0, betas=[-20e-27], lambda0=grid.lambda0)
            reference = solve(pulse, SimParams(; medium=longer, solver, z_saves=2,
                raman_model=nothing, self_steepening=false); progress=false)
            cascade = solve(pulse, [params, params]; progress=false)
            @test cascade[end].At[:, end] ≈ reference.At[:, end] rtol=1e-10
        end
    end

    @testset "H4 even FFT-bin propagation" begin
        for N in (2, 6, 10)
            small = create_grid(N, 2e-12, 1550e-9)
            @test ifftshift(small.V) ≈ 2π .* FFTW.fftfreq(N, 1 / small.dt) rtol=1e-12
            # Include the even-grid Nyquist bin and both signs of detuning.
            for V in small.V
                tone = cis.(-V .* small.t)
                input = Pulse(tone, ifft(tone), small)
                beta2, L = -20e-27, 0.2
                medium = Medium(length=L, gamma=0.0, betas=[beta2], lambda0=small.lambda0)
                sol = solve(input, SimParams(; medium, z_saves=2, raman_model=nothing,
                    self_steepening=false); progress=false)
                @test sol.At[:, end] ≈ tone .* cis(beta2 * V^2 * L / 2) rtol=1e-10
            end
        end
    end

    @testset "C3 signed tones and complex transmission" begin
        # Exact DFT-bin tones: exp(-i V t) must receive H(omega0 + V).
        dw = 2π / (grid.N * grid.dt)
        H(w) = (0.6 + 0.1 * tanh((w - grid.omega0) / dw)) * cis((w - grid.omega0) * 2e-13)
        for k in (0, 7, -11)
            tone = cis.(-k * dw .* grid.t)
            other = 0.3im .* cis.(-(k + 3) * dw .* grid.t)
            scalar = Pulse(tone, ifft(tone), grid)
            vector = VectorialPulse(tone, other, grid)
            filtered = apply(scalar, Filter(H))
            filtered_v = apply(vector, Filter(H))
            @test filtered.At ≈ H(grid.omega0 + k * dw) .* tone rtol=1e-12
            @test filtered_v.At[:, 1] ≈ filtered.At rtol=1e-12
            @test filtered_v.At[:, 2] ≈ H(grid.omega0 + (k + 3) * dw) .* other rtol=1e-12
            @test filtered_v.AW ≈ ifft(filtered_v.At, 1) rtol=1e-12
            @test pulse_energy(filtered) / pulse_energy(scalar) ≈ abs2(H(grid.omega0 + k * dw)) rtol=1e-12
        end
    end

    @testset "C4 Kerr and gain have independent normalization" begin
        gammas = (0.7, z -> 0.7 * (1 + z), ConstantNonlinearity(0.7),
            FrequencyDependentNonlinearity(w -> 0.7 * w / grid.omega0),
            NonlinearityFromEffectiveArea(2.6e-20, w -> 80e-12))
        for gamma in gammas, raman in (nothing, BlowWood()), shock in (false, true), g0 in (0.0, 2.0)
            passive = Medium(length=0.1, gamma=gamma, betas=[0.0], lambda0=grid.lambda0)
            active = AmplifyingMedium(length=0.1, gamma=gamma, g0=g0,
                Esat=pulse_energy(pulse), betas=[0.0], lambda0=grid.lambda0)
            models = map((passive, active)) do medium
                build_physics_model(grid, SimParams(; medium, raman_model=raman,
                    self_steepening=shock, z_saves=2))
            end
            p_rhs = copy(models[1].nonlinear_function(u, models[1], 0.03))
            a_rhs = copy(models[2].nonlinear_function(u, models[2], 0.03))
            if raman === nothing && g0 == 0.0
                physical_gamma = if gamma isa Number
                    fill(gamma, grid.N)
                elseif gamma isa Function
                    fill(gamma(0.03), grid.N)
                elseif gamma isa ConstantNonlinearity
                    fill(gamma.gamma, grid.N)
                elseif gamma isa FrequencyDependentNonlinearity
                    gamma.gamma_function.(grid.W)
                else
                    gamma.n2 .* grid.W ./ (c .* gamma.Aeff_function.(grid.W))
                end
                if shock && (gamma isa Number || gamma isa Function || gamma isa ConstantNonlinearity)
                    physical_gamma .*= grid.W ./ grid.omega0
                end
                @test a_rhs ≈ im .* ifftshift(physical_gamma) .* ifft(abs2.(u) .* u) rtol=1e-12
            end
            # E=Esat: amplitude correction is -g0/4, independent of gamma and Raman.
            expected = p_rhs - (g0 / 4) .* ifft(u)
            @test a_rhs ≈ expected rtol=1e-12
        end
    end

    @testset "C4 exact SPM phase and saturated energy evolution" begin
        for raman in (nothing, BlowWood())
            # With gamma=0, Raman must not affect gain. Disable ASE explicitly.
            active = AmplifyingMedium(length=0.2, gamma=0.0, g0=2.0,
                Esat=pulse_energy(pulse), noise_figure_db=0.0,
                betas=[0.0], lambda0=grid.lambda0)
            params = SimParams(medium=active, z_saves=5, raman_model=raman,
                self_steepening=true, solver=ERK4IP(rtol=1e-9, atol=1e-11))
            model = build_physics_model(grid, params)
            # Exercise the production deterministic integrator without ASE: the
            # noise hook is a no-op when aux_data has no noise_figure_db key.
            fields = map(fieldnames(typeof(model))) do name
                name === :aux_data ? (; g0=active.g0, Esat=active.Esat) : getfield(model, name)
            end
            deterministic = Soliton.PhysicsModel(fields...)
            Z, At, AW = Soliton.propagate(deterministic, pulse, params, params.solver, false)
            sol = Solution(grid.t, grid.W, grid.omega0, Z, At, AW)
            E0 = pulse_energy(pulse)
            for j in eachindex(sol.Z)
                E = sum(abs2, sol.At[:, j]) * grid.dt
                # Integral of dE/dz = g0*E/(1+E/Esat), no loss or noise.
                @test log(E/E0) + (E-E0)/active.Esat ≈ active.g0 * sol.Z[j] atol=1e-8
                @test sol.At[:, j] ≈ u .* sqrt(E/E0) rtol=1e-8
            end
        end
        active = AmplifyingMedium(length=0.2, gamma=0.7, g0=0.0,
            Esat=1e-9, betas=[0.0], lambda0=grid.lambda0)
        sol = solve(pulse, SimParams(medium=active, z_saves=2, raman_model=nothing,
            self_steepening=false, solver=ERK4IP(rtol=1e-9, atol=1e-11)); progress=false)
        @test sol.At[:, end] ≈ u .* exp.(0.7im .* abs2.(u) .* active.length) rtol=1e-8
    end
end

@testset "Remaining review regressions" begin
    grid = create_grid(128, 2e-12, 1550e-9)
    dw = 2π / (grid.N * grid.dt)
    tone = cis.(-7dw .* grid.t)
    pulse = Pulse(tone, ifft(tone), grid)

    @testset "Vectorial identity and cascades" begin
        input = VectorialPulse(tone, 0.3im .* cis.(11dw .* grid.t), grid)
        dispersion = TaylorDispersion([0.0])
        medium = BirefringentMedium(0.02, 0.0, 0.0, dispersion, dispersion, 0.0, grid.lambda0)
        for solver in (SSFM(0.002),), save_freq in (true, false)
            params = SimParams(; medium, solver, save_freq, z_saves=2,
                raman_model=nothing, self_steepening=false)
            sol = solve(input, params; progress=false)
            restored = VectorialPulse(sol)
            for pol in 1:2
                @test restored.AW[:, pol] ≈ ifft(restored.At[:, pol]) rtol=1e-12
            end
            @test restored.At ≈ input.At rtol=1e-10
            @test solve(input, [params, params]; progress=false)[end].At[:, :, end] ≈ input.At rtol=1e-10
        end
    end

    @testset "Analysis frequency alignment" begin
        @test pulse_energy(pulse) ≈ sum(abs2, tone) * grid.dt
        @test spectral_centroid(pulse) ≈ 7dw rtol=1e-12
        @test photon_number(pulse) ≈ 1 / (grid.omega0 + 7dw) rtol=1e-12
        spec = fill(0.1 + 0.0im, grid.N)
        spec[grid.N ÷ 2 .+ (-2:2) .+ 1] .= 1
        broad = Pulse(fft(ifftshift(spec)), ifftshift(spec), grid)
        @test spectral_bandwidth(broad) ≈ 4dw / (2π)
        # Nonuniform coherence exposes a half-array permutation.
        spec2 = copy(spec)
        spec2[grid.N ÷ 2 + 1] = im
        second = Pulse(fft(ifftshift(spec2)), ifftshift(spec2), grid)
        expected = spectral_coherence([spec, spec2])
        @test spectral_coherence([broad, second]) ≈ expected
        for saved in (true, false)
            sols = [Solution(grid.t, grid.W, grid.omega0, [0.0],
                reshape(p.At, :, 1), saved ? reshape(fftshift(p.AW), :, 1) : zeros(ComplexF64, 0, 0))
                for p in (broad, second)]
            @test spectral_coherence(sols) ≈ expected atol=1e-12
            @test photon_number(sols[1])[1] ≈ photon_number(broad) rtol=1e-12
        end
        _, V, S = spectrogram(pulse; n_delay=3, gate_fwhm=1e-10)
        @test V[argmax(S[:, 2])] ≈ 7dw
        _, Vshg, frog = shg_frog_trace(pulse; n_delay=3)
        @test Vshg[argmax(frog[:, 2])] ≈ 14dw
    end

    @testset "C4 saturated gain and exact tone phase" begin
        # A single Fourier tone retains constant temporal intensity. Its energy
        # obeys ln(E/E0)+(E-E0)/Esat=g0*z, and phase is gamma_eff*∫P dz.
        for gamma in (0.7, FrequencyDependentNonlinearity(w -> 0.7w / grid.omega0)),
            shock in (false, true), raman in (nothing, BlowWood())
            active = AmplifyingMedium(length=0.1, gamma=gamma, g0=2.0,
                Esat=pulse_energy(pulse), betas=[0.0], lambda0=grid.lambda0)
            params = SimParams(medium=active, z_saves=4, raman_model=raman,
                self_steepening=shock, solver=ERK4IP(rtol=1e-10, atol=1e-12))
            model = build_physics_model(grid, params)
            fields = map(fieldnames(typeof(model))) do name
                name === :aux_data ? (; g0=active.g0, Esat=active.Esat) : getfield(model, name)
            end
            deterministic = Soliton.PhysicsModel(fields...)
            Z, At, _ = Soliton.propagate(deterministic, pulse, params, params.solver, false)
            E0 = pulse_energy(pulse)
            gamma_eff = 0.7 * ((shock || gamma isa FrequencyDependentNonlinearity) ?
                (grid.omega0 + 7dw) / grid.omega0 : 1.0)
            # Discrete Raman DC response determines the constant-intensity factor.
            raman_factor = raman === nothing ? 1.0 :
                1 - model.fr + model.fr * grid.dt * real(model.RW[1])
            for j in eachindex(Z)
                E = sum(abs2, At[:, j]) * grid.dt
                @test log(E/E0) + (E-E0)/active.Esat ≈ active.g0 * Z[j] atol=1e-9
                integrated_power = ((E-E0) + (E^2-E0^2)/(2active.Esat)) /
                    (active.g0 * grid.N * grid.dt)
                @test At[:, j] ≈ tone .* sqrt(E/E0) .* cis(gamma_eff * raman_factor * integrated_power) rtol=1e-8
            end
        end
    end
end
