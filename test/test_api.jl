using Test
using Soliton

@testset verbose = true "API" begin
    @testset "Medium" begin
        medium = Medium(0.15, 0.11, 0.0, [-1.0e-26], 835e-9)
        @test medium.length == 0.15
        @test medium.gamma == 0.11
        @test medium.loss == 0.0
        @test medium.dispersion isa TaylorDispersion
        @test medium.dispersion.betas == [-1.0e-26]
        @test medium.lambda0 == 835e-9

        @test_throws ArgumentError Medium(-0.1, 0.11, 0.0, [-1.0e-26], 835e-9)
        @test_throws ArgumentError Medium(0.15, -0.1, 0.0, [-1.0e-26], 835e-9)
        @test_throws ArgumentError Medium(0.15, 0.11, -1.0, [-1.0e-26], 835e-9)
        @test_throws ArgumentError Medium(0.15, 0.11, 0.0, [-1.0e-26], -835e-9)
    end

    @testset "Medium keyword constructor" begin
        m = Medium(; length=0.15, gamma=0.11, betas=[-1.0e-26], lambda0=835e-9)
        @test m.length == 0.15
        @test m.loss == 0.0   # default
        @test m.dispersion.betas == [-1.0e-26]
        # mixed Int/Float arguments are promoted
        @test Medium(; length=1, gamma=0, loss=0, betas=[-1.0e-26], lambda0=835e-9) isa
            Medium
        # exactly one of betas / dispersion required
        @test_throws ArgumentError Medium(; length=0.1, gamma=0.1, lambda0=1e-6)
        @test_throws ArgumentError Medium(;
            length=0.1,
            gamma=0.1,
            betas=[1.0e-26],
            dispersion=TaylorDispersion([1.0e-26]),
            lambda0=1e-6,
        )
    end

    @testset "Dispersion models" begin
        grid = create_grid(2^9, 10e-12, 835e-9)
        # Taylor and an equivalent tabulated curve must agree on-grid
        b2 = -1.0e-26
        taylor = TaylorDispersion([b2])
        Btaylor = propagation_constant(grid.V, taylor)
        tab = TabulatedDispersion(grid.V, Btaylor)
        @test propagation_constant(grid.V, tab) ≈ Btaylor
        @test_throws ArgumentError TabulatedDispersion([1.0], [1.0])          # too few
        @test_throws ArgumentError TabulatedDispersion([2.0, 1.0], [0.0, 0.0]) # unsorted
        # An empty Taylor expansion means no dispersion (pure SPM)
        @test propagation_constant(grid.V, TaylorDispersion(Float64[])) == zeros(grid.N)
    end

    @testset "Sellmeier dispersion" begin
        # Fused Silica Sellmeier coefficients (in microns^2 for C)
        B = [0.6961663, 0.4079426, 0.8974794]
        C = [4.67914826e-3, 1.35120631e-2, 97.9340025]

        # Test constructor and defaults
        sd = SellmeierDispersion(B, C; microns=true)
        @test sd.B == B
        @test sd.C == C .* 1e-12

        @test_throws ArgumentError SellmeierDispersion([1.0], [1.0, 2.0])

        grid = create_grid(2^9, 10e-12, 835e-9)
        omega0 = grid.omega0

        # Test propagation_constant
        B_vals = propagation_constant(grid.V, sd, omega0)
        @test length(B_vals) == grid.N
        @test B_vals[grid.N ÷ 2 + 1] == 0.0 # B(0) = 0

        # Test that dispersion_operator works with Medium constructed with SellmeierDispersion
        medium = Medium(0.15, 0.11, 0.0, sd, 835e-9)
        D = dispersion_operator(grid, medium)
        @test length(D) == grid.N
        @test real(D[grid.N ÷ 2 + 1]) == 0.0 # Loss is 0, real part at center should be 0
    end

    @testset "Raman models" begin
        @test BlowWood().fr == 0.18
        @test LinAgrawal().fr == 0.245
        @test Hollenbeck().fr == 0.20
        # Time constants are in SI seconds
        @test BlowWood().tau1 < 1e-12
        @test LinAgrawal().taub < 1e-12
        # Keyword overrides
        @test BlowWood(; fr=0.10).fr == 0.10
    end

    @testset "SimParams" begin
        medium = Medium(0.15, 0.11, 0.0, [-1.0e-26], 835e-9)
        params = SimParams(; medium=medium, z_saves=50)
        @test params.z_saves == 50
        @test params.raman_model isa BlowWood
        @test params.self_steepening == false

        @test_throws ArgumentError SimParams(; medium=medium, z_saves=0)
        @test_throws ArgumentError SimParams(; medium=medium, rtol=-1e-3)
        @test_throws ArgumentError SimParams(; medium=medium, atol=-1e-4)

        nr = SimParams(; medium=medium, raman_model=nothing)
        @test nr.raman_model === nothing

        # Test save_freq=false memory optimization
        sf_false = SimParams(; medium=medium, z_saves=10, save_freq=false)
        @test sf_false.save_freq == false
        grid_sf = create_grid(2^9, 10e-12, 835e-9)
        pulse_sf = sech_pulse(grid_sf, 10.0, 100e-15)
        sol_sf = solve(pulse_sf, sf_false; progress=false)
        @test size(sol_sf.AW) == (0, 0)
        @test size(sol_sf.At) == (grid_sf.N, 10)

        # Test in-place loss_vector! and gain_vector!
        buf_loss = zeros(Float64, grid_sf.N)
        loss_vector!(buf_loss, grid_sf, medium)
        @test all(buf_loss .== 0.0)
        gain_vector!(buf_loss, grid_sf, medium)
        @test all(buf_loss .== 0.0)

        # Test multithreaded solve_sweep parameter sweep API
        powers_sweep = [50.0, 100.0, 200.0]
        sols_sweep = solve_sweep(powers_sweep; progress=false) do P0
            p = sech_pulse(grid_sf, P0, 100e-15)
            params = SimParams(; medium=medium, z_saves=5, raman_model=nothing)
            return (p, params)
        end
        @test length(sols_sweep) == 3
        @test sols_sweep[1] isa Solution
        @test peak_power(Pulse(sols_sweep[3])) > peak_power(Pulse(sols_sweep[1]))

        # Test solve_sweep with Vector of pulses and params
        pulses_v = [sech_pulse(grid_sf, P0, 100e-15) for P0 in powers_sweep]
        params_v = [SimParams(; medium=medium, z_saves=5, raman_model=nothing) for _ in 1:3]
        sols_v = solve_sweep(pulses_v, params_v; progress=false)
        @test length(sols_v) == 3
    end

    @testset "Pulses" begin
        grid = create_grid(2^12, 20e-12, 835e-9)

        sech = sech_pulse(grid, 100.0, 150e-15)
        @test length(sech.At) == grid.N
        @test length(sech.AW) == grid.N
        @test peak_power(sech) ≈ 100.0 rtol = 2e-3   # grid-discretization of the peak
        @test fwhm(sech; domain=:time) ≈ 150e-15 rtol = 5e-3

        gauss = gaussian_pulse(grid, 100.0, 150e-15)
        @test peak_power(gauss) ≈ 100.0 rtol = 2e-3
        @test fwhm(gauss; domain=:time) ≈ 150e-15 rtol = 1e-3

        cw = cw_pulse(grid, 25.0)
        @test all(abs2.(cw.At) .≈ 25.0)

        @test_throws ArgumentError sech_pulse(grid, -1.0, 150e-15)
        @test_throws ArgumentError sech_pulse(grid, 100.0, -150e-15)
    end

    @testset "build_physics_model" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.1, 0.11, 0.0, [-1.0e-26], 835e-9)

        m1 = build_physics_model(
            grid, SimParams(; medium=medium, raman_model=nothing, self_steepening=false)
        )
        @test m1.RW === nothing
        @test length(m1.D) == grid.N
        # Self-steepening off: W is the constant ω₀
        @test all(m1.W .== grid.omega0)

        m2 = build_physics_model(
            grid, SimParams(; medium=medium, raman_model=BlowWood(), self_steepening=true)
        )
        @test m2.RW !== nothing
        @test m2.fr == 0.18
        # Self-steepening on: W carries the absolute frequency ω₀ + Δω
        @test !all(m2.W .== grid.omega0)
    end

    @testset "Z-dependent gamma" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 1000.0, 100e-15)

        # Define z-dependent gamma function
        g(z) = 0.11 * exp(-z)

        medium = Medium(0.15, g, 0.0, [-1.0e-26], 835e-9)
        @test medium.gamma isa Function
        @test medium.gamma(0.0) == 0.11
        @test medium.gamma(0.1) ≈ 0.11 * exp(-0.1)

        # Run solver with z-dependent gamma
        params = SimParams(; medium=medium, z_saves=50, raman_model=nothing)
        sol = solve(pulse, params; progress=false)
        @test length(sol.Z) == 50
        @test size(sol.At, 2) == 50
    end

    @testset "Cascaded propagation & Lumped Elements" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = gaussian_pulse(grid, 10.0, 200e-15) # Pmax = 10 W

        # Test Amplifier
        amp = Amplifier(6.0) # 6 dB gain
        pulse_amp = apply(pulse, amp)
        @test peak_power(pulse_amp) ≈ peak_power(pulse) * 10.0^(6.0 / 10.0) rtol=1e-5

        # Test Attenuator
        att = Attenuator(3.0) # 3 dB loss
        pulse_att = apply(pulse, att)
        @test peak_power(pulse_att) ≈ peak_power(pulse) * 10.0^(-3.0 / 10.0) rtol=1e-5

        # Test Filter
        # Simple super-Gaussian low-pass filter in frequency
        filt_func(w) = abs(w - grid.omega0) < 1e12 ? 1.0 : 0.0
        filt = Filter(filt_func)
        pulse_filt = apply(pulse, filt)
        @test peak_power(pulse_filt) < peak_power(pulse)

        # Test Cascade: Fiber -> Amplifier -> Filter -> Fiber
        medium1 = Medium(0.01, 0.11, 0.0, [-1.0e-26], 835e-9)
        medium2 = Medium(0.01, 0.11, 0.0, [-1.0e-26], 835e-9)

        stage1 = SimParams(; medium=medium1, z_saves=5, raman_model=nothing)
        stage2 = amp
        stage3 = filt
        stage4 = SimParams(; medium=medium2, z_saves=5, raman_model=nothing)

        cascade = [stage1, stage2, stage3, stage4]
        results = solve(pulse, cascade; progress=false)

        @test length(results) == 4
        @test results[1] isa Solution
        @test results[2] isa Pulse
        @test results[3] isa Pulse
        @test results[4] isa Solution

        # Verify energy flow
        # Output of stage 1 final slice energy
        e1 = sum(abs2, results[1].At[:, end])
        # Output of stage 2 (amplifier) energy
        e2 = sum(abs2, results[2].At)
        @test e2 ≈ e1 * 10.0^(amp.gain_db / 10.0) rtol=1e-5

        # Test piping |> interface
        # pulse |> stage1 |> stage2 |> stage3 |> stage4
        piped_result = pulse |> stage1 |> stage2 |> stage3 |> stage4
        @test piped_result isa Solution
        @test piped_result.At[:, end] ≈ results[4].At[:, end]
    end

    @testset "Frequency-dependent and Effective-area Nonlinearity" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        # 1. ConstantNonlinearity matches Float64 gamma
        m_float = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        m_const = Medium(0.02, ConstantNonlinearity(0.11), 0.0, [-1.0e-26], 835e-9)

        params_float = SimParams(; medium=m_float, self_steepening=true)
        params_const = SimParams(; medium=m_const, self_steepening=true)

        sol_float = solve(pulse, params_float; progress=false)
        sol_const = solve(pulse, params_const; progress=false)

        @test sol_float.At[:, end] ≈ sol_const.At[:, end]

        # 2. FrequencyDependentNonlinearity
        # Let's define a flat frequency-dependent gamma
        g_func(w) = 0.11
        m_freq = Medium(
            0.02, FrequencyDependentNonlinearity(g_func), 0.0, [-1.0e-26], 835e-9
        )
        params_freq = SimParams(; medium=m_freq, self_steepening=false)
        sol_freq = solve(pulse, params_freq; progress=false)

        # Without self-steepening, a flat frequency-dependent gamma matches flat constant gamma
        params_no_shock = SimParams(; medium=m_const, self_steepening=false)
        sol_no_shock = solve(pulse, params_no_shock; progress=false)
        @test sol_freq.At[:, end] ≈ sol_no_shock.At[:, end]

        # 3. NonlinearityFromEffectiveArea
        # n2 = 2.5e-20 m^2/W, Aeff(w) = 10e-12 m^2 (constant 10 um^2)
        # gamma(omega0) = n2 * omega0 / (c * Aeff(omega0))
        c_const = 299792458.0
        n2 = 2.5e-20
        Aeff_const = 10e-12
        gamma_expected = n2 * grid.omega0 / (c_const * Aeff_const)

        m_area = Medium(
            0.02,
            NonlinearityFromEffectiveArea(n2, w -> Aeff_const),
            0.0,
            [-1.0e-26],
            835e-9,
        )
        params_area = SimParams(; medium=m_area, self_steepening=true)
        sol_area = solve(pulse, params_area; progress=false)

        # Should match ConstantNonlinearity with the calculated gamma_expected
        m_expected = Medium(
            0.02, ConstantNonlinearity(gamma_expected), 0.0, [-1.0e-26], 835e-9
        )
        params_expected = SimParams(; medium=m_expected, self_steepening=true)
        sol_expected = solve(pulse, params_expected; progress=false)

        @test sol_area.At[:, end] ≈ sol_expected.At[:, end]

        # 4. Waveguide Mode Overlap & Marcuse Aeff
        a = 4.1e-6 # SMF-28 core radius 4.1 um
        NA = 0.14  # NA
        lam = 1550e-9

        aeff_val = step_index_aeff(a, NA, lam)
        @test isapprox(aeff_val, 6.675e-11; rtol=0.05)

        marcuse = MarcuseAeff(a, NA)
        nl_marcuse = NonlinearityFromEffectiveArea(2.6e-20, marcuse)
        m_marcuse = Medium(0.1, nl_marcuse, 0.0, [-21.5e-27], 1550e-9)
        sol_marcuse = solve(
            pulse, SimParams(; medium=m_marcuse, raman_model=nothing); progress=false
        )
        @test sol_marcuse.Z[end] == 0.1

        # 5. b_integral resolves every NonlinearityModel variant to the same
        # physical γ, so all four equivalent media give the same B-integral.
        @test b_integral(sol_const, m_const) ≈ b_integral(sol_float, m_float)
        @test b_integral(sol_freq, m_freq) ≈ b_integral(sol_no_shock, m_const)
        @test b_integral(sol_area, m_area) ≈ b_integral(sol_expected, m_expected)
    end
end
