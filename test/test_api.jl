using Test
using JuGNLSE

@testset "API" begin
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
        @test Medium(; length=1, gamma=0, loss=0, betas=[-1.0e-26],
                     lambda0=835e-9) isa Medium
        # exactly one of betas / dispersion required
        @test_throws ArgumentError Medium(; length=0.1, gamma=0.1, lambda0=1e-6)
        @test_throws ArgumentError Medium(; length=0.1, gamma=0.1,
            betas=[1.0e-26], dispersion=TaylorDispersion([1.0e-26]), lambda0=1e-6)
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

        @test_throws ArgumentError SimParams(;
            medium=medium, z_saves=0)
        @test_throws ArgumentError SimParams(;
            medium=medium, rtol=-1e-3)
        @test_throws ArgumentError SimParams(;
            medium=medium, atol=-1e-4)

        nr = SimParams(; medium=medium, raman_model=nothing)
        @test nr.raman_model === nothing
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

        m1 = build_physics_model(grid, SimParams(; medium=medium,
            raman_model=nothing, self_steepening=false), ConstantGamma(medium.gamma))
        @test m1.RW === nothing
        @test length(m1.D) == grid.N
        # Self-steepening off: W is the constant ω₀
        @test all(m1.W .== grid.omega0)

        m2 = build_physics_model(grid, SimParams(; medium=medium,
            raman_model=BlowWood(), self_steepening=true), ConstantGamma(medium.gamma))
        @test m2.RW !== nothing
        @test m2.fr == 0.18
        # Self-steepening on: W carries the absolute frequency ω₀ + Δω
        @test !all(m2.W .== grid.omega0)
    end

    @testset "Gamma Coefficients" begin
        # Test ConstantGamma
        const_gamma = ConstantGamma(0.12)
        @test gamma(const_gamma, 800e-9, 0.0) == 0.12
        @test gamma(const_gamma, 1000e-9, 1.0) == 0.12

        # Test ZDependentGamma
        # Linear tapering from 0.1 to 0.2 over 1m
        z_taper_func(z) = 0.1 + (0.2 - 0.1) * z
        z_gamma = ZDependentGamma(z_taper_func)
        @test gamma(z_gamma, 800e-9, 0.0) ≈ 0.1
        @test gamma(z_gamma, 800e-9, 0.5) ≈ 0.15
        @test gamma(z_gamma, 800e-9, 1.0) ≈ 0.2

        # Test WavelengthDependentGamma
        # Simple quadratic dependency on wavelength
        lambda_dep_func(lambda) = 0.1 * (lambda / 800e-9)^2
        lambda_gamma = WavelengthDependentGamma(lambda_dep_func)
        @test gamma(lambda_gamma, 800e-9, 0.0) ≈ 0.1
        @test gamma(lambda_gamma, 1600e-9, 0.0) ≈ 0.4 # (1600/800)^2 = 2^2 = 4
    end

    @testset "GNLSEProblem with custom gamma" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        initial_pulse = sech_pulse(grid, 100.0, 150e-15)

        # Test with ZDependentGamma
        z_taper_func(z) = 0.1 + (0.2 - 0.1) * z / 0.15 # Taper over fiber length 0.15m
        z_gamma_coeff = ZDependentGamma(z_taper_func)
        medium_z_dep = Medium(length=0.15, gamma=0.11, loss=0.0, betas=[-1.0e-26], lambda0=835e-9)
        sim_params_z_dep = SimParams(; medium=medium_z_dep)

        problem_z_dep = GNLSEProblem(initial_pulse, medium_z_dep, sim_params_z_dep, z_gamma_coeff)

        # Solve for a short distance to check gamma at z=0 and z=L
        solution_z_dep = solve(problem_z_dep, ERK4IP(); progress=false)
        # Need to access the gamma value during propagation. This is harder to test directly from solution.
        # For now, we can check if the problem can be constructed and solved without error.
        @test problem_z_dep.gamma_coefficient isa ZDependentGamma
        # A more robust test would involve inspecting the gamma value during the solve! process,
        # which would require modifying the solver to return intermediate gamma values, or
        # using a mock gamma function that logs calls. For a simple check, just ensure it runs.


        # Test with WavelengthDependentGamma
        lambda_dep_func(lambda) = 0.1 * (lambda / grid.lambda0)^2
        lambda_gamma_coeff = WavelengthDependentGamma(lambda_dep_func)
        medium_lambda_dep = Medium(length=0.15, gamma=0.11, loss=0.0, betas=[-1.0e-26], lambda0=835e-9)
        sim_params_lambda_dep = SimParams(; medium=medium_lambda_dep)

        problem_lambda_dep = GNLSEProblem(initial_pulse, medium_lambda_dep, sim_params_lambda_dep, lambda_gamma_coeff)
        @test problem_lambda_dep.gamma_coefficient isa WavelengthDependentGamma
        solution_lambda_dep = solve(problem_lambda_dep, ERK4IP(); progress=false)
    end
end
