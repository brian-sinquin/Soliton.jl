using Test
using JuGNLSE

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
            raman_model=nothing, self_steepening=false))
        @test m1.RW === nothing
        @test length(m1.D) == grid.N
        # Self-steepening off: W is the constant ω₀
        @test all(m1.W .== grid.omega0)

        m2 = build_physics_model(grid, SimParams(; medium=medium,
            raman_model=BlowWood(), self_steepening=true))
        @test m2.RW !== nothing
        @test m2.fr == 0.18
        # Self-steepening on: W carries the absolute frequency ω₀ + Δω
        @test !all(m2.W .== grid.omega0)
    end
end
