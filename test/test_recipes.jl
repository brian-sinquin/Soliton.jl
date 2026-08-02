using Test
using JuGNLSE
using Plots

@testset "Plot Recipes" begin
    grid = create_grid(2^9, 5e-12, 1550e-9)
    pulse = sech_pulse(grid, 100.0, 100e-15)
    medium = Medium(0.05, 0.0011, 0.0, [-21.5e-27], 1550e-9)
    sol = solve(
        pulse, SimParams(; medium=medium, z_saves=5, raman_model=nothing); progress=false
    )

    p = plot(sol)
    @test p isa Plots.Plot
    @test length(p.subplots) == 4
end
