using Test
using Soliton
using RecipesBase

# These tests exercise `src/recipes.jl`'s `@recipe` definition directly via
# `RecipesBase.apply_recipe`, which returns the same `Vector{RecipeData}`
# (subplot/seriestype attributes + plotted (x, y[, z]) args) that `Plots.plot`
# builds internally, *before* any backend renders it. This validates the
# recipe's data/layout without depending on `Plots` (and, transitively, GR,
# Qt6, and the X11/Cairo/HarfBuzz/FFMPEG stack it pulls in), keeping this test
# file's precompile cost in line with the rest of the suite. An end-to-end
# check that `Plots.plot(sol)` itself still renders is left to manual/CI-only
# smoke testing, not the fast unit-test loop.

@testset "Plot Recipes" begin
    grid = create_grid(2^9, 5e-12, 1550e-9)
    pulse = sech_pulse(grid, 100.0, 100e-15)
    medium = Medium(0.05, 0.0011, 0.0, [-21.5e-27], 1550e-9)
    sol = solve(
        pulse, SimParams(; medium=medium, z_saves=5, raman_model=nothing); progress=false
    )

    rd = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sol)
    @test rd isa AbstractVector
    subplots = [r.plotattributes[:subplot] for r in rd]
    @test length(unique(subplots)) == 4
    @test Set(subplots) == Set(1:4)
    # Subplots 1-2 are heatmaps (temporal/spectral evolution); 3-4 are two
    # overlaid line series each (initial vs final slice).
    seriestypes = [r.plotattributes[:seriestype] for r in rd]
    @test count(==(:heatmap), seriestypes) == 2
    @test count(==(:path), seriestypes) == 4
end

@testset "Recipe spectral wavelength alignment" begin
    grid = create_grid(128, 2e-12, 1550e-9)
    dw = 2π / (grid.N * grid.dt)
    tone = cis.(-7dw .* grid.t)
    pulse = Pulse(tone, ifft(tone), grid)
    medium = Medium(length=0.01, gamma=0.0, betas=[0.0], lambda0=grid.lambda0)
    sol = solve(pulse, SimParams(; medium, z_saves=2, raman_model=nothing,
        self_steepening=false); progress=false)

    rd = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sol)
    spectral = rd[end]
    x, y = spectral.args
    @test x[argmax(y)] ≈ 2π * c / (grid.omega0 + 7dw) * 1e9
end
