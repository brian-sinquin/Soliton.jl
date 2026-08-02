using Test
using GNLSE

@testset "Commercial Fiber Library and Glass Presets" begin
    @testset "Glass Refractive Index Presets" begin
        # Test FusedSilica Sellmeier
        fs = FusedSilica()
        @test fs isa SellmeierDispersion
        @test length(fs.B) == 3
        @test length(fs.C) == 3

        # Test SF6 and SF57
        @test SF6() isa SellmeierDispersion
        @test SF57() isa SellmeierDispersion

        # Test GeO2-Doped Silica
        geo2 = GeO2DopedSilica(5.0)
        @test geo2 isa SellmeierDispersion
        @test_throws ArgumentError GeO2DopedSilica(-1.0)
        @test_throws ArgumentError GeO2DopedSilica(20.0)
    end

    @testset "Commercial Fiber Catalog" begin
        # Verify FiberLibrary contents
        @test haskey(FiberLibrary, "Corning_SMF28")
        @test haskey(FiberLibrary, "NKT_NL_PM_750")
        @test haskey(FiberLibrary, "Thorlabs_PM780")
        @test haskey(FiberLibrary, "Thorlabs_PM1550")
        @test haskey(FiberLibrary, "NKT_LMA10")

        # Test commercial_fiber factory
        smf28 = commercial_fiber("Corning_SMF28", length=100.0)
        @test smf28 isa Medium
        @test smf28.length == 100.0
        @test smf28.lambda0 == 1550e-9
        @test smf28.gamma == 0.00127

        # Custom wavelength & loss overrides
        custom_smf = commercial_fiber(
            "Corning_SMF28", length=50.0, lambda0=1310e-9, loss=0.0004
        )
        @test custom_smf.length == 50.0
        @test custom_smf.lambda0 == 1310e-9
        @test custom_smf.loss == 0.0004

        @test_throws ArgumentError commercial_fiber("NonExistentFiber", length=1.0)
    end

    @testset "Propagation with Commercial Fiber" begin
        grid = create_grid(2^10, 10e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 100.0, 100e-15)
        medium = commercial_fiber("Corning_SMF28", length=10.0)
        params = SimParams(; medium=medium, z_saves=3, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 10.0
    end
end
