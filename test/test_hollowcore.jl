using Test
using GNLSE

@testset "Hollow-Core PCF (HC-PCF) & Molecular Gas Raman" begin
    @testset "Gas Refractive Index & Pressure Scaling" begin
        n_ar_1bar = gas_refractive_index(:Ar, 800e-9, 1.0)
        n_ar_5bar = gas_refractive_index(:Ar, 800e-9, 5.0)

        @test n_ar_1bar > 1.0
        @test n_ar_5bar > n_ar_1bar
        @test (n_ar_5bar - 1.0) ≈ 5.0 * (n_ar_1bar - 1.0) rtol=1e-4

        @test_throws ArgumentError gas_refractive_index(:UnknownGas, 800e-9, 1.0)
    end

    @testset "HollowCoreFiber Constructor & Dispersion" begin
        # 30 μm core diameter (radius 15 μm) filled with 5 bar Argon at 800 nm
        hcf = HollowCoreFiber(
            radius=15e-6, gas=:Ar, pressure=5.0, length=0.5, lambda0=800e-9
        )

        @test hcf isa Medium
        @test hcf.length == 0.5
        @test hcf.gamma > 0.0
        @test hcf.dispersion isa TaylorDispersion

        # Propagation test
        grid = create_grid(2^10, 10e-12, 800e-9)
        pulse = gaussian_pulse(grid, 5000.0, 50e-15)
        params = SimParams(; medium=hcf, z_saves=3, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.5
    end

    @testset "HollowCoreFiber with grid (TabulatedDispersion branch)" begin
        # Regression test: passing `grid` selects the exact TabulatedDispersion
        # branch, which previously errored because the `length` keyword argument
        # shadowed Base.length inside the constructor.
        grid = create_grid(2^10, 10e-12, 800e-9)
        hcf = HollowCoreFiber(
            radius=15e-6, gas=:Ar, pressure=3.0, length=0.5, lambda0=800e-9, grid=grid
        )

        @test hcf isa Medium
        @test hcf.dispersion isa TabulatedDispersion

        pulse = sech_pulse(grid, 5000.0, 50e-15)
        params = SimParams(; medium=hcf, z_saves=3, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.5
    end

    @testset "Molecular Gas Raman Models" begin
        rot = MolecularRamanGas(:H2_rotational)
        vib = MolecularRamanGas(:H2_vibrational)

        @test rot.fr == 0.12
        @test vib.fr == 0.08

        grid = create_grid(2^10, 10e-12, 800e-9)
        fr, RT = raman_response(grid, rot)
        @test fr == 0.12
        @test length(RT) == grid.N

        @test_throws ArgumentError MolecularRamanGas(:InvalidGas)
    end

    @testset "Capillary confinement loss (opt-in)" begin
        # Regression test: docs/src/physics.md documents the Marcatili-Schmeltzer
        # capillary confinement loss as part of the HC-PCF model, but it was
        # never implemented, silently giving a lossless fiber by default.
        grid = create_grid(2^10, 5e-12, 800e-9)

        hcf_default = HollowCoreFiber(
            radius=15e-6, gas=:Ar, pressure=5.0, length=0.5, lambda0=800e-9
        )
        @test loss_vector(grid, hcf_default) == zeros(grid.N)

        hcf_lossy = HollowCoreFiber(
            radius=15e-6,
            gas=:Ar,
            pressure=5.0,
            length=0.5,
            lambda0=800e-9,
            confinement_loss=true,
        )
        lv = loss_vector(grid, hcf_lossy)
        @test all(isfinite, lv)
        @test all(>(0), lv)  # confinement loss is strictly positive everywhere

        # Extra `loss` stacks additively on top of the confinement loss.
        hcf_extra = HollowCoreFiber(
            radius=15e-6,
            gas=:Ar,
            pressure=5.0,
            length=0.5,
            lambda0=800e-9,
            loss=2.0,
            confinement_loss=true,
        )
        lv_extra = loss_vector(grid, hcf_extra)
        @test all(lv_extra .> lv)
    end
end
