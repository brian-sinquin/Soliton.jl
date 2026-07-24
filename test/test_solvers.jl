using Test
using JuGNLSE

@testset "Solvers" begin
    @testset "Solution structure" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)
        params = SimParams(; medium=medium, z_saves=8,
            raman_model=nothing, self_steepening=false)

        sol = solve(pulse, params, ERK4IP(); progress=false)

        @test length(sol.Z) == 8
        @test sol.Z[1] == 0.0
        @test sol.Z[end] ≈ medium.length
        @test issorted(sol.Z)
        @test size(sol.At) == (grid.N, 8)
        @test size(sol.AW) == (grid.N, 8)
        @test sol.t == grid.t
        @test sol.omega0 ≈ grid.omega0

        # First saved slice matches the input pulse
        @test sol.At[:, 1] ≈ pulse.At
    end

    @testset "Physics-flag combinations run" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        for raman in (nothing, BlowWood(), LinAgrawal(), Hollenbeck())
            for shock in (false, true)
                params = SimParams(; medium=medium, z_saves=4,
                    raman_model=raman, self_steepening=shock)
                sol = solve(pulse, params, ERK4IP(); progress=false)
                @test all(isfinite, sol.At)
                @test all(isfinite, sol.AW)
            end
        end
    end

    @testset "Adaptive tolerance" begin
        grid = create_grid(2^11, 12e-12, 835e-9)
        medium = Medium(0.05, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 200.0, 80e-15)

        loose = solve(pulse, SimParams(; medium=medium, z_saves=4,
            raman_model=nothing, rtol=1e-3, atol=1e-4), ERK4IP(); progress=false)
        tight = solve(pulse, SimParams(; medium=medium, z_saves=4,
            raman_model=nothing, rtol=1e-7, atol=1e-9), ERK4IP(); progress=false)

        # Both integrations should agree closely on the final field
        rel = sum(abs2, loose.At[:, end] .- tight.At[:, end]) /
              sum(abs2, tight.At[:, end])
        @test rel < 1e-3
    end

    @testset "Tabulated dispersion matches Taylor" begin
        grid = create_grid(2^11, 12e-12, 835e-9)
        b2 = -1.0e-26
        pulse = sech_pulse(grid, 150.0, 80e-15)

        taylor = Medium(0.05, 0.11, 0.0, [b2], 835e-9)
        # A tabulated curve sampled from the same β₂ parabola
        tab_disp = TabulatedDispersion(grid.V, propagation_constant(grid.V,
            TaylorDispersion([b2])))
        tabulated = Medium(0.05, 0.11, 0.0, tab_disp, 835e-9)

        st = solve(pulse, SimParams(; medium=taylor, z_saves=3,
            raman_model=BlowWood(), self_steepening=true), ERK4IP(); progress=false)
        sb = solve(pulse, SimParams(; medium=tabulated, z_saves=3,
            raman_model=BlowWood(), self_steepening=true), ERK4IP(); progress=false)

        rel = sum(abs2, st.At[:, end] .- sb.At[:, end]) /
              sum(abs2, st.At[:, end])
        @test rel < 1e-6
    end

    @testset "Solver interface and RK4" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)
        problem = GNLSEProblem(pulse, medium, SimParams(; raman_model=nothing, self_steepening=false, z_saves=8))

        # Test with default ERK4IP solver
        sol_erk4ip = solve(problem, ERK4IP(); progress=false)
        @test all(isfinite, sol_erk4ip.At)

        # Test with RK4 solver
        sol_rk4 = solve(problem, RK4(dz=1e-5); progress=false) # Smaller dz for RK4
        @test all(isfinite, sol_rk4.At)

        # Compare results (ERK4IP should be more accurate for the same 'effective' step size)
        # This is a qualitative test, as RK4 needs a much smaller step size to match ERK4IP
        rel = sum(abs2, sol_erk4ip.At[:, end] .- sol_rk4.At[:, end]) /
              sum(abs2, sol_erk4ip.At[:, end])
        @test rel < 1e-2 # Expect some difference, but should be reasonably close
    end
end
