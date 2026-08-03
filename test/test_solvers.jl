using Test
using Soliton

@testset "Solvers" begin
    @testset "Solution structure" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)
        params = SimParams(;
            medium=medium, z_saves=8, raman_model=nothing, self_steepening=false
        )

        sol = solve(pulse, params; progress=false)

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
                params = SimParams(;
                    medium=medium, z_saves=4, raman_model=raman, self_steepening=shock
                )
                sol = solve(pulse, params; progress=false)
                @test all(isfinite, sol.At)
                @test all(isfinite, sol.AW)
            end
        end
    end

    @testset "Adaptive tolerance" begin
        grid = create_grid(2^11, 12e-12, 835e-9)
        medium = Medium(0.05, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 200.0, 80e-15)

        loose = solve(
            pulse,
            SimParams(;
                medium=medium, z_saves=4, raman_model=nothing, rtol=1e-3, atol=1e-4
            );
            progress=false,
        )
        tight = solve(
            pulse,
            SimParams(;
                medium=medium, z_saves=4, raman_model=nothing, rtol=1e-7, atol=1e-9
            );
            progress=false,
        )

        # Both integrations should agree closely on the final field
        rel = sum(abs2, loose.At[:, end] .- tight.At[:, end]) / sum(abs2, tight.At[:, end])
        @test rel < 1e-3
    end

    @testset "Tabulated dispersion matches Taylor" begin
        grid = create_grid(2^11, 12e-12, 835e-9)
        b2 = -1.0e-26
        pulse = sech_pulse(grid, 150.0, 80e-15)

        taylor = Medium(0.05, 0.11, 0.0, [b2], 835e-9)
        # A tabulated curve sampled from the same β₂ parabola
        tab_disp = TabulatedDispersion(
            grid.V, propagation_constant(grid.V, TaylorDispersion([b2]))
        )
        tabulated = Medium(0.05, 0.11, 0.0, tab_disp, 835e-9)

        st = solve(
            pulse,
            SimParams(;
                medium=taylor, z_saves=3, raman_model=BlowWood(), self_steepening=true
            );
            progress=false,
        )
        sb = solve(
            pulse,
            SimParams(;
                medium=tabulated, z_saves=3, raman_model=BlowWood(), self_steepening=true
            );
            progress=false,
        )

        rel = sum(abs2, st.At[:, end] .- sb.At[:, end]) / sum(abs2, st.At[:, end])
        @test rel < 1e-6
    end

    @testset "Solver abstraction (GNLSESolver / ERK4IP)" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        # Test constructor validation
        @test_throws ArgumentError ERK4IP(; rtol=-1e-6)
        @test_throws ArgumentError ERK4IP(; atol=-1e-8)

        # Explicitly construct ERK4IP solver
        solver = ERK4IP(; rtol=1e-5, atol=1e-7, dz_init=1e-5)
        @test solver.rtol == 1e-5
        @test solver.atol == 1e-7
        @test solver.dz_init == 1e-5

        # Create SimParams with explicit solver
        params = SimParams(; medium=medium, z_saves=4, solver=solver)
        @test params.solver isa ERK4IP
        @test params.solver === solver

        # Run solve
        sol = solve(pulse, params; progress=false)
        @test length(sol.Z) == 4
        @test all(isfinite, sol.At)

        # Try to specify both solver and rtol (should throw)
        @test_throws ArgumentError SimParams(; medium=medium, solver=solver, rtol=1e-6)
    end

    @testset "Symmetric Split-Step Fourier Method (SSFM)" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        # Test constructor validation
        @test_throws ArgumentError SSFM(0.0)
        @test_throws ArgumentError SSFM(-1e-5)

        # Run SSFM
        solver_ssfm = SSFM(1e-5)
        params_ssfm = SimParams(; medium=medium, z_saves=5, solver=solver_ssfm)
        sol_ssfm = solve(pulse, params_ssfm; progress=false)

        @test length(sol_ssfm.Z) == 5
        @test all(isfinite, sol_ssfm.At)

        # Compare with ERK4IP
        solver_erk = ERK4IP(; rtol=1e-6, atol=1e-8)
        params_erk = SimParams(; medium=medium, z_saves=5, solver=solver_erk)
        sol_erk = solve(pulse, params_erk; progress=false)

        # Verify that both solvers agree closely on the final field profile
        rel_diff =
            sum(abs2, sol_ssfm.At[:, end] .- sol_erk.At[:, end]) /
            sum(abs2, sol_erk.At[:, end])
        @test rel_diff < 1e-4
    end

    @testset "Phase-Controlled Adaptive SSFM (AdaptiveSSFM)" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        medium = Medium(0.02, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        @test_throws ArgumentError AdaptiveSSFM(; phi_max=-1e-3)

        solver_ad = AdaptiveSSFM(; phi_max=1e-3)
        params_ad = SimParams(; medium=medium, z_saves=5, solver=solver_ad)
        sol_ad = solve(pulse, params_ad; progress=false)

        @test length(sol_ad.Z) == 5
        @test all(isfinite, sol_ad.At)

        solver_erk = ERK4IP(; rtol=1e-6, atol=1e-8)
        params_erk = SimParams(; medium=medium, z_saves=5, solver=solver_erk)
        sol_erk = solve(pulse, params_erk; progress=false)

        rel_diff =
            sum(abs2, sol_ad.At[:, end] .- sol_erk.At[:, end]) /
            sum(abs2, sol_erk.At[:, end])
        @test rel_diff < 1e-3
    end

    @testset "Spectrogram & FROG Diagnostics" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        pulse = gaussian_pulse(grid, 100.0, 100e-15)

        t_spec, v_spec, S = spectrogram(pulse; n_delay=50)
        @test length(t_spec) == 50
        @test size(S) == (grid.N, 50)
        @test all(isfinite, S)
        @test maximum(S) > 0

        t_frog, v_frog, I_frog = shg_frog_trace(pulse; n_delay=50)
        @test length(t_frog) == 50
        @test size(I_frog) == (grid.N, 50)
        @test all(isfinite, I_frog)
        @test maximum(I_frog) > 0
    end

    @testset "ERK4IP Operator Caching" begin
        grid = create_grid(2^10, 10e-12, 1550e-9)
        medium = Medium(0.05, 0.0011, 0.0, [-21.5e-27], 1550e-9)
        pulse = sech_pulse(grid, 100.0, 100e-15)

        sol = solve(
            pulse,
            SimParams(; medium=medium, solver=ERK4IP(), raman_model=nothing);
            progress=false,
        )
        @test sol.Z[end] == 0.05
    end

    @testset "ERK4IP diverges cleanly instead of hanging" begin
        # Regression test: previously, a diverging field drove local_error to
        # NaN, which always fails `local_error <= 1.0` in Julia, so the step
        # was rejected forever with no floor on dz — the solver hung forever
        # instead of erroring. An absurdly large gamma/power combination
        # should now throw a clear ErrorException within a bounded number of
        # rejected steps instead of looping indefinitely.
        grid = create_grid(2^8, 2e-12, 1550e-9)
        pulse = gaussian_pulse(grid, 1e12, 20e-15)
        medium = Medium(1.0, 100.0, 0.0, [-20e-27], 1550e-9)
        params = SimParams(;
            medium=medium, z_saves=10, raman_model=nothing, self_steepening=false
        )
        @test_throws ErrorException solve(pulse, params; progress=false)
    end

    @testset "z_saves must be >= 2" begin
        medium = Medium(0.05, 0.0011, 0.0, [-21.5e-27], 1550e-9)
        @test_throws ArgumentError SimParams(; medium=medium, z_saves=1)
        @test_throws ArgumentError SimParams(; medium=medium, z_saves=0)
    end
end
