using Test
using GNLSE

@testset "Vectorial/Birefringent Coupled Solver" begin
    @testset "Type validation" begin
        grid = create_grid(2^10, 10e-12, 835e-9)

        # Medium constructors
        dis_x = TaylorDispersion([-1.0e-26])
        dis_y = TaylorDispersion([-1.0e-26])
        @test_throws ArgumentError BirefringentMedium(
            -0.1, 0.11, 0.0, dis_x, dis_y, 0.0, 835e-9
        )
        @test_throws ArgumentError BirefringentMedium(
            0.1, -0.11, 0.0, dis_x, dis_y, 0.0, 835e-9
        )

        medium = BirefringentMedium(0.02, 0.11, 0.0, dis_x, dis_y, 0.0, 835e-9)
        @test medium.length == 0.02

        # Pulse constructors
        At_x = sech_pulse(grid, 100.0, 100e-15).At
        At_y = zeros(ComplexF64, grid.N)

        @test_throws ArgumentError VectorialPulse(zeros(ComplexF64, grid.N, 3), grid)
        @test_throws ArgumentError VectorialPulse(zeros(ComplexF64, grid.N-1, 2), grid)

        vpulse = VectorialPulse(At_x, At_y, grid)
        @test size(vpulse.At) == (grid.N, 2)
    end

    @testset "Comparison with scalar solver" begin
        # If we launch all energy into the x-axis with no coupling,
        # it should behave exactly like the scalar solver.
        grid = create_grid(2^10, 10e-12, 835e-9)

        # Standard scalar medium
        medium_scalar = Medium(0.01, 0.11, 0.0, [-1.0e-26], 835e-9)
        pulse_scalar = sech_pulse(grid, 500.0, 100e-15)
        params_scalar = SimParams(;
            medium=medium_scalar, z_saves=5, solver=SSFM(1e-5), raman_model=nothing
        )
        sol_scalar = solve(pulse_scalar, params_scalar; progress=false)

        # Vectorial medium with identical properties and deltabeta0 = 0
        dis_x = TaylorDispersion([-1.0e-26])
        dis_y = TaylorDispersion([-1.0e-26])
        medium_vec = BirefringentMedium(0.01, 0.11, 0.0, dis_x, dis_y, 0.0, 835e-9)

        # Load scalar pulse into X, zero in Y
        pulse_vec = VectorialPulse(pulse_scalar.At, zeros(ComplexF64, grid.N), grid)
        params_vec = SimParams(;
            medium=medium_vec, z_saves=5, solver=SSFM(1e-5), raman_model=nothing
        )
        sol_vec = solve(pulse_vec, params_vec; progress=false)

        # Check shapes
        @test size(sol_vec.At) == (grid.N, 2, 5)

        # Output in X should match scalar
        @test sol_vec.At[:, 1, end] ≈ sol_scalar.At[:, end]
        # Output in Y should remain exactly zero
        @test all(abs.(sol_vec.At[:, 2, end]) .< 1e-15)
    end

    @testset "Birefringent Walk-off" begin
        grid = create_grid(2^10, 15e-12, 835e-9)

        # Introduce a group velocity mismatch (walk-off) by giving different beta1 values:
        # beta1_x = 0, beta1_y = 100 fs/mm = 1e-10 s/m
        # Walk-off over 1 cm (0.01 m) should shift the Y pulse in time by:
        # delta_t = beta1_y * L = 1e-10 * 0.01 = 1 ps (1e-12 s)
        beta1_x = 0.0
        beta1_y = 1e-10

        dis_x = TaylorDispersion([0.0], beta1_x)
        dis_y = TaylorDispersion([0.0], beta1_y)

        medium = BirefringentMedium(0.01, 0.0, 0.0, dis_x, dis_y, 0.0, 835e-9)

        # Input identical pulses on both axes, centered in time
        pulse_x = gaussian_pulse(grid, 1.0, 100e-15)
        pulse_y = gaussian_pulse(grid, 1.0, 100e-15)
        pulse_vec = VectorialPulse(pulse_x.At, pulse_y.At, grid)

        params = SimParams(;
            medium=medium, z_saves=2, solver=SSFM(5e-5), raman_model=nothing
        )
        sol = solve(pulse_vec, params; progress=false)

        # Measure peak positions at output (save index 2)
        _, idx_x = findmax(abs2, sol.At[:, 1, end])
        _, idx_y = findmax(abs2, sol.At[:, 2, end])

        # Verify the time difference matches 1 ps
        t_diff = grid.t[idx_y] - grid.t[idx_x]
        @test t_diff ≈ 1.0e-12 rtol=0.02
    end

    @testset "Vectorial Analysis and Multi-stage Cascading" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        px = gaussian_pulse(grid, 100.0, 100e-15)
        py = gaussian_pulse(grid, 50.0, 100e-15)
        vpulse = VectorialPulse(px.At, py.At, grid)

        # pulse_energy and peak_power for VectorialPulse
        E_v = pulse_energy(vpulse)
        P_v = peak_power(vpulse)
        @test E_v ≈ pulse_energy(px) + pulse_energy(py)
        @test P_v ≈ peak_power(px) + peak_power(py)

        # Lumped elements on VectorialPulse
        amp = Amplifier(6.0) # ~2x amplitude boost (+6 dB)
        vpulse_amp = apply(vpulse, amp)
        @test peak_power(vpulse_amp) ≈ P_v * (10.0^(6.0/10.0))

        # Multi-stage cascaded propagation with VectorialPulse & LumpedElements
        dis_x = TaylorDispersion([-1.0e-26])
        dis_y = TaylorDispersion([-1.0e-26])
        medium = BirefringentMedium(0.01, 0.11, 0.0, dis_x, dis_y, 0.0, 835e-9)
        params = SimParams(;
            medium=medium, z_saves=3, solver=SSFM(1e-5), raman_model=nothing
        )

        stages = [params, amp, PMDElement(0.1e-12)]
        results = solve(vpulse, stages; progress=false)
        @test length(results) == 3
        @test results[1] isa VectorialSolution
        @test results[2] isa VectorialPulse
        @test results[3] isa VectorialPulse

        # Callable piping syntax
        vpulse_piped = vpulse |> Amplifier(3.0) |> params |> VectorialPulse
        @test vpulse_piped isa VectorialPulse
    end
end
