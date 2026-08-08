using Test
using Soliton

@testset "Second-Order (χ⁽²⁾) Coupled Solver — Phase 1" begin
    @testset "Type validation" begin
        grid = create_grid(2^10, 20e-12, 1550e-9)

        @test_throws ArgumentError SecondOrderMedium(
            -0.1, 1.0, TaylorDispersion([0.0]), TaylorDispersion([0.0]), 0.0, 0.0, 0.0, 0.0, 1550e-9
        )
        @test_throws ArgumentError SecondOrderMedium(
            0.01, 1.0, TaylorDispersion([0.0]), TaylorDispersion([0.0]), 0.0, -1.0, 0.0, 0.0, 1550e-9
        )
        @test_throws ArgumentError SecondOrderMedium(
            0.01, 1.0, TaylorDispersion([0.0]), TaylorDispersion([0.0]), 0.0, 0.0, -1.0, 0.0, 1550e-9
        )

        medium = SecondOrderMedium(;
            length=0.01, kappa=1.0, betas_fund=[0.0], betas_sh=[0.0], deltak0=0.0,
            lambda0=1550e-9,
        )
        @test medium.length == 0.01
        @test medium.poling_period == 0.0

        At_fund = sech_pulse(grid, 100.0, 100e-15).At
        At_sh = zeros(ComplexF64, grid.N)

        @test_throws ArgumentError SecondOrderPulse(zeros(ComplexF64, grid.N, 3), grid)
        @test_throws ArgumentError SecondOrderPulse(zeros(ComplexF64, grid.N - 1, 2), grid)

        qpulse = SecondOrderPulse(At_fund, At_sh, grid)
        @test size(qpulse.At) == (grid.N, 2)
    end

    @testset "Manley-Rowe power conservation (lossless, Δk≠0)" begin
        # Total power P_fund + P_sh must be exactly conserved for a lossless
        # medium regardless of phase mismatch -- energy conservation for a
        # 2-photons-in/1-photon-out process holds independent of Δk (Δk only
        # affects conversion efficiency and oscillation period, not net energy).
        grid = create_grid(2^10, 20e-12, 1550e-9)
        pulse = SecondOrderPulse(sech_pulse(grid, 50.0, 500e-15).At, zeros(ComplexF64, grid.N), grid)

        for deltak0 in (0.0, 500.0, -1500.0)
            medium = SecondOrderMedium(;
                length=0.005, kappa=2.0, betas_fund=[0.0], betas_sh=[0.0],
                deltak0=deltak0, lambda0=1550e-9,
            )
            params = SimParams(; medium=medium, z_saves=11, raman_model=nothing, solver=SSFM(1e-6))
            sol = solve(pulse, params; progress=false)

            E_fund = vec(sum(abs2.(sol.At[:, 1, :]); dims=1))
            E_sh = vec(sum(abs2.(sol.At[:, 2, :]); dims=1))
            E_total = E_fund .+ E_sh

            @test all(isapprox.(E_total, E_total[1]; rtol=1e-4))
        end
    end

    @testset "Undepleted-pump sinc² SHG scaling vs Δk" begin
        # In the undepleted-pump limit (small κ√P₁L), |A₂(L)|² ≈ κ²P₁₀L²sinc²(ΔkL/2)
        # (sinc(x) = sin(x)/x). Sweep Δk and compare directly to this closed form.
        grid = create_grid(2^10, 20e-12, 1550e-9)
        P10 = 1.0  # W -- weak enough to stay in the undepleted regime
        kappa = 0.5
        L = 0.005

        At_fund = fill(ComplexF64(sqrt(P10)), grid.N)
        At_sh = zeros(ComplexF64, grid.N)
        pulse = SecondOrderPulse(At_fund, At_sh, grid)

        for deltak0 in (0.0, 200.0, 500.0, 1000.0, 2000.0)
            medium = SecondOrderMedium(;
                length=L, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0],
                deltak0=deltak0, lambda0=1550e-9,
            )
            params = SimParams(; medium=medium, z_saves=2, raman_model=nothing, solver=SSFM(1e-6))
            sol = solve(pulse, params; progress=false)

            P2_numeric = abs2(sol.At[1, 2, end])

            x = deltak0 * L / 2
            sinc_val = x == 0 ? 1.0 : sin(x) / x
            P2_analytic = kappa^2 * P10^2 * L^2 * sinc_val^2

            @test isapprox(P2_numeric, P2_analytic; rtol=1e-3, atol=1e-8)
        end
    end

    @testset "Depleted-pump exact tanh² SHG at Δk=0" begin
        # At perfect phase matching, the lossless degenerate-SHG coupled equations
        # have the exact closed-form solution P_sh(z)/P_fund(0) = tanh²(κ√P_fund(0)·z),
        # valid all the way to full depletion (not just the undepleted-pump limit).
        grid = create_grid(2^10, 20e-12, 1550e-9)
        P10 = 100.0
        kappa = 15.0  # kappa*sqrt(P10)*L = 1.5 -> well into the depleted-pump regime
        L = 0.01

        At_fund = fill(ComplexF64(sqrt(P10)), grid.N)
        At_sh = zeros(ComplexF64, grid.N)
        pulse = SecondOrderPulse(At_fund, At_sh, grid)

        medium = SecondOrderMedium(;
            length=L, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0], deltak0=0.0,
            lambda0=1550e-9,
        )
        params = SimParams(; medium=medium, z_saves=11, raman_model=nothing, solver=SSFM(1e-6))
        sol = solve(pulse, params; progress=false)

        z = range(0.0, L; length=11)
        for (i, zi) in enumerate(z)
            eta_analytic = tanh(kappa * sqrt(P10) * zi)^2
            P2_analytic = eta_analytic * P10
            P2_numeric = abs2(sol.At[1, 2, i])
            @test isapprox(P2_numeric, P2_analytic; rtol=1e-3, atol=1e-6)
        end

        # Full depletion at large κ√P₁L: nearly all power converted to SH
        @test abs2(sol.At[1, 2, end]) / P10 > 0.75
    end

    @testset "kappa=0 disables coupling" begin
        grid = create_grid(2^10, 20e-12, 1550e-9)
        fund_pulse = sech_pulse(grid, 50.0, 500e-15)
        pulse = SecondOrderPulse(fund_pulse.At, zeros(ComplexF64, grid.N), grid)

        medium = SecondOrderMedium(;
            length=0.005, kappa=0.0, betas_fund=[-1000e-27], betas_sh=[-500e-27],
            deltak0=1234.0, lambda0=1550e-9,
        )
        params = SimParams(; medium=medium, z_saves=5, raman_model=nothing, solver=SSFM(1e-5))
        sol = solve(pulse, params; progress=false)

        # No second-harmonic generation without coupling
        @test all(iszero, sol.At[:, 2, :])
        # Fundamental only disperses (compare against scalar Medium propagation
        # with matching dispersion and zero Kerr)
        scalar_medium = Medium(0.005, 0.0, 0.0, [-1000e-27], 1550e-9)
        scalar_params = SimParams(;
            medium=scalar_medium, z_saves=5, raman_model=nothing, solver=SSFM(1e-5)
        )
        scalar_sol = solve(fund_pulse, scalar_params; progress=false)
        @test sol.At[:, 1, end] ≈ scalar_sol.At[:, end] rtol=1e-6
    end

    @testset "Quasi-phase-matching restores conversion at nonzero Δk" begin
        # Poling with period Λ ≈ 2π/Δk should recover strong conversion despite
        # a phase mismatch large enough to fully suppress unpoled SHG.
        grid = create_grid(2^10, 20e-12, 1550e-9)
        P10 = 4.0
        kappa = 1.0
        L = 0.01
        deltak0 = 2000.0  # 1/m -- large enough to strongly suppress unpoled SHG over L
        poling_period = 2π / deltak0

        At_fund = fill(ComplexF64(sqrt(P10)), grid.N)
        At_sh = zeros(ComplexF64, grid.N)
        pulse = SecondOrderPulse(At_fund, At_sh, grid)

        medium_unpoled = SecondOrderMedium(;
            length=L, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0], deltak0=deltak0,
            poling_period=0.0, lambda0=1550e-9,
        )
        medium_poled = SecondOrderMedium(;
            length=L, kappa=kappa, betas_fund=[0.0], betas_sh=[0.0], deltak0=deltak0,
            poling_period=poling_period, lambda0=1550e-9,
        )

        params_unpoled = SimParams(;
            medium=medium_unpoled, z_saves=2, raman_model=nothing, solver=SSFM(1e-6)
        )
        params_poled = SimParams(;
            medium=medium_poled, z_saves=2, raman_model=nothing, solver=SSFM(1e-6)
        )

        sol_unpoled = solve(pulse, params_unpoled; progress=false)
        sol_poled = solve(pulse, params_poled; progress=false)

        P2_unpoled = abs2(sol_unpoled.At[1, 2, end])
        P2_poled = abs2(sol_poled.At[1, 2, end])

        @test P2_poled > 10 * P2_unpoled
    end
end
