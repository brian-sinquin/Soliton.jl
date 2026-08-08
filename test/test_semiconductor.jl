using Test
using Soliton

@testset "Semiconductor Waveguides (SOI, TPA, Free-Carrier Dynamics)" begin
    grid = create_grid(2^10, 10e-12, 1550e-9)
    pulse = gaussian_pulse(grid, 50.0, 1e-12) # 50 W peak power pulse

    @testset "SemiconductorMedium Constructor" begin
        soi = SemiconductorMedium(
            length=0.01, # 1 cm silicon nanowire
            gamma=300.0, # 300 /W/m Kerr nonlinearity
            alpha2=5e-12, # 5 cm/GW TPA coefficient
            Aeff=0.1e-12, # 0.1 μm² effective area
            tau_c=1e-9,  # 1 ns carrier lifetime
            betas=[-1000e-27], # anomalous dispersion
            lambda0=1550e-9,
        )

        @test soi.length == 0.01
        @test soi.alpha2 == 5e-12
        @test soi.Aeff == 0.1e-12
        @test soi.sigma_fca == 1.45e-21
        @test soi.k_fcr == 5.3e-27
        @test soi.alpha3 == 0.0
    end

    @testset "TPA Loss & Free Carrier Blue Shift" begin
        soi = SemiconductorMedium(
            length=0.005,
            gamma=100.0,
            alpha2=5e-12,
            Aeff=0.1e-12,
            tau_c=1e-9,
            betas=[-1000e-27],
            lambda0=1550e-9,
        )
        params = SimParams(; medium=soi, z_saves=5, raman_model=nothing)

        sol = solve(pulse, params; progress=false)
        @test sol isa Solution
        @test sol.Z[end] == 0.005

        # TPA must reduce pulse energy below lossless Kerr propagation
        E_in = pulse_energy(pulse)
        E_out = pulse_energy(Pulse(sol))
        @test E_out < E_in
    end

    @testset "3PA Loss & Free Carrier Generation" begin
        # alpha3 = 0 must reproduce pure-TPA propagation exactly (SiGe mid-IR default)
        common_tpa = (alpha2=5e-12, Aeff=0.1e-12, tau_c=1e-9, betas=[-1000e-27], lambda0=1550e-9)
        soi_tpa_only = SemiconductorMedium(; length=0.005, gamma=100.0, common_tpa...)
        soi_zero_3pa = SemiconductorMedium(; length=0.005, gamma=100.0, alpha3=0.0, common_tpa...)
        params_tpa_only = SimParams(; medium=soi_tpa_only, z_saves=5, raman_model=nothing)
        params_zero_3pa = SimParams(; medium=soi_zero_3pa, z_saves=5, raman_model=nothing)
        sol_tpa_only = solve(pulse, params_tpa_only; progress=false)
        sol_zero_3pa = solve(pulse, params_zero_3pa; progress=false)
        @test sol_zero_3pa.At ≈ sol_tpa_only.At rtol=1e-12

        # Enabling 3PA must add extra nonlinear loss beyond TPA alone.
        # alpha3 = 2e-27 m^3/W^2 ~ 2e-3 cm^3/GW^2, the reported mid-IR Si value.
        soi_3pa = SemiconductorMedium(; length=0.005, gamma=100.0, alpha3=2e-27, common_tpa...)
        params_3pa = SimParams(; medium=soi_3pa, z_saves=5, raman_model=nothing)
        sol_3pa = solve(pulse, params_3pa; progress=false)

        E_tpa_only = pulse_energy(Pulse(sol_tpa_only))
        E_3pa = pulse_energy(Pulse(sol_3pa))
        @test E_3pa < E_tpa_only
    end

    @testset "Analytic Ground-Truth Cross-Validation (TPA+3PA scalar ODE)" begin
        # With gamma=0, no dispersion, and FCA/FCR disabled, every time sample
        # propagates independently under the decoupled scalar ODE
        #   dP/dz = -c2*P^2 - c3*P^3,  c2 = alpha2/Aeff, c3 = alpha3/Aeff^2
        # (no dispersion => no coupling between time bins; gamma=0 => no phase
        # feedback into the loss). We integrate this ODE with a dedicated fine
        # RK4 — independent of Soliton's own ERK4IP/SSFM propagator — as a
        # genuine ground truth, since no other GNLSE package (e.g. gnlse-python)
        # implements TPA/3PA to cross-check against.
        function _rk4_ground_truth(P0::AbstractVector, c2::Real, c3::Real, L::Real; nsteps::Int=20_000)
            dz = L / nsteps
            P = copy(P0)
            f(P) = @. -c2 * P^2 - c3 * P^3
            for _ in 1:nsteps
                k1 = f(P)
                k2 = f(P .+ (0.5dz) .* k1)
                k3 = f(P .+ (0.5dz) .* k2)
                k4 = f(P .+ dz .* k3)
                @. P += (dz / 6) * (k1 + 2k2 + 2k3 + k4)
            end
            return P
        end

        gtgrid = create_grid(2^9, 20e-12, 1550e-9)
        gtpulse = sech_pulse(gtgrid, 40.0, 2.0e-12)
        Aeff = 0.1e-12
        L = 0.005
        P0 = abs2.(gtpulse.At)

        for (alpha2, alpha3) in ((5e-12, 0.0), (0.0, 2e-27), (5e-12, 2e-27))
            medium = SemiconductorMedium(;
                length=L, gamma=0.0, alpha2=alpha2, alpha3=alpha3, Aeff=Aeff,
                sigma_fca=0.0, k_fcr=0.0, tau_c=1.0e-9,
                betas=Float64[], lambda0=1550e-9,
            )
            params = SimParams(; medium=medium, raman_model=nothing, z_saves=2, self_steepening=false)
            sol = solve(gtpulse, params; progress=false)

            P_numeric = abs2.(sol.At[:, end])
            P_ground_truth = _rk4_ground_truth(P0, alpha2 / Aeff, alpha3 / Aeff^2, L)

            @test P_numeric ≈ P_ground_truth rtol=1e-5
        end
    end

    @testset "Self-steepening applies to Kerr/TPA/3PA but not FCA/FCR" begin
        # Shock weighting (gamma_W/omega0) is only physically appropriate for the
        # bound-electron polarization response (Kerr/TPA/3PA); free-carrier plasma
        # effects (FCA/FCR) are slowly-varying and must not pick it up. Isolate FCA
        # (gamma=0, k_fcr=0, alpha3=0) and verify the output is identical whether
        # self-steepening is on or off, since only FCA is active in the loss.
        common = (alpha2=5e-12, Aeff=0.1e-12, tau_c=1e-9, betas=[-1000e-27], lambda0=1550e-9)
        soi_fca_only = SemiconductorMedium(;
            length=0.005, gamma=0.0, k_fcr=0.0, sigma_fca=1.45e-21, common...
        )
        sol_shock_off = solve(
            pulse,
            SimParams(; medium=soi_fca_only, z_saves=5, raman_model=nothing, self_steepening=false);
            progress=false,
        )
        sol_shock_on = solve(
            pulse,
            SimParams(; medium=soi_fca_only, z_saves=5, raman_model=nothing, self_steepening=true);
            progress=false,
        )
        # FCA-driven loss/N_c generation involves only |A|^2 and |A|^4 -- TPA still
        # picks up shock weighting, so this isn't an exact match, but the *linear*
        # FCA/FCR contribution itself must not scale with the shock factor. Confirm
        # by comparing against a Kerr/TPA-only medium (k_fcr=0, sigma_fca=0): the
        # difference between FCA-on and FCA-off runs must be shock-independent.
        soi_no_fca = SemiconductorMedium(;
            length=0.005, gamma=0.0, k_fcr=0.0, sigma_fca=0.0, common...
        )
        sol_nofca_off = solve(
            pulse,
            SimParams(; medium=soi_no_fca, z_saves=5, raman_model=nothing, self_steepening=false);
            progress=false,
        )
        sol_nofca_on = solve(
            pulse,
            SimParams(; medium=soi_no_fca, z_saves=5, raman_model=nothing, self_steepening=true);
            progress=false,
        )
        # TPA-only path may differ slightly between shock on/off (self-steepening
        # perturbs TPA too); but the *FCA-attributable* extra loss --
        # |A_fca_off|^2 - |A_fca_on|^2 relative to the matching no-FCA run -- must
        # be the same regardless of the shock switch, since FCA never sees gamma_W.
        fca_effect_shock_off = abs2.(sol_shock_off.At[:, end]) .- abs2.(sol_nofca_off.At[:, end])
        fca_effect_shock_on = abs2.(sol_shock_on.At[:, end]) .- abs2.(sol_nofca_on.At[:, end])
        @test fca_effect_shock_off ≈ fca_effect_shock_on rtol=1e-4
    end

    @testset "NonlinearityModel gamma types" begin
        # Regression test: SemiconductorMedium's build_physics_model/_semiconductor_spm
        # used to only handle Number/Function gamma inline, hitting a MethodError for
        # ConstantNonlinearity etc. Now dispatched via `_semiconductor_gamma_at_z`.
        common = (alpha2=5e-12, Aeff=0.1e-12, tau_c=1e-9, betas=[-1000e-27], lambda0=1550e-9)

        soi_num = SemiconductorMedium(; length=0.005, gamma=100.0, common...)
        soi_const = SemiconductorMedium(; length=0.005, gamma=ConstantNonlinearity(100.0), common...)
        params_num = SimParams(; medium=soi_num, z_saves=5, raman_model=nothing)
        params_const = SimParams(; medium=soi_const, z_saves=5, raman_model=nothing)

        sol_num = solve(pulse, params_num; progress=false)
        sol_const = solve(pulse, params_const; progress=false)
        @test sol_const isa Solution
        # ConstantNonlinearity(100.0) must behave identically to plain gamma=100.0
        @test sol_const.At ≈ sol_num.At rtol=1e-8

        soi_freq = SemiconductorMedium(; length=0.005, gamma=FrequencyDependentNonlinearity(w -> 100.0), common...)
        params_freq = SimParams(; medium=soi_freq, z_saves=5, raman_model=nothing)
        @test_throws ArgumentError solve(pulse, params_freq; progress=false)
    end
end
