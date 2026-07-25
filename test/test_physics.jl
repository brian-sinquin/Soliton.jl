using Test
using JuGNLSE

# --- local measurement helpers -------------------------------------------

# FWHM of an intensity profile on a monotonic axis
function _row_fwhm(intensity, axis)
    half = 0.5 * maximum(intensity)
    idx = findall(>=(half), intensity)
    return axis[last(idx)] - axis[first(idx)]
end

# Intensity-weighted centroid of a field on a monotonic axis
_centroid(field, axis) = sum(axis .* abs2.(field)) / sum(abs2.(field))

# RMS spectral width [rad/s] from a (fftshifted) spectrum aligned with grid.V
function _rms_width(AW, V)
    P = abs2.(AW)
    m = sum(V .* P) / sum(P)
    return sqrt(sum((V .- m) .^ 2 .* P) / sum(P))
end

@testset "Physics" begin
    @testset "Linear dispersion broadens a Gaussian by √2 at L_D" begin
        FWHM = 200e-15
        b2 = -1.0e-26
        grid = create_grid(2^13, 20e-12, 835e-9)
        T0sq = FWHM^2 / (4 * log(2))
        LD = T0sq / abs(b2)

        medium = Medium(LD, 0.0, 0.0, [b2], 835e-9)   # γ = 0, propagate one L_D
        pulse = gaussian_pulse(grid, 1.0, FWHM)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=2,
            raman_model=nothing, self_steepening=false); progress=false)

        out = _row_fwhm(abs2.(sol.At[:, end]), grid.t)
        @test out ≈ FWHM * sqrt(2) rtol = 0.02
    end

    @testset "Fundamental soliton keeps its peak power" begin
        b2 = -1.0e-26
        T0 = 100e-15
        gam = 0.1
        P0 = abs(b2) / (gam * T0^2)            # N = 1 condition
        zsol = (π / 2) * T0^2 / abs(b2)

        grid = create_grid(2^12, 20e-12, 835e-9)
        medium = Medium(2 * zsol, gam, 0.0, [b2], 835e-9)
        pulse = sech_pulse(grid, P0, 2 * log(1 + sqrt(2)) * T0)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=20,
            raman_model=nothing, self_steepening=false); progress=false)

        peaks = [maximum(abs2.(sol.At[:, i])) for i in 1:size(sol.At, 2)]
        @test all(p -> isapprox(p, P0; rtol=0.03), peaks)
    end

    @testset "Energy is conserved without loss" begin
        grid = create_grid(2^12, 20e-12, 835e-9)
        medium = Medium(0.3, 0.1, 0.0, [-1.0e-26], 835e-9)
        pulse = sech_pulse(grid, 50.0, 100e-15)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=10,
            raman_model=BlowWood(), self_steepening=false); progress=false)

        e0 = sum(abs2, sol.At[:, 1])
        e1 = sum(abs2, sol.At[:, end])
        @test e1 / e0 ≈ 1.0 rtol = 1e-4
    end

    @testset "Loss decays energy as exp(-α z)" begin
        loss_dB = 5.0
        L = 0.1
        grid = create_grid(2^11, 20e-12, 835e-9)
        medium = Medium(L, 0.0, loss_dB, [-1.0e-26], 835e-9)   # γ = 0
        pulse = gaussian_pulse(grid, 10.0, 200e-15)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=2,
            raman_model=nothing, self_steepening=false); progress=false)

        alpha = log(10.0^(loss_dB / 10.0))
        e0 = sum(abs2, sol.At[:, 1])
        e1 = sum(abs2, sol.At[:, end])
        @test e1 / e0 ≈ exp(-alpha * L) rtol = 1e-3
    end

    @testset "Raman self-frequency shift (red-shift)" begin
        b2 = -1.0e-26
        T0 = 50e-15
        gam = 0.1
        P0 = abs(b2) / (gam * T0^2)

        grid = create_grid(2^13, 20e-12, 835e-9)
        medium = Medium(0.4, gam, 0.0, [b2], 835e-9)
        pulse = sech_pulse(grid, P0, 2 * log(1 + sqrt(2)) * T0)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=6,
            raman_model=BlowWood(), self_steepening=false); progress=false)

        # Spectrum shifts to lower absolute optical frequency (longer wavelength)
        w0 = _centroid(sol.AW[:, 1], grid.W)
        w1 = _centroid(sol.AW[:, end], grid.W)
        @test w1 < w0

        # A red-shifted soliton slows down (anomalous GVD) ⇒ arrives later
        t0 = _centroid(sol.At[:, 1], grid.t)
        t1 = _centroid(sol.At[:, end], grid.t)
        @test t1 > t0
    end

    @testset "Self-phase modulation broadens the spectrum" begin
        grid = create_grid(2^12, 20e-12, 835e-9)
        medium = Medium(0.5, 1.0, 0.0, Float64[], 835e-9)   # no dispersion
        pulse = sech_pulse(grid, 100.0, 100e-15)
        sol = solve(pulse, SimParams(; medium=medium, z_saves=4,
            raman_model=nothing, self_steepening=false); progress=false)

        w0 = _rms_width(sol.AW[:, 1], grid.V)
        w1 = _rms_width(sol.AW[:, end], grid.V)
        @test w1 > w0
    end

    @testset "Tapered fiber varying gamma matches SPM analytical phase" begin
        grid = create_grid(2^10, 10e-12, 835e-9)
        # Taper parameters
        gamma0 = 50.0
        alpha_taper = 8.0
        L = 0.15
        
        # z-dependent gamma
        g(z) = gamma0 * exp(-alpha_taper * z)
        
        # No dispersion, no loss, no self-steepening, no Raman
        medium = Medium(L, g, 0.0, Float64[], 835e-9)
        pulse = gaussian_pulse(grid, 50.0, 200e-15) # Pmax = 50 W
        
        # Solve GNLSE
        sol = solve(pulse, SimParams(; medium=medium, z_saves=2,
            raman_model=nothing, self_steepening=false, rtol=1e-8, atol=1e-10); progress=false)
            
        # Analytical phase shift: ϕ = |A(0,t)|² * ∫₀ˡ γ(z) dz
        # ∫₀ˡ γ(z) dz = gamma0 * (1 - exp(-alpha_taper * L)) / alpha_taper
        gamma_integrated = gamma0 * (1.0 - exp(-alpha_taper * L)) / alpha_taper
        
        phase_analytical = abs2.(sol.At[:, 1]) .* gamma_integrated
        significant_indices = findall(x -> x > 1.0, abs2.(sol.At[:, 1]))

        # Compare complex phasors directly to avoid phase wrapping issues
        phasor_numerical = sol.At[significant_indices, end] ./ sol.At[significant_indices, 1]
        phasor_analytical = exp.(1im .* phase_analytical[significant_indices])
        
        @test phasor_numerical ≈ phasor_analytical rtol=1e-4
    end

    @testset "Free-carrier refraction (FCR) blue-shifts spectrum" begin
        grid = create_grid(2^11, 20e-12, 1550e-9)
        soi = SemiconductorMedium(
            length = 0.005,
            gamma = 50.0,
            alpha2 = 1e-12,
            Aeff = 0.1e-12,
            sigma_fca = 0.0, # disable FCA loss to isolate phase shift
            k_fcr = 5.3e-27,  # active FCR plasma effect
            tau_c = 1e-9,
            betas = Float64[],
            lambda0 = 1550e-9
        )
        pulse = gaussian_pulse(grid, 100.0, 1e-12)
        sol = solve(pulse, SimParams(; medium=soi, raman_model=nothing, z_saves=2); progress=false)

        w0 = _centroid(sol.AW[:, 1], grid.W)
        w1 = _centroid(sol.AW[:, end], grid.W)
        # Free carrier accumulation decreases refractive index ⇒ blue-shift (higher ω)
        @test w1 > w0
    end

    @testset "EDFA gain saturation energy limit" begin
        grid = create_grid(2^11, 20e-12, 1550e-9)
        # 1 m amplifier with g0 = 10 dB/m (2.3026 Np/m), Esat = 1 μJ
        edfa = AmplifyingMedium(
            length = 1.0,
            gamma = 0.0,
            g0_db = 10.0,
            Esat = 1e-6,
            noise_figure_db = 0.0,
            betas = Float64[],
            lambda0 = 1550e-9
        )
        # Intense pulse E_in >> Esat -> saturates gain
        pulse = gaussian_pulse(grid, 50.0e3, 1e-12) # 50 nJ input
        sol = solve(pulse, SimParams(; medium=edfa, raman_model=nothing, z_saves=2); progress=false)

        E_in = pulse_energy(pulse)
        E_out = pulse_energy(Pulse(sol))
        
        # Gain ratio E_out / E_in must not exceed small-signal gain 10^(10/10) = 10.0
        @test E_out / E_in <= 10.0 + 1e-12
        @test E_out > E_in
    end
end
