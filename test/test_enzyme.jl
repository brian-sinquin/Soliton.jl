using Test
using Soliton
using Enzyme
using FFTW: plan_fft, plan_ifft
using LinearAlgebra: mul!

"""
Complex-valued finite-difference gradient of a real-valued `f(x)` at `x0[i]`,
matching Enzyme's convention of returning `df/dRe(x) + im*df/dIm(x)` for a
`Complex`-typed `Duplicated` variable.
"""
function fd_grad_at(f, x0, i; eps=1e-7)
    f0 = f(x0)
    xr = copy(x0)
    xr[i] += eps
    xi = copy(x0)
    xi[i] += eps * im
    return ((f(xr) - f0) / eps) + im * ((f(xi) - f0) / eps)
end

@testset "Enzyme adjoint AD (FFTW mul! rule)" begin
    grid = create_grid(2^6, 10e-12, 1550e-9)

    @testset "isolated to_time (unnormalized forward FFT)" begin
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        plan = plan_fft(u0)

        loss(u, p) = begin
            y = similar(u)
            mul!(y, p, u)
            sum(abs2, y)
        end

        i = argmax(abs.(u0))
        du = zero(u0)
        Enzyme.autodiff(
            Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(u0, du), Enzyme.Const(plan)
        )
        g_fd = fd_grad_at(u -> loss(u, plan), u0, i)
        @test isapprox(du[i], g_fd; rtol=1e-4, atol=1e-6)
    end

    @testset "isolated to_freq (normalized inverse FFT / ScaledPlan)" begin
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        plan = plan_ifft(u0)

        loss(u, p) = begin
            y = similar(u)
            mul!(y, p, u)
            sum(abs2, y)
        end

        i = argmax(abs.(u0))
        du = zero(u0)
        Enzyme.autodiff(
            Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(u0, du), Enzyme.Const(plan)
        )
        g_fd = fd_grad_at(u -> loss(u, plan), u0, i)
        @test isapprox(du[i], g_fd; rtol=1e-4, atol=1e-6)
    end

    @testset "full _spm nonlinear step (FFT + Kerr term)" begin
        medium = Medium(0.01, 0.11, 0.0, [-1.0e-26], 1550e-9)
        params = SimParams(; medium=medium, solver=SSFM(1e-5))
        template = zeros(ComplexF64, grid.N)
        model = Soliton.build_physics_model(grid, params, template)
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        i = argmax(abs.(u0))

        loss(u, m) = sum(abs2, Soliton._spm(u, m, 0.0))

        g_fd = fd_grad_at(u -> loss(u, model), u0, i)

        du = zero(u0)
        shadow = Enzyme.make_zero(model)
        Enzyme.autodiff(
            Enzyme.Reverse,
            loss,
            Enzyme.Active,
            Enzyme.Duplicated(u0, du),
            Enzyme.Duplicated(model, shadow),
        )
        @test isapprox(du[i], g_fd; rtol=1e-4, atol=1e-6)
    end

    @testset "full SSFM propagation end-to-end (roadmap step 2)" begin
        # Roadmap step 2 in docs/src/dev/adjoint_ad.md: differentiate a whole
        # fixed-step SSFM `propagate` loop (not just one isolated nonlinear
        # step) for a scalar loss, w.r.t. `medium.gamma` and
        # `TaylorDispersion.betas`. `Medium`/`SimParams`/`PhysicsModel` are all
        # rebuilt from the active parameters *inside* the differentiated
        # function — unlike ForwardDiff, Enzyme differentiates the compiled
        # program directly rather than needing `Dual`-typed containers, so
        # this (including the `plan_fft`/`plan_ifft` construction, which
        # doesn't depend on gamma/betas and so is simply treated as inactive)
        # is not a problem the way it would be for forward-mode.
        #
        # `dz == L` (a single physical step) keeps this fast while still
        # exercising the full step machinery: the initial linear half-step,
        # the nonlinear-operator evaluation, the Euler update, the (no-op for
        # a plain `Medium`) ASE-noise hook, and the final half-step/output copy.
        grid = create_grid(2^6, 10e-12, 1550e-9)
        pulse0 = sech_pulse(grid, 100.0, 1e-12)
        At0 = copy(pulse0.At)
        AW0 = copy(pulse0.AW)
        L = 0.01
        lambda0 = 1550e-9
        dz = L
        z_saves = 2

        function ssfm_energy_loss(
            gamma::Real,
            betas::AbstractVector{<:Real},
            grid::Grid,
            At0::AbstractVector,
            AW0::AbstractVector,
            L::Real,
            lambda0::Real,
            dz::Real,
            z_saves::Int,
        )
            medium = Medium(L, gamma, 0.0, betas, lambda0)
            params = SimParams(;
                medium=medium,
                z_saves=z_saves,
                raman_model=nothing,
                self_steepening=false,
                solver=SSFM(dz),
                save_freq=false,
            )
            model = Soliton.build_physics_model(grid, params, At0)
            pulse = Pulse(copy(At0), copy(AW0), grid)
            _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
            return sum(abs2, At[:, end])
        end

        gamma0 = 0.11
        betas0 = [-1.0e-26]

        @testset "gradient w.r.t. gamma" begin
            loss_of_gamma(g) =
                ssfm_energy_loss(g, betas0, grid, At0, AW0, L, lambda0, dz, z_saves)
            h = 1e-4 * gamma0
            g_fd = (loss_of_gamma(gamma0 + h) - loss_of_gamma(gamma0 - h)) / (2h)

            dgamma = Enzyme.autodiff(
                Enzyme.Reverse,
                ssfm_energy_loss,
                Enzyme.Active,
                Enzyme.Active(gamma0),
                Enzyme.Const(betas0),
                Enzyme.Const(grid),
                Enzyme.Const(At0),
                Enzyme.Const(AW0),
                Enzyme.Const(L),
                Enzyme.Const(lambda0),
                Enzyme.Const(dz),
                Enzyme.Const(z_saves),
            )[1][1]

            @test isapprox(dgamma, g_fd; rtol=1e-3)
        end

        @testset "gradient w.r.t. betas (beta2)" begin
            loss_of_beta2(b2) =
                ssfm_energy_loss(gamma0, [b2], grid, At0, AW0, L, lambda0, dz, z_saves)
            h = 1e-4 * abs(betas0[1])
            g_fd = (loss_of_beta2(betas0[1] + h) - loss_of_beta2(betas0[1] - h)) / (2h)

            dbetas = zero(betas0)
            Enzyme.autodiff(
                Enzyme.Reverse,
                ssfm_energy_loss,
                Enzyme.Active,
                Enzyme.Const(gamma0),
                Enzyme.Duplicated(betas0, dbetas),
                Enzyme.Const(grid),
                Enzyme.Const(At0),
                Enzyme.Const(AW0),
                Enzyme.Const(L),
                Enzyme.Const(lambda0),
                Enzyme.Const(dz),
                Enzyme.Const(z_saves),
            )

            @test isapprox(dbetas[1], g_fd; rtol=1e-3)
        end
    end
end
