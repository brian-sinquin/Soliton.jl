using Test
using Soliton
using Enzyme
using FFTW: plan_fft, plan_ifft, fftshift
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
        # step) for a scalar loss, w.r.t. `TaylorDispersion.betas`.
        #
        # First attempt (kept only in git history) rebuilt `Medium`/
        # `SimParams`/`PhysicsModel` from scratch *inside* the differentiated
        # function. That hits a *different* failure than the single-step
        # case documented above: `EnzymeRuntimeActivityError` inside
        # `build_physics_model`, not from `model`'s buffers being used as
        # `Const` scratch space (the earlier, already-solved error), but from
        # `_to_device`'s `similar(template, ...)`/`fill!` — a single helper
        # reused to build both active fields (`D`, which depends on `betas`)
        # and inactive ones (`gamma_W`/`W`, constant when self-steepening is
        # off) — so Enzyme's static activity analysis can't separate the two
        # call sites and flags a "mismatched activity" phi node. This is a
        # real, currently-open gap: making whole-model reconstruction
        # differentiable would need either teaching Enzyme's analysis about
        # `_to_device`, or restructuring `build_physics_model` so active and
        # inactive fields are never built through the same generic helper.
        #
        # This test sidesteps that gap the way the already-working `_spm`
        # test above does: build `model` *once*, then only *mutate* its
        # existing `D` buffer in place from `betas` inside the differentiated
        # function (via `Duplicated(model, shadow)`, exactly like the
        # existing single-step test's buffer mutation) rather than
        # reconstructing the whole struct. `medium.gamma` is a plain
        # immutable `Float64` field with no in-place buffer to mutate the
        # same way, so a `gamma`-gradient version of this test would need
        # `PhysicsModel` to hold gamma in a mutable container (e.g. a
        # `Ref`/1-element `Vector`) — not done here; only `betas` is
        # validated end-to-end for now.
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
        gamma0 = 0.11
        betas0 = [-1.0e-26]

        medium0 = Medium(L, gamma0, 0.0, betas0, lambda0)
        params = SimParams(;
            medium=medium0,
            z_saves=z_saves,
            raman_model=nothing,
            self_steepening=false,
            solver=SSFM(dz),
            save_freq=false,
        )
        model = Soliton.build_physics_model(grid, params, At0)

        function ssfm_energy_loss_betas(
            betas::AbstractVector{<:Real},
            model::Soliton.PhysicsModel,
            grid::Grid,
            At0::AbstractVector,
            AW0::AbstractVector,
            params::SimParams,
        )
            m_disp = Medium(L, gamma0, 0.0, collect(betas), lambda0)
            model.D .= fftshift(dispersion_operator(grid, m_disp))
            pulse = Pulse(copy(At0), copy(AW0), grid)
            _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
            return sum(abs2, At[:, end])
        end

        @testset "gradient w.r.t. betas (beta2)" begin
            function loss_of_beta2(b2)
                m = Soliton.build_physics_model(grid, params, At0)
                m_disp = Medium(L, gamma0, 0.0, [b2], lambda0)
                m.D .= fftshift(dispersion_operator(grid, m_disp))
                pulse = Pulse(copy(At0), copy(AW0), grid)
                _, At, _ = Soliton.propagate(m, pulse, params, params.solver, false)
                return sum(abs2, At[:, end])
            end
            h = 1e-4 * abs(betas0[1])
            g_fd = (loss_of_beta2(betas0[1] + h) - loss_of_beta2(betas0[1] - h)) / (2h)

            dbetas = zero(betas0)
            shadow = Enzyme.make_zero(model)
            Enzyme.autodiff(
                Enzyme.Reverse,
                ssfm_energy_loss_betas,
                Enzyme.Active,
                Enzyme.Duplicated(betas0, dbetas),
                Enzyme.Duplicated(model, shadow),
                Enzyme.Const(grid),
                Enzyme.Const(At0),
                Enzyme.Const(AW0),
                Enzyme.Const(params),
            )

            @test isapprox(dbetas[1], g_fd; rtol=1e-3)
        end
    end
end
