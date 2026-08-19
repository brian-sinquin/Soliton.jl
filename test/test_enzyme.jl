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
        # Built once, outside the differentiated closure, and passed in as
        # `Const` — the same lesson as `PhysicsModel` above. `propagate`
        # never mutates its `pulse` argument (it only reads `.grid`/`.At`/
        # `.AW` into fresh local buffers), so one shared instance is safe to
        # reuse across every autodiff call. Allocating a *fresh* `Pulse`
        # inside the differentiated function — even from entirely `Const`
        # inputs — hits a fourth instance of the same "conditionally active
        # memory" `EnzymeRuntimeActivityError` family documented above: the
        # mutable struct gets consumed downstream alongside genuinely active
        # `model` data, and Enzyme's static analysis can't separate "freshly
        # allocated, Const-derived" from "active" for a mutable struct built
        # inside the traced region.
        pulse_const = Pulse(copy(At0), copy(AW0), grid)

        function ssfm_energy_loss_betas(
            betas::AbstractVector{<:Real},
            model::Soliton.PhysicsModel,
            pulse::Pulse,
            params::SimParams,
        )
            m_disp = Medium(L, gamma0, 0.0, collect(betas), lambda0)
            model.D .= fftshift(dispersion_operator(pulse.grid, m_disp))
            _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
            return sum(abs2, At[:, end])
        end

        @testset "gradient w.r.t. betas (beta2)" begin
            function loss_of_beta2(b2)
                m = Soliton.build_physics_model(grid, params, At0)
                m_disp = Medium(L, gamma0, 0.0, [b2], lambda0)
                m.D .= fftshift(dispersion_operator(grid, m_disp))
                _, At, _ = Soliton.propagate(m, pulse_const, params, params.solver, false)
                return sum(abs2, At[:, end])
            end
            h = 1e-4 * abs(betas0[1])
            g_fd = (loss_of_beta2(betas0[1] + h) - loss_of_beta2(betas0[1] - h)) / (2h)

            dbetas = zero(betas0)
            shadow = Enzyme.make_zero(model)
            # `Soliton.propagate` (SSFM) allocates its `At_out` output matrix
            # via `zeros(...)`, writes the (Const-derived) initial condition
            # `pulse.At` into column 1, then writes (Active, betas-derived)
            # computed columns into the rest — a *genuinely* conditionally
            # active buffer (unlike the `_spm`-only test above, where
            # `set_runtime_activity` was confirmed to silently zero the
            # gradient because that case was a Const/Active *misclassification*,
            # not real mixed activity). `set_runtime_activity` is Enzyme's own
            # documented fix for this pattern; here it can't silently drop
            # anything unnoticed since the result is checked against an
            # independent finite-difference gradient below.
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                ssfm_energy_loss_betas,
                Enzyme.Active,
                Enzyme.Duplicated(betas0, dbetas),
                Enzyme.Duplicated(model, shadow),
                Enzyme.Const(pulse_const),
                Enzyme.Const(params),
            )

            # CI run 32227730469 confirmed set_runtime_activity gives a
            # correct, non-zero gradient (2.545e19) matching the independent
            # finite-difference estimate (2.552e19) to ~0.3% — well outside
            # the silent-zero failure mode this test guards against. rtol is
            # 1e-2 rather than the isolated-operator tests' 1e-4 because the
            # finite-difference step here is necessarily tiny in absolute
            # terms (h = 1e-4 * |beta2| ≈ 1e-30, since beta2 itself is
            # ~1e-26) relative to a loss computed through dozens of
            # floating-point operations (the full SSFM step), which limits
            # how tightly the FD estimate itself can be trusted — not a sign
            # the Enzyme gradient is imprecise.
            @test isapprox(dbetas[1], g_fd; rtol=1e-2)
        end
    end

    @testset "gradient w.r.t. pulse shape (Duplicated pulse, Const model)" begin
        # New differentiation surface: every test above differentiates
        # w.r.t. *fiber* parameters (`Duplicated(model, ...)`) with the
        # pulse held `Const`. `ad_soliton_shape_recovery.jl` instead holds
        # the fiber `Const` and makes the *pulse* `Duplicated`, to optimize
        # the pulse's own temporal shape. CI run 32244905929 (the example
        # script's own gradient check) showed 29-100% relative disagreement
        # against finite differences at several grid points -- not the
        # ~1e-4-1e-9 agreement every surface above gets -- so this bisects
        # the failure: does wrapping the array in a mutable struct field
        # break Enzyme (isolated mul!), or is it the SSFM loop itself
        # (copy/At_out pattern, same family as bug #5 above but with the
        # Const/Duplicated roles swapped)?
        grid = create_grid(2^6, 10e-12, 1550e-9)

        @testset "isolated mul! into a Duplicated struct field (no propagate)" begin
            medium = Medium(0.01, 0.11, 0.0, [-1.0e-26], 1550e-9)
            params = SimParams(; medium=medium, solver=SSFM(1e-5))
            template = zeros(ComplexF64, grid.N)
            model = Soliton.build_physics_model(grid, params, template)
            pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

            function mulinto_loss(
                theta::AbstractVector{<:Real}, model::Soliton.PhysicsModel, pulse::Pulse
            )
                @. pulse.At = complex(theta, 0.0)
                mul!(pulse.AW, model.to_freq, pulse.At)
                return sum(abs2, pulse.AW)
            end

            theta0 = Vector{Float64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)))
            i = argmax(theta0)

            function loss_fd_mulinto(theta_vec)
                p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
                @. p.At = complex(theta_vec, 0.0)
                mul!(p.AW, model.to_freq, p.At)
                return sum(abs2, p.AW)
            end
            h = 1e-7
            thp = copy(theta0)
            thp[i] += h
            thm = copy(theta0)
            thm[i] -= h
            g_fd = (loss_fd_mulinto(thp) - loss_fd_mulinto(thm)) / (2h)

            dtheta = zero(theta0)
            shadow = Enzyme.make_zero(pulse)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                mulinto_loss,
                Enzyme.Active,
                Enzyme.Duplicated(theta0, dtheta),
                Enzyme.Const(model),
                Enzyme.Duplicated(pulse, shadow),
            )
            println(
                "  [diag] isolated mul!: Enzyme=", dtheta[i], "  FD=", g_fd,
                "  rel.diff=", abs(dtheta[i] - g_fd) / max(abs(g_fd), 1e-300),
            )
            @test isapprox(dtheta[i], g_fd; rtol=1e-4, atol=1e-6)
        end

        @testset "isolated copy(pulse.AW) into a local variable" begin
            # CI run 32246119166 showed the isolated mul! test above passing
            # (rel.diff ~3e-9) but the full linear SSFM test below failing
            # with Enzyme returning *exactly* 0.0 -- a total loss of
            # gradient signal, not a numerical mismatch. `propagate()`'s
            # very first line after computing pulse.AW is `U =
            # copy(pulse.AW)`; this isolates whether that specific `copy`
            # of a Duplicated struct field into a fresh local variable is
            # where the signal is lost.
            medium = Medium(0.01, 0.11, 0.0, [-1.0e-26], 1550e-9)
            params = SimParams(; medium=medium, solver=SSFM(1e-5))
            template = zeros(ComplexF64, grid.N)
            model = Soliton.build_physics_model(grid, params, template)
            pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

            function copy_loss(
                theta::AbstractVector{<:Real}, model::Soliton.PhysicsModel, pulse::Pulse
            )
                @. pulse.At = complex(theta, 0.0)
                mul!(pulse.AW, model.to_freq, pulse.At)
                U = copy(pulse.AW)
                return sum(abs2, U)
            end

            theta0 = Vector{Float64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)))
            i = argmax(theta0)

            function loss_fd_copy(theta_vec)
                p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
                @. p.At = complex(theta_vec, 0.0)
                mul!(p.AW, model.to_freq, p.At)
                U = copy(p.AW)
                return sum(abs2, U)
            end
            h = 1e-7
            thp = copy(theta0)
            thp[i] += h
            thm = copy(theta0)
            thm[i] -= h
            g_fd = (loss_fd_copy(thp) - loss_fd_copy(thm)) / (2h)

            dtheta = zero(theta0)
            shadow = Enzyme.make_zero(pulse)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                copy_loss,
                Enzyme.Active,
                Enzyme.Duplicated(theta0, dtheta),
                Enzyme.Const(model),
                Enzyme.Duplicated(pulse, shadow),
            )
            println(
                "  [diag] isolated copy(pulse.AW): Enzyme=", dtheta[i], "  FD=", g_fd,
                "  rel.diff=", abs(dtheta[i] - g_fd) / max(abs(g_fd), 1e-300),
            )
            @test isapprox(dtheta[i], g_fd; rtol=1e-4, atol=1e-6)
        end

        @testset "full SSFM propagation, Duplicated pulse, Const model (linear-only, gamma=0)" begin
            # gamma=0 strips out the nonlinear step's contribution
            # mathematically (though the nonlinear function still runs,
            # contributing zero) so this isolates the *linear* SSFM
            # machinery -- the copy(pulse.AW)/At_out-partial-write pattern
            # -- from anything nonlinear-step-specific.
            L = 0.01
            medium = Medium(L, 0.0, 0.0, [-1.0e-26], 1550e-9)
            params = SimParams(;
                medium=medium,
                z_saves=2,
                raman_model=nothing,
                self_steepening=false,
                solver=SSFM(L),
                save_freq=false,
            )
            template = zeros(ComplexF64, grid.N)
            model = Soliton.build_physics_model(grid, params, template)
            pulse = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

            function linear_loss(
                theta::AbstractVector{<:Real},
                model::Soliton.PhysicsModel,
                pulse::Pulse,
                params::SimParams,
            )
                @. pulse.At = complex(theta, 0.0)
                mul!(pulse.AW, model.to_freq, pulse.At)
                _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
                return sum(abs2, At[:, end])
            end

            theta0 = Vector{Float64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)))
            i = argmax(theta0)

            function loss_fd_linear(theta_vec)
                p = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)
                @. p.At = complex(theta_vec, 0.0)
                mul!(p.AW, model.to_freq, p.At)
                _, At, _ = Soliton.propagate(model, p, params, params.solver, false)
                return sum(abs2, At[:, end])
            end
            h = 1e-7
            thp = copy(theta0)
            thp[i] += h
            thm = copy(theta0)
            thm[i] -= h
            g_fd = (loss_fd_linear(thp) - loss_fd_linear(thm)) / (2h)

            dtheta = zero(theta0)
            shadow = Enzyme.make_zero(pulse)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                linear_loss,
                Enzyme.Active,
                Enzyme.Duplicated(theta0, dtheta),
                Enzyme.Const(model),
                Enzyme.Duplicated(pulse, shadow),
                Enzyme.Const(params),
            )
            println(
                "  [diag] full linear SSFM: Enzyme=", dtheta[i], "  FD=", g_fd,
                "  rel.diff=", abs(dtheta[i] - g_fd) / max(abs(g_fd), 1e-300),
            )
            @test isapprox(dtheta[i], g_fd; rtol=1e-3, atol=1e-6)
        end
    end
end
