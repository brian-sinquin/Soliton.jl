using Test
using Soliton
using Enzyme
using FFTW: plan_fft, plan_ifft, fftshift
using LinearAlgebra: mul!, dot

const SolitonEnzyme = Base.get_extension(Soliton, :SolitonEnzymeExt)

# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------

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

"""
Central finite difference of a real-valued `f(x)` w.r.t. the real `x0[i]`.
"""
function fd_central(f, x0, i; h=1e-7)
    xp = copy(x0)
    xp[i] += h
    xm = copy(x0)
    xm[i] -= h
    return (f(xp) - f(xm)) / (2h)
end

relative_difference(a, b) = abs(a - b) / max(abs(b), 1e-300)

# ---------------------------------------------------------------------------
# Shared fixtures for the pulse-shape gradient surface
# ---------------------------------------------------------------------------

"""
Build a `(model, params)` pair for a fixed-step SSFM run over `n_steps` steps of
a single-`Medium` fiber. Everything optional (Raman, self-steepening, spectral
saving) is off so each test isolates one variable.
"""
function ssfm_fixture(grid; L, gamma, beta2, n_steps=1)
    medium = Medium(L, gamma, 0.0, [beta2], 1550e-9)
    params = SimParams(;
        medium=medium,
        z_saves=2,
        raman_model=nothing,
        self_steepening=false,
        solver=SSFM(L / n_steps),
        save_freq=false,
    )
    model = Soliton.build_physics_model(grid, params, zeros(ComplexF64, grid.N))
    return model, params
end

fresh_pulse(grid) = Pulse(zeros(ComplexF64, grid.N), zeros(ComplexF64, grid.N), grid)

"""
Differentiate `loss(theta, model, pulse, params)` w.r.t. the real vector `theta0`
with Enzyme, and compare component `i` against a central finite difference of
the same loss.

`model` is `Duplicated` with a throwaway zero shadow even though its own
gradient is never read: `propagate` and the nonlinear-step functions write
active, pulse-derived data through `model`'s scratch buffers, and a `Const`
model leaves Enzyme no shadow memory for those intermediates — it silently
returns an all-zero gradient. See `docs/src/dev/adjoint_ad.md`.
"""
function check_theta_gradient(loss, grid, model, params, theta0, i; h=1e-7)
    g_fd = fd_central(theta0, i; h=h) do theta
        loss(theta, model, fresh_pulse(grid), params)
    end

    dtheta = zero(theta0)
    pulse = fresh_pulse(grid)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        loss,
        Enzyme.Active,
        Enzyme.Duplicated(theta0, dtheta),
        Enzyme.Duplicated(model, Enzyme.make_zero(model)),
        Enzyme.Duplicated(pulse, Enzyme.make_zero(pulse)),
        Enzyme.Const(params),
    )
    return dtheta[i], g_fd
end

# Losses exercised below. They share a signature so one driver covers them all;
# the ones that never propagate simply ignore `params`.

function loss_mul_into_field(theta, model, pulse, params)
    @. pulse.At = complex(theta, 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    return sum(abs2, pulse.AW)
end

function loss_copy_field(theta, model, pulse, params)
    @. pulse.At = complex(theta, 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    U = copy(pulse.AW)
    return sum(abs2, U)
end

function loss_output_energy(theta, model, pulse, params)
    @. pulse.At = complex(theta, 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
    return sum(abs2, At[:, end])
end

function loss_shape_mismatch(theta, model, pulse, params)
    @. pulse.At = complex(theta, 0.0)
    mul!(pulse.AW, model.to_freq, pulse.At)
    _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
    return sum(abs2, abs.(At[:, end]) .- abs.(theta))
end

@testset "Enzyme adjoint AD (FFTW mul! rule)" begin
    # Everything below differentiates through FFTW, which only works because the
    # weak-dependency extension is loaded. Assert that up front so a packaging
    # regression reads as "extension missing" rather than as a wall of
    # `EnzymeNoDerivativeError`s.
    @test SolitonEnzyme !== nothing

    grid = create_grid(2^6, 10e-12, 1550e-9)

    @testset "adjoint plan identity <Px, y> == <x, P'y>" begin
        # Direct algebraic check of the extension's adjoint factorization,
        # independent of Enzyme: if the normalization is wrong, this fails
        # immediately and unambiguously.
        #
        # The `(N, 2)` region-1 plans are what the vectorial solver builds.
        # They are the reason the scale is derived from the plan itself rather
        # than from `length(y)`: for those, the DFT normalization is the
        # transform length `N`, not the array length `2N`.
        v = zeros(ComplexF64, 8)
        m = zeros(ComplexF64, 8, 2)
        plans = (
            ("plan_fft (vector)", plan_fft(v), v),
            ("plan_ifft (vector, ScaledPlan)", plan_ifft(v), v),
            ("plan_fft (N x 2, region 1)", plan_fft(m, 1), m),
            ("plan_ifft (N x 2, region 1)", plan_ifft(m, 1), m),
        )
        for (name, p, template) in plans
            @testset "$name" begin
                x = randn(ComplexF64, size(template))
                y = randn(ComplexF64, size(template))
                padj, scale = SolitonEnzyme._adjoint_plan(p)
                @test dot(p * x, y) ≈ dot(x, scale * (padj * y))
                # AbstractFFTs exposes the same operator as `p'` (its
                # AdjointPlan). We can't use it directly — AdjointPlan has no
                # `mul!`, so it allocates — but it is a maintained upstream
                # oracle for our factorization's normalization.
                @test scale * (padj * y) ≈ p' * y
            end
        end
    end

    @testset "isolated plan application" begin
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        i = argmax(abs.(u0))
        loss(u, p) = begin
            y = similar(u)
            mul!(y, p, u)
            sum(abs2, y)
        end

        plans = (
            "to_time (unnormalized forward FFT)" => plan_fft(u0),
            "to_freq (normalized inverse FFT / ScaledPlan)" => plan_ifft(u0),
        )
        for (name, plan) in plans
            @testset "$name" begin
                du = zero(u0)
                Enzyme.autodiff(
                    Enzyme.Reverse,
                    loss,
                    Enzyme.Active,
                    Enzyme.Duplicated(u0, du),
                    Enzyme.Const(plan),
                )
                g_fd = fd_grad_at(u -> loss(u, plan), u0, i)
                @test isapprox(du[i], g_fd; rtol=1e-4, atol=1e-6)
            end
        end
    end

    @testset "forward mode (linear plan => tangent obeys the same map)" begin
        # Reverse mode tapes an entire propagation to produce a gradient, so for
        # a design problem with a handful of parameters forward mode is cheaper
        # in both time and memory. These check the forward rule computes the
        # directional derivative correctly.
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        loss(u, p) = begin
            y = similar(u)
            mul!(y, p, u)
            sum(abs2, y)
        end

        for (name, plan) in ("plan_fft" => plan_fft(u0), "plan_ifft" => plan_ifft(u0))
            @testset "isolated $name" begin
                seed = randn(ComplexF64, length(u0))
                # Plain `Forward` (not `ForwardWithPrimal`) returns a 1-tuple
                # holding only the derivative.
                (g_fwd,) = Enzyme.autodiff(
                    Enzyme.Forward,
                    loss,
                    Enzyme.Duplicated,
                    Enzyme.Duplicated(u0, seed),
                    Enzyme.Const(plan),
                )
                # Forward mode gives d/dε f(u0 + ε*seed); compare against a
                # central difference along that same direction.
                h = 1e-7
                g_fd = (loss(u0 .+ h .* seed, plan) - loss(u0 .- h .* seed, plan)) / (2h)
                @test isapprox(g_fwd, g_fd; rtol=1e-4, atol=1e-6)
            end
        end

        @testset "full SSFM propagation w.r.t. beta2" begin
            # The differentiation surface `ad_ssfm_enzyme_compression.jl` uses,
            # which optimizes a single scalar through the whole solver — the
            # case where forward mode should be preferred outright.
            L, lambda0, gamma0 = 0.01, 1550e-9, 0.11
            betas0 = [-1.0e-26]
            model, params = ssfm_fixture(grid; L=L, gamma=gamma0, beta2=betas0[1])
            pulse0 = sech_pulse(grid, 100.0, 1e-12)
            pulse_const = Pulse(copy(pulse0.At), copy(pulse0.AW), grid)

            function fwd_loss(betas, model, pulse, params)
                m_disp = Medium(L, gamma0, 0.0, collect(betas), lambda0)
                model.D .= fftshift(dispersion_operator(pulse.grid, m_disp))
                _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
                return sum(abs2, At[:, end])
            end

            (g_fwd,) = Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Forward),
                fwd_loss,
                Enzyme.Duplicated,
                Enzyme.Duplicated(betas0, [1.0]),   # seed = d/d(beta2)
                Enzyme.Duplicated(model, Enzyme.make_zero(model)),
                Enzyme.Const(pulse_const),
                Enzyme.Const(params),
            )

            h = 1e-4 * abs(betas0[1])
            g_fd = fd_central(betas0, 1; h=h) do b
                fwd_loss(b, model, pulse_const, params)
            end
            println(
                "  [diag] forward mode d/dbeta2: Enzyme=", g_fwd, "  FD=", g_fd,
                "  rel.diff=", relative_difference(g_fwd, g_fd),
            )
            # Same loose rtol as the reverse-mode betas test, and for the same
            # reason: h ~ 1e-30 in absolute terms limits the FD estimate, not
            # the AD one.
            @test isapprox(g_fwd, g_fd; rtol=1e-2)
        end
    end

    @testset "full _spm nonlinear step (FFT + Kerr term)" begin
        medium = Medium(0.01, 0.11, 0.0, [-1.0e-26], 1550e-9)
        params = SimParams(; medium=medium, solver=SSFM(1e-5))
        model = Soliton.build_physics_model(grid, params, zeros(ComplexF64, grid.N))
        u0 = Vector{ComplexF64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)) .+ 0im)
        i = argmax(abs.(u0))

        loss(u, m) = sum(abs2, Soliton._spm(u, m, 0.0))
        g_fd = fd_grad_at(u -> loss(u, model), u0, i)

        du = zero(u0)
        Enzyme.autodiff(
            Enzyme.Reverse,
            loss,
            Enzyme.Active,
            Enzyme.Duplicated(u0, du),
            Enzyme.Duplicated(model, Enzyme.make_zero(model)),
        )
        @test isapprox(du[i], g_fd; rtol=1e-4, atol=1e-6)
    end

    @testset "full SSFM propagation w.r.t. betas (roadmap step 2)" begin
        # Differentiate a whole fixed-step SSFM `propagate` loop w.r.t.
        # `TaylorDispersion.betas`. Two caller-side patterns are load-bearing
        # here, both documented in docs/src/dev/adjoint_ad.md:
        #
        #  - `model` is built once outside the differentiated function, which
        #    then only *mutates* its existing `D` buffer. Rebuilding the model
        #    inside instead hits an `EnzymeRuntimeActivityError` in
        #    `build_physics_model`, where the `_to_device` helper constructs
        #    both active (`D`) and inactive (`gamma_W`/`W`) fields.
        #  - `pulse` is likewise built once and passed as `Const`; `propagate`
        #    never mutates it. Allocating a fresh `Pulse` inside the closure —
        #    even from entirely `Const` inputs — trips the same error.
        #
        # `dz == L` (one physical step) keeps this fast while still covering
        # the initial half-step, the nonlinear evaluation, the Euler update,
        # the (no-op here) ASE hook, and the final half-step/output copy.
        pulse0 = sech_pulse(grid, 100.0, 1e-12)
        At0 = copy(pulse0.At)
        L, lambda0, gamma0 = 0.01, 1550e-9, 0.11
        betas0 = [-1.0e-26]

        model, params = ssfm_fixture(grid; L=L, gamma=gamma0, beta2=betas0[1])
        pulse_const = Pulse(copy(At0), copy(pulse0.AW), grid)

        function loss_betas(betas, model, pulse, params)
            m_disp = Medium(L, gamma0, 0.0, collect(betas), lambda0)
            model.D .= fftshift(dispersion_operator(pulse.grid, m_disp))
            _, At, _ = Soliton.propagate(model, pulse, params, params.solver, false)
            return sum(abs2, At[:, end])
        end

        h = 1e-4 * abs(betas0[1])
        g_fd = fd_central(betas0, 1; h=h) do b
            loss_betas(b, model, pulse_const, params)
        end

        dbetas = zero(betas0)
        # `propagate` writes the `Const`-derived initial condition into column 1
        # of its freshly-allocated `At_out`, then active, betas-derived columns
        # into the rest — genuinely mixed activity, which is exactly what
        # `set_runtime_activity` is for. It is not taken on faith: the result is
        # checked against the independent finite difference below, which is what
        # would catch it silently zeroing the gradient.
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            loss_betas,
            Enzyme.Active,
            Enzyme.Duplicated(betas0, dbetas),
            Enzyme.Duplicated(model, Enzyme.make_zero(model)),
            Enzyme.Const(pulse_const),
            Enzyme.Const(params),
        )

        # rtol is looser than the isolated operators' 1e-4 because the FD step
        # is necessarily tiny in absolute terms (h = 1e-4*|beta2| ~ 1e-30) next
        # to a loss computed through a full SSFM step — a limit on how tightly
        # the FD estimate itself can be trusted, not on the Enzyme gradient.
        @test isapprox(dbetas[1], g_fd; rtol=1e-2)
    end

    @testset "gradient w.r.t. pulse shape (Duplicated pulse)" begin
        # The mirror image of the betas surface above: the fiber is fixed and
        # the *pulse's own shape* is the design variable, as in
        # `examples/ad_soliton_shape_recovery.jl`.
        #
        # These cases form a bisection ladder — each adds exactly one ingredient
        # to the one before (struct-field write, local copy, the SSFM loop, the
        # nonlinear term, multiple steps, a second direct use of `theta`) so a
        # regression names its own cause.
        theta0 = Vector{Float64}(exp.(-grid.t .^ 2 ./ (2 * (1e-12)^2)))
        i = argmax(theta0)

        # The last case needs much stronger dispersion over a much longer fiber
        # than the rest: its loss is |At_out| - |theta|, and at L=0.01 with
        # beta2=-1e-26 the pulse barely reshapes, so both sides land at ~1e-17
        # and any tolerance "passes" without checking anything.
        cases = (
            (
                name="isolated mul! into a Duplicated struct field",
                loss=loss_mul_into_field, L=0.01, gamma=0.11, beta2=-1.0e-26,
                n_steps=1, rtol=1e-4,
            ),
            (
                name="isolated copy(pulse.AW) into a local",
                loss=loss_copy_field, L=0.01, gamma=0.11, beta2=-1.0e-26,
                n_steps=1, rtol=1e-4,
            ),
            (
                name="full SSFM, linear (gamma=0), single step",
                loss=loss_output_energy, L=0.01, gamma=0.0, beta2=-1.0e-26,
                n_steps=1, rtol=1e-3,
            ),
            (
                name="full SSFM, nonlinear (gamma!=0), single step",
                loss=loss_output_energy, L=0.01, gamma=0.11, beta2=-1.0e-26,
                n_steps=1, rtol=1e-3,
            ),
            (
                name="full SSFM, linear, multi-step (n_steps=5)",
                loss=loss_output_energy, L=0.01, gamma=0.0, beta2=-1.0e-26,
                n_steps=5, rtol=1e-3,
            ),
            (
                name="full SSFM, theta used twice (propagated + direct)",
                loss=loss_shape_mismatch, L=1.0, gamma=0.0, beta2=-1.0e-24,
                n_steps=1, rtol=1e-3,
            ),
        )

        for case in cases
            @testset "$(case.name)" begin
                model, params = ssfm_fixture(
                    grid; L=case.L, gamma=case.gamma, beta2=case.beta2,
                    n_steps=case.n_steps,
                )
                g_ad, g_fd = check_theta_gradient(
                    case.loss, grid, model, params, theta0, i
                )
                println(
                    "  [diag] ", case.name, ": Enzyme=", g_ad, "  FD=", g_fd,
                    "  rel.diff=", relative_difference(g_ad, g_fd),
                )
                @test isapprox(g_ad, g_fd; rtol=case.rtol, atol=1e-6)
            end
        end
    end
end
