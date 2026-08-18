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
end
