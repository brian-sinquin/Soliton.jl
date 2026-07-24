module Solvers

using ..Types: AbstractGNLSESolver, GNLSEProblem, Solution, Medium, Grid, SimParams, AbstractGammaCoefficient, Pulse
using ..Physics: build_physics_model
using ProgressMeter
using FFTW
using LinearAlgebra

export ERK4IP, RK4, solve

"""
    ERK4IP

Fourth-order Runge-Kutta in the Interaction Picture (ERK4IP) solver for GNLSE.
"""
struct ERK4IP <: AbstractGNLSESolver
    dz::Float64
    function ERK4IP(dz::Real)
        dz > 0 || throw(ArgumentError("Step size `dz` must be positive"))
        new(Float64(dz))
    end
end
ERK4IP() = ERK4IP(1e-4) # Default dz

"""
    solve(problem::GNLSEProblem, solver::ERK4IP; progress::Bool=true)

Solves the GNLSE using the ERK4IP method.
"""
function solve(problem::GNLSEProblem, solver::ERK4IP; progress::Bool=true)
    # Unpack problem parameters
    medium = problem.medium
    grid = problem.grid
    initial_pulse = problem.initial_pulse
    sim_params = problem.sim_params
    gamma_coefficient = problem.gamma_coefficient

    # Pre-compute operators for zero-allocation propagation
    model = build_physics_model(grid, sim_params, gamma_coefficient)

    # Initialize solution arrays
    Z = collect(0.0:solver.dz:medium.length)
    Nz = length(Z)
    At = Matrix{ComplexF64}(undef, grid.N, Nz)
    AW = Matrix{ComplexF64}(undef, grid.N, Nz)

    # Initial condition
    At[:, 1] = initial_pulse.At
    AW[:, 1] = initial_pulse.AW

    # Pre-allocate k-arrays for ERK4IP method
    k1_f = similar(model.buf_f1)
    k2_f = similar(model.buf_f1)
    k3_f = similar(model.buf_f1)
    k4_f = similar(model.buf_f1)

    # Progress meter
    pm = Progress(Nz - 1; showspeed=true, enabled=progress, barlen=20)

    # Propagation loop
    for i = 1:Nz-1
        z = Z[i]
        dz = solver.dz

        # ERK4IP steps (solving u_z = L*u + N(u)) in Interaction Picture
        
        # Step 1: k1 = N(u_n)
        k1_f = model.nonlinear_function(At[:, i], model, z)

        # Step 2: k2 = N(exp(L*dz/2)*u_n + (dz/2)*exp(L*dz/2)*k1)
        # Linear part: exp(L*dz/2)*u_n = ifft(exp(D*dz/2) * fft(u_n))
        # Note: AW is in FFT order, D is in FFT order.
        model.buf_f1 .= AW[:, i] .* exp.(model.D .* (dz / 2))
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        mul!(model.buf_t2, model.to_time, k1_f)
        # Combine At_n + (dz/2)*k1_t
        for j in 1:length(model.buf_t1)
            model.buf_t1[j] += (dz / 2) * model.buf_t2[j]
        end
        k2_f .= model.nonlinear_function(model.buf_t1, model, z + dz / 2)

        # Step 3: k3 = N(exp(L*dz/2)*u_n + (dz/2)*exp(L*dz/2)*k2)
        # Recalculate linear part for step 3
        model.buf_f1 .= AW[:, i] .* exp.(model.D .* (dz / 2))
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        mul!(model.buf_t2, model.to_time, k2_f)
        model.buf_t1 .+= (dz / 2) .* model.buf_t2
        k3_f .= model.nonlinear_function(model.buf_t1, model, z + dz / 2)

        # Step 4: k4 = N(exp(L*dz)*u_n + dz*exp(L*dz)*k3)
        model.buf_f1 .= AW[:, i] .* exp.(model.D .* dz)
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        mul!(model.buf_t2, model.to_time, k3_f)
        model.buf_t1 .+= dz .* model.buf_t2
        k4_f .= model.nonlinear_function(model.buf_t1, model, z + dz)

        # Combine the steps (frequency domain)
        # Using explicit broadcasting to avoid broadcasting over 'model' or other non-array fields
        AW[:, i+1] .= AW[:, i] .* exp.(model.D .* dz) .+ (dz / 6) .* (
            k1_f .* exp.(model.D .* dz) .+
            2 .* k2_f .* exp.(model.D .* (dz / 2)) .+
            2 .* k3_f .* exp.(model.D .* (dz / 2)) .+
            k4_f
        )
        mul!(At[:, i+1], model.to_time, AW[:, i+1])

        next!(pm)
    end

    return Solution(grid.t, grid.W, grid.omega0, Z, At, AW)
end

"""
    RK4

Fourth-order Runge-Kutta (RK4) solver for GNLSE.
"""
struct RK4 <: AbstractGNLSESolver
    dz::Float64
    function RK4(dz::Real)
        dz > 0 || throw(ArgumentError("Step size `dz` must be positive"))
        new(Float64(dz))
    end
end
RK4() = RK4(1e-4) # Default dz

"""
    solve(problem::GNLSEProblem, solver::RK4; progress::Bool=true)

Solves the GNLSE using the RK4 method.
"""
function solve(problem::GNLSEProblem, solver::RK4; progress::Bool=true)
    medium = problem.medium
    grid = problem.grid
    initial_pulse = problem.initial_pulse
    sim_params = problem.sim_params
    gamma_coefficient = problem.gamma_coefficient

    model = build_physics_model(grid, sim_params, gamma_coefficient)

    Z = collect(0.0:solver.dz:medium.length)
    Nz = length(Z)
    At = Matrix{ComplexF64}(undef, grid.N, Nz)
    AW = Matrix{ComplexF64}(undef, grid.N, Nz)

    At[:, 1] = initial_pulse.At
    AW[:, 1] = initial_pulse.AW

    k1_f = similar(model.buf_f1)
    k2_f = similar(model.buf_f1)
    k3_f = similar(model.buf_f1)
    k4_f = similar(model.buf_f1)

    pm = Progress(Nz - 1; showspeed=true, enabled=progress, barlen=20)

    for i = 1:Nz-1
        z = Z[i]
        dz = solver.dz

        @. k1_f = dz * (model.D * AW[:, i] + model.nonlinear_function(At[:, i], model, z))
        @. model.buf_f1 = AW[:, i] + k1_f / 2
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        @. k2_f = dz * (model.D * model.buf_f1 + model.nonlinear_function(model.buf_t1, model, z + dz / 2))
        @. model.buf_f1 = AW[:, i] + k2_f / 2
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        @. k3_f = dz * (model.D * model.buf_f1 + model.nonlinear_function(model.buf_t1, model, z + dz / 2))
        @. model.buf_f1 = AW[:, i] + k3_f
        mul!(model.buf_t1, model.to_time, model.buf_f1)
        @. k4_f = dz * (model.D * model.buf_f1 + model.nonlinear_function(model.buf_t1, model, z + dz))
        # Update AW
        AW[:, i+1] .= AW[:, i] .+ (k1_f .+ 2 .* k2_f .+ 2 .* k3_f .+ k4_f) ./ 6
        mul!(At[:, i+1], model.to_time, AW[:, i+1])

        next!(pm)
    end

    return Solution(grid.t, grid.W, grid.omega0, Z, At, AW)
end

function solve(problem::GNLSEProblem; progress::Bool=true)
    solve(problem, ERK4IP(); progress=progress)
end

function solve(pulse::Pulse, params::SimParams, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)
    problem = GNLSEProblem(pulse, params.medium, params)
    solve(problem, solver; progress=progress)
end

end # module
