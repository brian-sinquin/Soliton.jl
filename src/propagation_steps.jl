module PropagationSteps

using ..Types: Medium, Pulse, SimParams, GNLSEProblem, Solution, ConstantGamma
using ..Solvers: solve
import ..JuGNLSE: c
using FFTW: ifft

export AbstractPropagationStep, Fiber, Loss, Filter, Amplifier, propagate!

abstract type AbstractPropagationStep end

struct Fiber <: AbstractPropagationStep
    medium::Medium
    length::Float64
    z_saves::Int

    function Fiber(medium::Medium, length::Real, z_saves::Int=200)
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        z_saves > 0 || throw(ArgumentError("z_saves must be positive"))
        segment_medium = Medium(
            length=length,
            gamma=medium.gamma,
            loss=medium.loss,
            dispersion=medium.dispersion,
            lambda0=medium.lambda0,
        )
        new(segment_medium, Float64(length), z_saves)
    end
end

struct Loss <: AbstractPropagationStep
    loss_dB::Float64
    function Loss(loss_dB::Real)
        loss_dB >= 0 || throw(ArgumentError("Loss must be non-negative"))
        new(Float64(loss_dB))
    end
end

function propagate!(pulse::Pulse, step::Fiber; rtol::Float64=1e-6, atol::Float64=1e-8, progress::Bool=true)
    sim_params = SimParams(
        medium=step.medium,
        z_saves=step.z_saves,
        raman_model=nothing,
        self_steepening=false,
        rtol=rtol,
        atol=atol,
    )
    problem = GNLSEProblem(
        medium=step.medium,
        grid=pulse.grid,
        initial_pulse=pulse,
        sim_params=sim_params,
        gamma_coefficient=ConstantGamma(step.medium.gamma),
    )
    sol = solve(problem; progress=progress)
    pulse.At .= sol.At[:, end]
    pulse.AW .= sol.AW[:, end]
    return sol
end

function propagate!(pulse::Pulse, step::Loss)
    loss_factor = 10^(-step.loss_dB / 10)
    pulse.At .*= sqrt(loss_factor)
    pulse.AW .*= sqrt(loss_factor)
    return nothing
end

struct Filter <: AbstractPropagationStep
    filter_function::Function
end

function propagate!(pulse::Pulse, step::Filter)
    pulse.AW .= step.filter_function(pulse.grid.W, pulse.AW)
    pulse.At .= ifft(pulse.AW) .* pulse.grid.N
    return nothing
end

struct Amplifier <: AbstractPropagationStep
    gain_dB::Float64
    function Amplifier(gain_dB::Real)
        new(Float64(gain_dB))
    end
end

function propagate!(pulse::Pulse, step::Amplifier)
    gain_factor = 10^(step.gain_dB / 10)
    pulse.At .*= sqrt(gain_factor)
    pulse.AW .*= sqrt(gain_factor)
    return nothing
end

function propagate!(pulse::Pulse, steps::Vector{<:AbstractPropagationStep}; kwargs...)
    full_solution_Z = Float64[]
    full_solution_At = Matrix{ComplexF64}(undef, length(pulse.At), 0)
    full_solution_AW = Matrix{ComplexF64}(undef, length(pulse.AW), 0)
    current_pulse = Pulse(copy(pulse.At), copy(pulse.AW), pulse.grid)
    for (i, step) in enumerate(steps)
        if step isa Fiber
            sol = propagate!(current_pulse, step; kwargs...)
            append!(full_solution_Z, sol.Z)
            full_solution_At = hcat(full_solution_At, sol.At)
            full_solution_AW = hcat(full_solution_AW, sol.AW)
        else
            propagate!(current_pulse, step)
        end
    end
    if isempty(full_solution_Z)
        return Solution(current_pulse.grid.t, current_pulse.grid.W, current_pulse.grid.omega0, [0.0,], reshape(current_pulse.At, :, 1), reshape(current_pulse.AW, :, 1))
    else
        return Solution(current_pulse.grid.t, current_pulse.grid.W, current_pulse.grid.omega0, full_solution_Z, full_solution_At, full_solution_AW)
    end
end

end # module
