using Test
using JuGNLSE
using FFTW
using LinearAlgebra

@testset "PropagationSteps" begin
    # Setup for tests (using real JuGNLSE types)
    N = 2^10
    time_window = 10e-12
    lambda0 = 1550e-9
    grid = create_grid(N, time_window, lambda0)

    P0 = 1000.0 # Peak power in Watts
    T0 = 50e-15 # Pulse duration in seconds
    initial_pulse = sech_pulse(grid, P0, T0)
    initial_energy = pulse_energy(initial_pulse)

    # Test 1: Loss step
    @testset "Loss Step" begin
        pulse = deepcopy(initial_pulse)
        loss_dB = 3.0 # 3 dB loss means energy halves
        loss_step = Loss(loss_dB)
        propagate!(pulse, loss_step)
        
        final_energy = pulse_energy(pulse)
        expected_energy = initial_energy * 10^(-loss_dB / 10)
        @test final_energy ≈ expected_energy rtol=1e-6
    end

    # Test 2: Amplifier step
    @testset "Amplifier Step" begin
        pulse = deepcopy(initial_pulse)
        gain_dB = 3.0 # 3 dB gain means energy doubles
        amplifier_step = Amplifier(gain_dB)
        propagate!(pulse, amplifier_step)
        
        final_energy = pulse_energy(pulse)
        expected_energy = initial_energy * 10^(gain_dB / 10)
        @test final_energy ≈ expected_energy rtol=1e-6
    end

    # Test 3: Fiber step
    @testset "Fiber Step" begin
        pulse = deepcopy(initial_pulse)
        
        # Create a real medium
        dispersion = TaylorDispersion([-11.83e-27])
        medium = Medium(length=0.1, gamma=0.11, loss=0.0, dispersion=dispersion, lambda0=lambda0)
        
        fiber_length = 0.1 # meters
        fiber_step = Fiber(medium, fiber_length)
        
        # Propagate through fiber
        # This will run the real solver now
        sol = propagate!(pulse, fiber_step; progress=false)
        
        @test size(sol.At, 2) > 1 # Should have multiple distance points
        @test sol.Z[end] ≈ fiber_length # Check if length is correctly recorded
    end

    # Test 4: Filter step
    @testset "Filter Step" begin
        pulse = deepcopy(initial_pulse)
        
        # A dummy filter function that zeroes out high frequencies
        function dummy_filter(W, AW)
            filtered_AW = deepcopy(AW)
            threshold_W = 2π * 100e9 # 100 GHz
            filtered_AW[abs.(W) .> threshold_W] .= 0.0
            return filtered_AW
        end

        filter_step = Filter(dummy_filter)
        
        initial_AW_sum = sum(abs2.(pulse.AW))
        propagate!(pulse, filter_step)
        final_AW_sum = sum(abs2.(pulse.AW))
        
        # Expect AW to change due to filtering
        @test !(final_AW_sum ≈ initial_AW_sum)
        # Verify At is updated
        @test sum(abs2.(ifft(pulse.AW) .* pulse.grid.N)) ≈ sum(abs2.(pulse.At)) rtol=1e-6
    end
end
