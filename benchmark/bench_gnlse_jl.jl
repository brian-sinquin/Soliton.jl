using Soliton
using Statistics
using LinearAlgebra

# Scenario: SCG in PCF
function run_gnlse_benchmark()
    # 1. Grid
    N = 2^14
    time_window = 10e-12  # 10 ps
    lambda0 = 835e-9
    grid = create_grid(N, time_window, lambda0)

    # 2. Medium (PCF parameters typical for SCG)
    # Betas at 835 nm:
    betas = [
        -11.830e-27,
        8.1038e-41,
        -9.5205e-56,
        2.0737e-70,
        -5.3943e-85,
        1.3486e-99,
        -2.5495e-114,
        3.0524e-129,
        -1.7140e-144,
    ]

    dispersion = TaylorDispersion(betas)
    gamma = 0.11 # 1 / (W * m)
    length_m = 0.15 # 15 cm

    medium = Medium(length_m, gamma, 0.0, dispersion, lambda0)

    # 3. Pulse
    T0 = 50e-15  # 50 fs
    P0 = 10000.0 # 10 kW
    pulse = sech_pulse(grid, P0, T0)

    # 4. Params (Raman on, Self-steepening on)
    params = SimParams(;
        medium=medium,
        raman_model=BlowWood(),
        self_steepening=true,
        solver=SSFM(0.0001), # Use RK4IP adaptive later, or fixed step SSFM
    )

    # 5. Run (Warmup)
    println("Warming up Soliton...")
    warmup_start = time_ns()
    solve(pulse, params; progress=false)
    warmup_end = time_ns()
    println("Warmup time: ", (warmup_end - warmup_start) / 1e9, " s")

    # 6. Benchmark
    println("Running benchmark...")
    bench_start = time_ns()
    sol = solve(pulse, params; progress=false)
    bench_end = time_ns()
    exec_time = (bench_end - bench_start) / 1e9
    println("Execution time: ", exec_time, " s")

    # Save timing to a file for the runner
    open("time_gnlse_jl.txt", "w") do io
        println(io, exec_time)
    end
end

run_gnlse_benchmark()
