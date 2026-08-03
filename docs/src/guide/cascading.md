```@meta
CurrentModule = GNLSE
```

# Cascaded Propagation & Piping

GNLSE.jl supports fluent multi-stage simulation pipelines using Julia's native `|>` operator.

## Lumped Elements

Point-like optical components between fiber segments are represented by [`LumpedElement`](@ref) subtypes:

| Type | Description |
|:---|:---|
| `Amplifier(gain_db)` | Boosts field amplitude (positive gain) |
| `Attenuator(loss_db)` | Reduces field amplitude (loss in dB) |
| `Filter(f)` | Applies frequency-domain transfer function `f(ω)` |

```julia
using Soliton

amp      = Amplifier(20.0)         # +20 dB gain
att      = Attenuator(3.0)         # −3 dB
bandpass = Filter(ω -> abs(ω - 2π*2.99792458e8/1550e-9) < 1e12 ? 1.0 : 0.0)
```

Elements can be applied directly to a pulse:
```julia
amplified_pulse = apply(pulse, amp)
```

Or used in a pipeline (see below).

## Piping with `|>`

When `SimParams` receives a `Pulse` as a callable argument, it propagates it and returns a `Solution`. When a `LumpedElement` receives a `Pulse`, it transforms it immediately.

This enables the Julia `|>` pipe operator for clean multi-stage chains:

```julia
using Soliton

grid = create_grid(2^12, 20e-12, 1550e-9)
pulse = sech_pulse(grid, 1000.0, 1e-12)

# Define stages
fiber1 = SimParams(; medium=Medium(5.0, 0.0011, 0.0, [-21.5e-27], 1550e-9))
amp    = Amplifier(13.0)    # +13 dB EDFA
fiber2 = SimParams(; medium=Medium(10.0, 0.0011, 0.2e-3, [-21.5e-27], 1550e-9))
filter = Filter(ω -> exp(-((ω - 2π*c/1550e-9)^2) / (2 * (1e12)^2)))

# Pipe: pulse → fiber1 → amplifier → filter → fiber2 → final solution
sol = pulse |> fiber1 |> amp |> filter |> fiber2
```

!!! note "Solution-to-Pulse conversion"
    When a `SimParams` or `LumpedElement` receives a [`Solution`](@ref) (output of `solve`), it automatically extracts the final field state as a new [`Pulse`](@ref) before processing it. This makes the pipe seamless.

## Cascaded Simulation with `solve`

For more complex topologies, [`solve`](@ref) also accepts a `Vector` of stages
directly, rather than piping them one at a time with `|>`:

```julia
stages = [fiber1, amp, fiber2, filter]
results = solve(pulse, stages)

# results is a Vector of Pulse/Solution objects, one per stage
```

## Example: Fiber Amplifier Chain

```julia
using Soliton

grid = create_grid(2^12, 20e-12, 1550e-9)
pulse = gaussian_pulse(grid, 1.0, 10e-12)  # low power, 10 ps input

# Stage 1: short pre-amplifier fiber
pre_amp_fiber = SimParams(;
    medium = Medium(5.0, 0.005, 0.0, [-21.5e-27], 1550e-9),
    raman_model = nothing,
)

# Lumped EDFA gain
edfa = Amplifier(30.0)   # 30 dB = 1000x power

# Stage 2: main amplifier fiber
main_fiber = SimParams(;
    medium = Medium(20.0, 0.005, 0.0, [-21.5e-27], 1550e-9),
    raman_model = BlowWood(),
    self_steepening = true,
)

# Band-pass filter to clean up ASE
bpf = Filter(ω -> begin
    Δω = ω - 2π * 2.99792458e8 / 1550e-9
    exp(-Δω^2 / (2 * (2π * 1e12)^2))  # 1 THz Gaussian filter
end)

# Run the full chain
sol = pulse |> pre_amp_fiber |> edfa |> main_fiber |> bpf

println("Output peak power: ", maximum(abs2, sol.At[:, end]), " W")
```
