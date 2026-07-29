```@meta
CurrentModule = JuGNLSE
```

# Raman Scattering Models

Raman scattering introduces a delayed, frequency-dependent nonlinear response that is important for pulses shorter than ~1 ps. JuGNLSE.jl provides three validated models for fused-silica fibers.

## Choosing a Raman Model

Pass a Raman model to [`SimParams`](@ref):

```julia
params = SimParams(;
    medium      = medium,
    raman_model = BlowWood(),      # or LinAgrawal() or Hollenbeck()
)
```

To disable Raman scattering (pure Kerr):

```julia
params = SimParams(;
    medium      = medium,
    raman_model = nothing,
)
```

## Available Models

### `BlowWood()`

The Blow & Wood (1989) two-exponential model. Simple and widely used:

```math
h_R(t) = \frac{\tau_1^2 + \tau_2^2}{\tau_1 \tau_2^2} e^{-t/\tau_2} \sin(t/\tau_1)
```

Default parameters: ``\tau_1 = 12.2`` fs, ``\tau_2 = 32`` fs, ``f_R = 0.18``.

```julia
raman = BlowWood()
```

### `LinAgrawal()`

The Lin & Agrawal (2006) model, fits measured Raman gain spectrum of silica more accurately over a wider bandwidth:

```julia
raman = LinAgrawal()
```

### `Hollenbeck()`

The Hollenbeck & Cantrell (2002) model — the most accurate for fused silica, incorporating both isotropic and anisotropic contributions:

```julia
raman = Hollenbeck()
```

!!! tip
    For broadband supercontinuum simulations (>1 octave), `Hollenbeck()` is recommended as it fits the silica Raman gain spectrum most accurately.

## Inspecting the Raman Response

The time-domain Raman response can be inspected directly:

```julia
fr, h_R = raman_response(grid, BlowWood())

println("Fractional Raman contribution: ", fr)
# plot(grid.t * 1e15, h_R, xlabel="t [fs]", label="hR(t)")
```

## Effect on Pulse Evolution

Raman scattering produces several physically important effects:

| Phenomenon | Description |
|:---|:---|
| **Soliton self-frequency shift** | Red-shift of fundamental solitons due to Raman amplification |
| **Raman gain** | Amplification of red-shifted spectral components |
| **Supercontinuum generation** | Combined Raman + soliton fission extends the spectrum |

### Example: Observing the Soliton Self-Frequency Shift

```julia
using JuGNLSE

grid = create_grid(2^12, 20e-12, 1550e-9)

medium = Medium(;
    length   = 5.0,
    gamma    = 0.01,
    betas    = [-21.5e-27],
    lambda0  = 1550e-9,
)

# Fundamental soliton input (T₀=50 fs → ~12 nm SSFS visible over 10 soliton periods)
# Note: SSFS ∝ T₀⁻⁴; using T₀=200 fs gives only ~0.05 nm (invisible!)
T0 = 50e-15  # 50 fs soliton half-width
P0 = abs(medium.dispersion.betas[1]) / (medium.gamma * T0^2)
pulse = sech_pulse(grid, P0, T0 * 2 * log(1 + sqrt(2)))

# With Raman: expect red-shift
params_raman = SimParams(; medium=medium, raman_model=Hollenbeck(), z_saves=100)
sol_raman = solve(pulse, params_raman)

# Without Raman: soliton stays centered
params_kerr = SimParams(; medium=medium, raman_model=nothing, z_saves=100)
sol_kerr = solve(pulse, params_kerr)
```
