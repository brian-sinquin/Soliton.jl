```@meta
CurrentModule = JuGNLSE
```

# Birefringent & Vectorial Propagation

JuGNLSE.jl supports the simulation of birefringent fibers and waveguides through the **Coupled GNLSE** — propagating two orthogonal polarization components simultaneously.

## When to Use the Vectorial Solver

- Birefringent fibers (PMF, bow-tie, PANDA)
- Polarization-maintaining waveguides
- Study of polarization coupling and rotation
- Soliton trapping via cross-phase modulation (XPM)
- Coherent four-wave mixing between polarization axes

## Setting Up a Birefringent Simulation

### 1. Define a `BirefringentMedium`

```julia
using JuGNLSE

# Separate dispersion models for each polarization axis
disp_x = TaylorDispersion([-21.5e-27])      # x-axis dispersion
disp_y = TaylorDispersion([-21.5e-27])      # y-axis dispersion (can differ)

medium = BirefringentMedium(
    1.0,       # length [m]
    0.0011,    # nonlinear coefficient γ [1/(W·m)]
    0.0,       # loss [dB/m]
    disp_x,    # dispersion model for x polarization
    disp_y,    # dispersion model for y polarization
    1e4,       # Δβ₀ = β₀ₓ - β₀ᵧ [1/m] — birefringence
    1550e-9,   # center wavelength [m]
)
```

### 2. Create a `VectorialPulse`

A [`VectorialPulse`](@ref) holds two time-domain envelopes ``A_x(t)`` and ``A_y(t)``:

```julia
grid = create_grid(2^12, 20e-12, 1550e-9)

# Both axes excited equally
pulse_x = sech_pulse(grid, 500.0, 1e-12)
pulse_y = sech_pulse(grid, 500.0, 1e-12)

vpulse = VectorialPulse(pulse_x.At, pulse_y.At, grid)
```

For a linearly polarized input (all energy in x):
```julia
vpulse = VectorialPulse(pulse_x.At, zeros(ComplexF64, grid.N), grid)
```

### 3. Solve

```julia
params = SimParams(;
    medium  = medium,
    z_saves = 100,
    solver  = SSFM(1e-4),       # Fixed-step SSFM (required for vectorial)
    raman_model = nothing,       # Raman not yet supported in vectorial
)

vsol = solve(vpulse, params)
```

### 4. Access Results

[`VectorialSolution`](@ref) stores results as 3D arrays `(N, 2, z_saves)`:

```julia
# Time-domain fields
Ax = vsol.At[:, 1, end]     # x-polarization at fiber output
Ay = vsol.At[:, 2, end]     # y-polarization at fiber output

# Frequency-domain fields
AWx = vsol.AW[:, 1, end]
AWy = vsol.AW[:, 2, end]
```

## Walk-Off (Group Velocity Mismatch)

The ``\beta_1`` parameter in `TaylorDispersion` controls the group velocity of each axis. A difference in group velocity causes the two polarization components to walk off in time:

```julia
# X-axis: reference group velocity
disp_x = TaylorDispersion([-21.5e-27], 0.0)

# Y-axis: 100 fs/mm faster (walk-off = 1e-10 s/m)
disp_y = TaylorDispersion([-21.5e-27], 1.0e-10)
```

Over a length ``L``, the temporal walk-off is:
``\Delta t = |\beta_{1x} - \beta_{1y}| \times L``

## Physical Couplings

The vectorial solver includes all polarization-coupling mechanisms:

| Term | Physics |
|:---|:---|
| ``i \gamma \|A_x\|^2 A_x`` | Self-phase modulation (SPM) |
| ``i \frac{2\gamma}{3} \|A_y\|^2 A_x`` | Cross-phase modulation (XPM) |
| ``\frac{i\gamma}{3} A_y^2 A_x^* e^{-2i\Delta\beta_0 z}`` | Coherent FWM coupling |

The coherent FWM term transfers energy between axes and drives polarization rotation. For high birefringence (``\Delta\beta_0 L \gg 2\pi``), this term averages to zero and only SPM/XPM survive.

## Piping with Vectorial Pulses

The `|>` pipe operator works seamlessly for vectorial simulations too:

```julia
fiber_in  = SimParams(; medium=medium_in,  solver=SSFM(1e-4), raman_model=nothing)
fiber_out = SimParams(; medium=medium_out, solver=SSFM(1e-4), raman_model=nothing)

vsol = vpulse |> fiber_in |> fiber_out
```

!!! warning "Raman Not Yet Supported"
    Raman scattering is not yet implemented in the vectorial (birefringent) solver. Setting `raman_model` to anything other than `nothing` will issue a warning and fall back to Kerr-only nonlinearity.
