# GNLSE Copilot Instructions

High-performance Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE) for ultrafast optical pulse propagation in fibers.

## Core Architecture

**Split-Step Method in Interaction Picture**: The solver alternates between dispersion (frequency domain) and nonlinearity (time domain):

- Linear operator `D̂(ω)` handles dispersion and loss (frequency domain)
- Nonlinear operator `N̂[A]` includes Kerr, Raman, and shock terms (time domain)
- Adaptive steppers use embedded error estimation for automatic step size control

**Key Data Flow**:

```julia
Pulse (At, Aw) → PhysicsModel (operators, FFT plans) → Adaptive stepper → Results (z, At, Aw)
```

## Critical Julia Performance Rules

**Type Stability (Non-Negotiable)**:

- Return types must be inferrable from input types, not values
- Use `zero(x)` not `0`, `one(x)` not `1` for generic code
- Check with `@code_warntype` before committing - no red types allowed in hot paths

**Zero Allocations in Solvers**:

- Pre-allocate all arrays in `PhysicsModel` and `StepperState` structs
- Use `@.` macro for in-place vectorized operations: `@. A = A + B * C`
- Verify with `@btime` - target is 0 allocations per timestep

**FFT Efficiency**:

- Always use pre-planned FFTs stored in `PhysicsModel` (see [src/solvers/erk4ip.jl#L120-150](src/solvers/erk4ip.jl))
- Call `mul!(output, fft_plan, input)` NOT `fft(input)` in loops
- Grid size must be power of 2 for FFTW optimization

## Sign Conventions (Physics Critical)

**Always reference `research/GNLSE_Julia_Implementation_Spec.md` for complete details:**

- Dispersion operator: `D̂(ω) = -α/2 + i∑(βₙ/n!)(iω)ⁿ`
- β₂ < 0 = anomalous dispersion (solitons), β₂ > 0 = normal dispersion
- Beta array indexing: First element is β₂, second is β₃ (no β₀ or β₁)
- FFT convention: Forward transform time→frequency, frequency in rad/s centered at zero

## Testing Workflow

**Always run comparison tests** - physics accuracy is non-negotiable:

```powershell
julia --project=. -e "using Pkg; Pkg.test()"
```

Comparison tests in `test/comparison_tests/` validate against published papers. Acceptable error: < 1% vs reference. Never skip these after modifying solvers or operators.

## Common Patterns

**Creating simulations** (see [README.md](README.md#L56-92)):
README.md for current API):

```julia
grid = create_grid(2^12, 10e-12, 835e-9)  # Power of 2, time window, wavelength
medium = Medium(length, gamma, [β₂, β₃], alpha, λ₀)
pulse = sech_pulse(grid, duration, power, wavelength)
params = SimParams(medium=medium, n_saves=200, raman=true, shock=true)
results = solve(pulse, params, rtol=1e-6)
```

**Adding physics effects**: Modify nonlinearity computation. Follow existing patterns: compute in time domain, use pre-allocated buffers, return nothing for in-place operations.

**New Raman models**: Implement via abstract base type. Must provide frequency-domain response for efficient convolu

## Module Organization Principles

- **Core types**: Immutable structs for `Medium`, `Grid`, `Pulse`, `SimParams` - never add mutable fields to physics types
- **Grid generation**: Time-frequency grids use FFT-compatible conventions (power of 2 sizes)
- **Pulse generation**: Initial pulse shapes (sech, Gaussian, CW) with configurable duration definitions
- **Solver interface**: User-facing `solve()` dispatches to appropriate solver implementations
- **Solver internals**: Pre-computed operators and FFT plans, propagation state buffers separate from physics model

## Documentation Standards

Every exported function needs docstrings with:

- Physics context (equation, reference to spec docs)
- Units in brackets: `[m]`, `[W]`, `[s]`
- Example with realistic parameters from examples/
- Cross-reference to spec: "See Section X in research/GNLSE_Julia_Implementation_Spec.md"

## Before Committing

1. **Type check**: `@code_warntype` on modified functions - no red types
2. **Benchmark**: `@btime` shows zero allocations in solver hot paths
3. **Test physics**: Run Dudley comparison test, must pass < 1% error
4. **Check dependencies**: Only FFTW, LinearAlgebra allowed in src/ (Plots, Test in test/ only)

## Key References

- **Physics specification**: `research/GNLSE_Julia_Implementation_Spec.md` - Complete mathematical foundation
- **Performance rules**: `research/Julia_Optimization_Guidelines.md` - Julia-specific optimization patterns
- **Reference implementations**: `research/reference_implementations/` - Working code from established packages
- **Detailed instructions**: `docs/development/COPILOT_INSTRUCTIONS.md` - Comprehensive development guide
