```@meta
CurrentModule = JuGNLSE
```

# Example 5: Higher-Order Soliton Compression

**Reproducing the periodic breathing and temporal compression of N=3 solitons
(Mollenauer, Stolen & Gordon, 1980)**

DOI: [10.1103/PhysRevLett.45.1095](https://doi.org/10.1103/PhysRevLett.45.1095)

---

## Physical Background

When a sech² pulse with N > 1 is launched in the anomalous-dispersion regime, it forms a
**higher-order soliton** that undergoes periodic compression. At each integer multiple of the
soliton half-period ``z_{1/2} = \frac{\pi}{2} L_D``, the pulse returns to its original shape.
At intermediate distances it compresses dramatically, reaching a minimum duration near:

```math
z_{\min} \approx 0.32 \frac{L_D}{N}
```

with a compression ratio approximately:

```math
F_c \approx 4.1 N
```

The N=3 soliton was the first experimentally observed higher-order soliton in an optical fiber.

## Simulation

```julia
using JuGNLSE

# ─── Standard silica fiber at 1.06 µm (Mollenauer 1980 experimental regime) ─
lambda0 = 1060e-9            # [m]
beta2   = -9.0e-27           # [s²/m] anomalous dispersion
gamma   = 0.0019             # [1/(W·m)]

# ─── N=3 soliton ─────────────────────────────────────────────────────────────
N  = 3
T0 = 4e-12              # 4 ps half-width (≈ 7 ps FWHM)
P0 = N^2 * abs(beta2) / (gamma * T0^2)   # N=3 soliton peak power [W]
LD = T0^2 / abs(beta2)

FWHM = 2 * log(1 + sqrt(2)) * T0
Zhalf = (π/2) * LD      # soliton half-period

println("N=3 Soliton:")
println("  Peak power P₀ = $(round(P0; sigdigits=3)) W")
println("  Dispersion length L_D = $(round(LD; sigdigits=3)) m")
println("  Half-period z₁/₂ = $(round(Zhalf; sigdigits=3)) m")
println("  Expected compression ratio ≈ $(4.1 * N)x")
println("  Expected compression point ≈ $(round(0.32 * LD / N; sigdigits=2)) m")

# ─── Grid and medium ─────────────────────────────────────────────────────────
grid   = create_grid(2^12, 60e-12, lambda0)
medium = Medium(;
    length  = Zhalf,         # propagate one half-period (full breathing cycle)
    gamma   = gamma,
    loss    = 0.0,
    betas   = [beta2],
    lambda0 = lambda0,
)

pulse  = sech_pulse(grid, P0, FWHM)

# ─── Solve: no Raman (clean higher-order soliton) ────────────────────────────
params = SimParams(;
    medium          = medium,
    z_saves         = 500,        # dense z-sampling to capture the narrow minimum
    raman_model     = nothing,
    self_steepening = false,
)
sol = solve(pulse, params)

# ─── Find compression ratio and position ─────────────────────────────────────
# Peak power at each saved position
peaks = [maximum(abs2, sol.At[:, i]) for i in axes(sol.At, 2)]
Pmax  = maximum(peaks)
z_max = sol.Z[argmax(peaks)]
compression_ratio = Pmax / P0

println("\nSimulation results:")
println("  Maximum peak power = $(round(Pmax; sigdigits=3)) W")
println("  Compression ratio  = $(round(compression_ratio; sigdigits=3))x")
println("  At z              = $(round(z_max; sigdigits=3)) m")

# ─── N=1 reference: stable (no compression) ───────────────────────────────────
P0_N1 = abs(beta2) / (gamma * T0^2)
pulse_N1 = sech_pulse(grid, P0_N1, FWHM)
sol_N1 = solve(pulse_N1, SimParams(; medium=medium, z_saves=100,
                                    raman_model=nothing, self_steepening=false))

peaks_N1 = [maximum(abs2, sol_N1.At[:, i]) for i in axes(sol_N1.At, 2)]
println("\nN=1 soliton (reference): peak power variation = ",
    round((maximum(peaks_N1) - minimum(peaks_N1)) / P0_N1 * 100; sigdigits=2), " %  (expect ≈ 0)")
```

## Expected Results

| Quantity | Analytical | Simulation |
|:---|:---|:---|
| Compression ratio | ≈ 12× (N=3) | ≈ 11–13× |
| Compression point | ≈ 0.107 × L_D | ≈ 0.10–0.12 × L_D |
| Half-period | π/2 × L_D | restored at z = π/2 × L_D |

## Higher-Order Soliton Gallery

Run all soliton orders 1–4 and compare:

```julia
for N in 1:4
    P0_N = N^2 * abs(beta2) / (gamma * T0^2)
    pulse_N = sech_pulse(grid, P0_N, FWHM)
    sol_N   = solve(pulse_N, SimParams(; medium=medium, z_saves=200,
                                        raman_model=nothing, self_steepening=false))
    peaks_N = [maximum(abs2, sol_N.At[:, i]) for i in axes(sol_N.At, 2)]
    Fc = maximum(peaks_N) / P0_N
    println("N=$N: max compression = $(round(Fc; sigdigits=3))x  (theory ≈ $(4.1*N)x)")
end
```

## Soliton Fission vs. Ideal Compression

In practice, the clean higher-order soliton evolution is disrupted by:
- **Third-order dispersion β₃**: breaks the periodicity
- **Raman scattering**: shifts individual solitons in frequency (see Example 2)
- **Self-steepening**: distorts the pulse shape

Enable these to observe realistic (imperfect) soliton dynamics:

```julia
medium_realistic = Medium(;
    length  = Zhalf,
    gamma   = gamma,
    loss    = 0.0,
    betas   = [beta2, 5e-41],    # add β₃
    lambda0 = lambda0,
)
sol_real = solve(pulse, SimParams(;
    medium          = medium_realistic,
    z_saves         = 500,
    raman_model     = BlowWood(),
    self_steepening = true,
))
```

## References

> L. F. Mollenauer, R. H. Stolen, and J. P. Gordon, "Experimental observation of picosecond
> pulse narrowing and solitons in optical fibers,"
> *Phys. Rev. Lett.* **45**, 1095–1098 (1980).
> DOI: [10.1103/PhysRevLett.45.1095](https://doi.org/10.1103/PhysRevLett.45.1095)

> G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press (2019).
> Chapter 5: "Optical Solitons"
