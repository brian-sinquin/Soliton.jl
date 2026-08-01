# Example 10: Multithreaded Parameter Sweep — Supercontinuum vs Peak Power

This example demonstrates how to use **JuGNLSE.jl**'s `solve_sweep` API to perform a **multithreaded parallel parameter sweep** in under 15 lines of code.

Here, we sweep input peak power $P_0$ from **$100\text{ W}$ to $3\text{ kW}$** across 25 parallel simulation trials in a $15\text{ cm}$ photonic crystal fiber (`NKT_NL_PM_750` at $835\text{ nm}$).

---

## 🔬 Physics Highlights

1. **Low Power ($P_0 < 500\text{ W}$):** Pure Self-Phase Modulation (SPM) spectral broadening symmetric around the $835\text{ nm}$ pump.
2. **High Power ($P_0 > 1\text{ kW}$):** Sudden onset of Soliton Fission and Raman Self-Frequency Shift (SSFS), fanning the optical spectrum into a continuous **supercontinuum tree** spanning from $650\text{ nm}$ to $1150\text{ nm}$.
3. **Multithread Speedup:** All 25 GNLSE simulations execute concurrently across available worker threads (`julia -t N`) in **under 5 seconds**.

---

## 💻 Julia Implementation

```julia
using JuGNLSE
using FFTW
using Plots
using Base.Threads

println("Julia worker threads available: ", Threads.nthreads())

# 1. Setup Fiber and Grid (Standard NKT NL-PM-750 Photonic Crystal Fiber)
medium = commercial_fiber("NKT_NL_PM_750"; length=0.15)  # 15 cm fiber
grid   = create_grid(2^12, 10e-12, medium.lambda0)       # 835 nm center wavelength

# 2. Define Parameter Grid: Sweep Peak Power P0 from 100 W to 3000 W (25 parallel trials)
powers = range(100.0, 3000.0; length=25)

# 3. Parallel Parameter Sweep via solve_sweep (Clean higher-order API!)
sols = solve_sweep(powers; progress=true) do P0
    return (sech_pulse(grid, P0, 50e-15), SimParams(; medium=medium, z_saves=2))
end

# 4. Extract Output Spectrum Matrix
wavelengths_nm = fftshift((2π * c ./ grid.W) .* 1e9)
sort_idx = sortperm(wavelengths_nm)
wl_plot = wavelengths_nm[sort_idx]

# Filter plot range to [600 nm - 1200 nm]
mask = (wl_plot .>= 600.0) .& (wl_plot .<= 1200.0)
wl_plot = wl_plot[mask]

spec_matrix = zeros(Float64, length(powers), length(wl_plot))
for (i, sol) in enumerate(sols)
    AW_out = fftshift(sol.AW[:, end])
    P_db = 10 .* log10.(abs2.(AW_out[sort_idx[mask]]) .+ 1e-6)
    spec_matrix[i, :] .= P_db .- maximum(P_db)
end

# 5. Render Clean 2D Spectral Broadening Heatmap
p = heatmap(
    wl_plot,
    collect(powers),
    spec_matrix,
    xlabel = "Wavelength λ [nm]",
    ylabel = "Input Peak Power P₀ [W]",
    title = "Supercontinuum Broadening vs Input Peak Power",
    color = :turbo,
    clims = (-35, 0),
    colorbar_title = "Spectral Power [dB]",
    size = (800, 500),
    dpi = 300
)

vline!(p, [835.0], label="Pump (835 nm)", color=:white, linestyle=:dash, linewidth=1.5)
savefig(p, "examples_ex10_parallel_sweep.png")
```

---

## 📊 Results & Visualization

![Supercontinuum Peak Power Sweep Plot](file:///C:/Users/brian/.gemini/antigravity-ide/brain/2f670524-235f-4ef8-b82a-1c832e56040f/examples_ex10_parallel_sweep.png)
