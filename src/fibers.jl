"""
Commercial Fiber Catalog and RefractiveIndex.io Glass Presets.

Provides pre-configured Sellmeier dispersion models for standard optical glasses
and commercial optical fiber specifications (Corning, NKT Photonics, Thorlabs).
"""

# ===========================================================================
# 1. Glass Refractive Index Presets (RefractiveIndex.io Models)
# ===========================================================================

"""
    FusedSilica() -> SellmeierDispersion

3-term Sellmeier model for pure fused silica (SiO₂) glass from Malitson (1965).
Valid over 0.21 μm to 3.71 μm.

Reference: I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965).
"""
function FusedSilica()
    # B coefficients (dimensionless)
    B = [0.6961663, 0.4079426, 0.8974794]
    # C coefficients (μm²)
    C = [0.0684043^2, 0.1162414^2, 9.896161^2]
    return SellmeierDispersion(B, C; microns=true)
end

"""
    SF6() -> SellmeierDispersion

3-term Sellmeier model for Schott SF6 heavy flint glass.

Reference: SCHOTT optical glass datasheet, SF6 (standard manufacturer Sellmeier
fit; see e.g. the SCHOTT glass catalog or refractiveindex.info, "SCHOTT-SF: SF6").
"""
function SF6()
    B = [1.72412305, 0.390104889, 1.04572858]
    C = [0.0134872362, 0.0569318095, 118.557185]
    return SellmeierDispersion(B, C; microns=true)
end

"""
    SF57() -> SellmeierDispersion

3-term Sellmeier model for Schott SF57 lead-silicate glass.

Reference: SCHOTT optical glass datasheet, SF57 (standard manufacturer Sellmeier
fit; see e.g. the SCHOTT glass catalog or refractiveindex.info, "SCHOTT-SF: SF57").
"""
function SF57()
    B = [1.81651371, 0.42889364, 1.07186278]
    C = [0.0143704198, 0.0592801172, 121.419942]
    return SellmeierDispersion(B, C; microns=true)
end

"""
    GeO2DopedSilica(weight_percent::Real) -> SellmeierDispersion

Sellmeier dispersion model for germania-doped silica (GeO₂-SiO₂) cores with
germania concentration `weight_percent` [0 to 15 %].

Reference: J. W. Fleming, Electron. Lett. 14, 326 (1978).
"""
function GeO2DopedSilica(x::Real)
    0 <= x <= 15 || throw(ArgumentError("GeO2 weight percent x must be between 0 and 15%"))
    # Interpolated Sellmeier coefficients based on Fleming (1978)
    B1 = 0.6961663 + 0.00100 * x
    B2 = 0.4079426 + 0.00315 * x
    B3 = 0.8974794 + 0.00286 * x

    C1 = 0.0684043^2 + 0.00008 * x
    C2 = 0.1162414^2 + 0.00021 * x
    C3 = 9.896161^2 + 0.01200 * x

    return SellmeierDispersion([B1, B2, B3], [C1, C2, C3]; microns=true)
end

# ===========================================================================
# 2. Commercial Fiber Specification Catalog
# ===========================================================================

"""
    FiberSpec

Data structure storing commercial fiber parameter specifications.
"""
struct FiberSpec
    name::String
    manufacturer::String
    description::String
    default_lambda0::Float64 # [m]
    gamma::Float64          # [1/(W·m)]
    default_loss::Float64   # [dB/m]
    dispersion::DispersionModel
end

"""
    FiberLibrary

Catalog dictionary mapping standard fiber key strings to [`FiberSpec`](@ref) objects.
Available keys:

  - `"Corning_SMF28"`
  - `"NKT_NL_PM_750"`
  - `"Thorlabs_PM780"`
  - `"Thorlabs_PM1550"`
  - `"NKT_LMA10"`

`FiberLibrary` is a plain `Dict{String, FiberSpec}` and can be extended at runtime:

```julia
JuGNLSE.FiberLibrary[\"MyCustomFiber\"] = FiberSpec(
    \"My Fiber\", \"Lab\", \"Custom PCF\", 800e-9, 0.05, 0.001, TaylorDispersion([-5e-27])
)
medium = commercial_fiber(\"MyCustomFiber\"; length=0.1)
```

!!! warning "Representative values, not verified datasheet copies"

    The γ, loss, and dispersion coefficients below are typical published
    values for each fiber type/wavelength (of the kind commonly used in
    simulation studies), not values transcribed from a specific dated
    manufacturer datasheet. Individual fiber spools/batches vary, and
    manufacturers revise specifications over time. For quantitative
    comparison against a real experiment, always confirm γ, loss, and
    dispersion against your fiber's current datasheet or a direct
    measurement rather than relying on these presets.
"""
const FiberLibrary = Dict{String, FiberSpec}(
    "Corning_SMF28" => FiberSpec(
        "Corning SMF-28e+",
        "Corning",
        "Standard single-mode telecom fiber for C-band (1550 nm)",
        1550e-9,
        0.00127, # γ ≈ 1.27 /W/km = 0.00127 /W/m
        0.0002,  # 0.2 dB/km = 0.0002 dB/m
        TaylorDispersion([-22.95e-27, 0.13e-39]), # β₂ = -22.95 ps²/km, β₃ = 0.13 ps³/km
    ),
    "NKT_NL_PM_750" => FiberSpec(
        "NKT Photonics NL-PM-750",
        "NKT Photonics",
        "Highly nonlinear PM photonic crystal fiber for supercontinuum generation",
        835e-9,
        0.11,   # γ = 110 /W/km = 0.11 /W/m
        0.001,  # 1 dB/km
        TaylorDispersion([
            -11.830e-27,
            8.1038e-41,
            -9.5205e-56,
            2.0737e-70,
            -5.3943e-85,
            1.3486e-99,
            -2.5495e-114,
            3.0524e-129,
            -1.7140e-144,
        ]),
    ),
    "Thorlabs_PM780" => FiberSpec(
        "Thorlabs PM780-HP",
        "Thorlabs",
        "Polarization maintaining single-mode fiber for 780 nm Ti:Sapphire delivery",
        780e-9,
        0.0055,
        0.003, # 3 dB/km
        TaylorDispersion([31.3e-27, 0.06e-39]), # β₂ = +31.3 ps²/km (normal dispersion)
    ),
    "Thorlabs_PM1550" => FiberSpec(
        "Thorlabs PM1550-HP",
        "Thorlabs",
        "Polarization maintaining single-mode fiber for 1550 nm",
        1550e-9,
        0.0012,
        0.0002,
        TaylorDispersion([-22.5e-27, 0.12e-39]),
    ),
    "NKT_LMA10" => FiberSpec(
        "NKT Photonics LMA-10",
        "NKT Photonics",
        "Large Mode Area photonic crystal fiber (Aeff ≈ 75 μm²)",
        1064e-9,
        0.0015,
        0.0005,
        TaylorDispersion([15.2e-27, 0.05e-39]),
    ),
)

"""
    commercial_fiber(name::String; length::Real, lambda0::Union{Real,Nothing}=nothing, loss::Union{Real,Nothing}=nothing) -> Medium

Create a [`Medium`](@ref) from the commercial fiber catalog.

# Arguments

  - `name::String`: Key name in [`FiberLibrary`](@ref) (e.g. `"Corning_SMF28"`, `"NKT_NL_PM_750"`).
  - `length::Real`: Propagation length [m].
  - `lambda0`: Center wavelength [m] (defaults to fiber spec default).
  - `loss`: Fiber attenuation [dB/m] (defaults to fiber spec default).

# Example

```julia
medium = commercial_fiber(\"Corning_SMF28\"; length=100.0)
```
"""
function commercial_fiber(
    name::String;
    length::Real,
    lambda0::Union{Real, Nothing}=nothing,
    loss::Union{Real, Nothing}=nothing,
)
    haskey(FiberLibrary, name) ||
        throw(ArgumentError("Unknown fiber spec '$name'. Available: $(keys(FiberLibrary))"))
    spec = FiberLibrary[name]
    wl = lambda0 === nothing ? spec.default_lambda0 : Float64(lambda0)
    alpha = loss === nothing ? spec.default_loss : Float64(loss)
    return Medium(length, spec.gamma, alpha, spec.dispersion, wl)
end

# ===========================================================================
# 3. Gas Refractive Index & Hollow-Core Fiber (HC-PCF) Guidance Models
# ===========================================================================

"""
    gas_refractive_index(gas::Symbol, lambda_m::Real, pressure_bar::Real; temperature_K::Real=293.15) -> Float64

Refractive index of gas at wavelength `lambda_m` [m], pressure `pressure_bar` [bar], and temperature `temperature_K` [K].
Supported gases: `:Ar`, `:Ne`, `:Kr`, `:Xe`, `:H2`, `:N2`, `:Air`.

Reference: Börzsönyi et al., Opt. Express 21, 21086 (2013); Peck & Khanna, J. Opt. Soc. Am. 67, 1550 (1977).
"""
function gas_refractive_index(
    gas::Symbol, lambda_m::Real, pressure_bar::Real; temperature_K::Real=293.15
)
    l_um = lambda_m * 1e6 # [μm]
    inv_l2 = 1.0 / (l_um^2)

    # (n_STP - 1) * 1e8 at 1 bar, 273.15 K
    n_minus_1_1e8 = if gas === :Ar
        5547.61 + 515.39 / (1.0 - 0.003029 * inv_l2)
    elseif gas === :Ne
        6151.7 + 382.4 / (1.0 - 0.0035 * inv_l2)
    elseif gas === :Kr
        2610.5 + 26116.3 / (1.0 - 0.00494 * inv_l2)
    elseif gas === :Xe
        10657.0 + 35147.0 / (1.0 - 0.0084 * inv_l2)
    elseif gas === :H2
        1358.0 + 7799.0 / (1.0 - 0.0041 * inv_l2)
    elseif gas === :N2 || gas === :Air
        6498.2 + 29398.0 / (1.0 - 0.0073 * inv_l2)
    else
        throw(
            ArgumentError(
                "Unsupported gas '$gas'. Supported: :Ar, :Ne, :Kr, :Xe, :H2, :N2, :Air"
            ),
        )
    end

    n_stp_minus_1 = n_minus_1_1e8 * 1e-8
    # Scaling with pressure and temperature: (P / 1.0) * (273.15 / T)
    n_gas = 1.0 + n_stp_minus_1 * pressure_bar * (273.15 / temperature_K)
    return n_gas
end

"""
    gas_n2(gas::Symbol, pressure_bar::Real) -> Float64

Nonlinear index n₂ [m²/W] of gas at pressure `pressure_bar` [bar] (STP values scaled linearly with pressure).

Reference (order-of-magnitude STP values): J. K. Wahlstrand, Y.-H. Cheng &
H. M. Milchberg, Phys. Rev. A 85, 043820 (2012) for noble gases; typical
literature values for H₂/N₂ (e.g. compiled in Börzsönyi et al., Opt. Express
18, 25847 (2010)). These are representative constants, not a specific-source
tabulation — for quantitative work, verify against a dataset for your exact
wavelength and pressure regime.
"""
function gas_n2(gas::Symbol, pressure_bar::Real)
    n2_stp = if gas === :Ar
        1.0e-23
    elseif gas === :Ne
        0.8e-24
    elseif gas === :Kr
        3.4e-23
    elseif gas === :Xe
        1.0e-22
    elseif gas === :H2
        1.2e-23
    elseif gas === :N2 || gas === :Air
        2.5e-23
    else
        1.0e-23
    end
    return n2_stp * pressure_bar
end

"""
    _capillary_confinement_loss_dB_per_m(w, gas, pressure, temperature, radius) -> Float64

Marcatili-Schmeltzer capillary confinement loss for the fundamental HE₁₁ mode of a
hollow dielectric (silica-clad) waveguide, evaluated at absolute angular frequency `w`:

    α(λ) = (u₀₁/2π)² · λ²/a³ · (ν²+1)/√(ν²-1)

where `a` is the core radius, and `ν = n_clad(λ)/n_gas(λ,P)` is the ratio of the silica
cladding index (Malitson fused-silica Sellmeier fit, [`FusedSilica`](@ref)) to the
gas core index. Returns the loss in [dB/m] (the natural-log/Np result of the formula
above is converted via `10/ln(10)`), for direct use as a `Medium.loss` function.

Reference: E. A. J. Marcatili & R. A. Schmeltzer, Bell Syst. Tech. J. 43, 1783 (1964).
"""
function _capillary_confinement_loss_dB_per_m(
    w::Real, gas::Symbol, pressure::Real, temperature::Real, radius::Real
)
    u01 = 2.4048255577
    lam = 2π * c / w
    ngas = gas_refractive_index(gas, lam, pressure; temperature_K=temperature)

    silica = FusedSilica()
    n2_clad = 1.0
    for i in eachindex(silica.B)
        n2_clad += silica.B[i] * lam^2 / (lam^2 - silica.C[i])
    end
    nclad = sqrt(n2_clad)

    nu = nclad / ngas
    alpha_Np = (u01 / (2π))^2 * lam^2 / radius^3 * (nu^2 + 1.0) / sqrt(nu^2 - 1.0)  # [Np/m]
    return alpha_Np * (10.0 / log(10.0))  # [dB/m]
end

"""
    HollowCoreFiber(; radius, gas, pressure, length, lambda0, loss=0.0, temperature=293.15, grid=nothing) -> Medium

Construct a [`Medium`](@ref) for a gas-filled Hollow-Core Photonic Crystal Fiber (HC-PCF).

Calculates pressure-dependent dispersion using the Marcatili-Schmeltzer / Zeisberger capillary
anti-resonance guidance model:

    β(ω, P) = (ω/c) · √(n_gas²(ω, P) - (u₀₁ c / (ω R_core))²)

where u₀₁ ≈ 2.40483 is the fundamental HE₁₁ mode Bessel root, and R_core is the hollow core radius [m].

The wavelength-dependent Marcatili-Schmeltzer capillary confinement loss can
optionally be added to the returned `Medium`'s `loss` field via
`confinement_loss=true`; `loss` below always adds any extra
(surface-scattering, bend, splice) attenuation on top of it.

!!! warning "Confinement loss magnitude"

    This is the bare single-wall thick-capillary formula — it does *not* include
    the anti-resonant wall-thickness transmission-window term that real
    negative-curvature HC-PCF designs use to suppress loss by orders of
    magnitude. At small core radii (tens of μm) it can predict loss far higher
    than measured in modern HC-PCF (e.g. hundreds of dB/m rather than the
    dB/m-to-dB/km typical of real fibers), so it is opt-in rather than the
    default and should be treated as a conservative (worst-case) capillary
    bound, not a quantitative prediction for anti-resonant/Kagome designs.

# Arguments

  - `radius::Real`: Core radius R_core [m] (e.g. 15e-6 for 30 μm core diameter)
  - `gas::Symbol`: Gas species (`:Ar`, `:Ne`, `:Kr`, `:Xe`, `:H2`, `:N2`)
  - `pressure::Real`: Gas pressure P [bar]
  - `length::Real`: Propagation length [m]
  - `lambda0::Real`: Central wavelength [m]
  - `loss::Real`: Extra fiber attenuation beyond confinement loss [dB/m] (default 0.0)
  - `confinement_loss::Bool`: Add the Marcatili-Schmeltzer capillary confinement
    loss on top of `loss` (default `false`; see warning above)
  - `temperature::Real`: Temperature T [K] (default 293.15 K)
  - `grid::Union{Grid, Nothing}`: Optional simulation grid. If provided, constructs exact [`TabulatedDispersion`](@ref) over the full frequency span.

# Example

```julia
# 30 μm core HC-PCF filled with 5 bar Argon at 800 nm
hcf = HollowCoreFiber(; radius=15e-6, gas=:Ar, pressure=5.0, length=0.5, lambda0=800e-9)
```
"""
function HollowCoreFiber(;
    radius::Real,
    gas::Symbol,
    pressure::Real,
    length::Real,
    lambda0::Real,
    loss::Real=0.0,
    confinement_loss::Bool=false,
    temperature::Real=293.15,
    grid::Union{Grid, Nothing}=nothing,
)
    radius > 0 || throw(ArgumentError("Core radius must be positive"))
    pressure >= 0 || throw(ArgumentError("Pressure must be non-negative"))

    u01 = 2.4048255577
    w0 = 2π * c / lambda0
    dw = 1e-4 * w0

    function beta_at_w(w)
        lam = 2π * c / w
        ngas = gas_refractive_index(gas, lam, pressure; temperature_K=temperature)
        k0 = w / c
        arg = ngas^2 - (u01 / (k0 * radius))^2
        arg_clamped = max(1e-10, arg)
        return k0 * sqrt(arg_clamped)
    end

    b0 = beta_at_w(w0)
    bp = (beta_at_w(w0 + dw) - beta_at_w(w0 - dw)) / (2dw)

    disp = if grid !== nothing
        # Exact continuous tabulated dispersion over the simulation grid
        V_detuning = grid.V
        B_vals = zeros(Float64, Base.length(V_detuning))
        for k in eachindex(V_detuning)
            wk = w0 + V_detuning[k]
            if wk > 0
                B_vals[k] = beta_at_w(wk) - b0 - bp * V_detuning[k]
            end
        end
        TabulatedDispersion(V_detuning, B_vals)
    else
        b2 = (beta_at_w(w0 + dw) - 2b0 + beta_at_w(w0 - dw)) / (dw^2)
        b3 =
            (
                beta_at_w(w0 + 2dw) - 2beta_at_w(w0 + dw) + 2beta_at_w(w0 - dw) -
                beta_at_w(w0 - 2dw)
            ) / (2dw^3)
        b4 =
            (
                beta_at_w(w0 + 2dw) - 4beta_at_w(w0 + dw) + 6b0 - 4beta_at_w(w0 - dw) +
                beta_at_w(w0 - 2dw)
            ) / (dw^4)
        TaylorDispersion([b2, b3, b4])
    end

    Aeff = 0.84 * π * radius^2
    n2_val = gas_n2(gas, pressure)
    gamma_val = 2π * n2_val / (lambda0 * Aeff)

    loss_final = if confinement_loss
        extra_loss = loss
        w ->
            extra_loss +
            _capillary_confinement_loss_dB_per_m(w, gas, pressure, temperature, radius)
    else
        loss
    end

    return Medium(length, gamma_val, loss_final, disp, lambda0)
end
