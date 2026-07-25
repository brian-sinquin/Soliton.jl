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
"""
function SF6()
    B = [1.72412305, 0.390104889, 1.04572858]
    C = [0.0134872362, 0.0569318095, 118.557185]
    return SellmeierDispersion(B, C; microns=true)
end

"""
    SF57() -> SellmeierDispersion

3-term Sellmeier model for Schott SF57 lead-silicate glass.
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
    C3 = 9.896161^2  + 0.01200 * x

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
"""
const FiberLibrary = Dict{String, FiberSpec}(
    "Corning_SMF28" => FiberSpec(
        "Corning SMF-28e+",
        "Corning",
        "Standard single-mode telecom fiber for C-band (1550 nm)",
        1550e-9,
        0.00127, # γ ≈ 1.27 /W/km = 0.00127 /W/m
        0.0002,  # 0.2 dB/km = 0.0002 dB/m
        TaylorDispersion([-22.95e-27, 0.13e-39]) # β₂ = -22.95 ps²/km, β₃ = 0.13 ps³/km
    ),
    "NKT_NL_PM_750" => FiberSpec(
        "NKT Photonics NL-PM-750",
        "NKT Photonics",
        "Highly nonlinear PM photonic crystal fiber for supercontinuum generation",
        835e-9,
        0.11,   # γ = 110 /W/km = 0.11 /W/m
        0.001,  # 1 dB/km
        TaylorDispersion([
            -11.830e-27, 8.1038e-41, -9.5205e-56, 2.0737e-70,
            -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129,
            -1.7140e-144
        ])
    ),
    "Thorlabs_PM780" => FiberSpec(
        "Thorlabs PM780-HP",
        "Thorlabs",
        "Polarization maintaining single-mode fiber for 780 nm Ti:Sapphire delivery",
        780e-9,
        0.0055,
        0.003, # 3 dB/km
        TaylorDispersion([31.3e-27, 0.06e-39]) # β₂ = +31.3 ps²/km (normal dispersion)
    ),
    "Thorlabs_PM1550" => FiberSpec(
        "Thorlabs PM1550-HP",
        "Thorlabs",
        "Polarization maintaining single-mode fiber for 1550 nm",
        1550e-9,
        0.0012,
        0.0002,
        TaylorDispersion([-22.5e-27, 0.12e-39])
    ),
    "NKT_LMA10" => FiberSpec(
        "NKT Photonics LMA-10",
        "NKT Photonics",
        "Large Mode Area photonic crystal fiber (Aeff ≈ 75 μm²)",
        1064e-9,
        0.0015,
        0.0005,
        TaylorDispersion([15.2e-27, 0.05e-39])
    )
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
medium = commercial_fiber("Corning_SMF28", length=100.0)
```
"""
function commercial_fiber(
    name::String;
    length::Real,
    lambda0::Union{Real, Nothing}=nothing,
    loss::Union{Real, Nothing}=nothing
)
    haskey(FiberLibrary, name) || throw(ArgumentError("Unknown fiber spec '$name'. Available: $(keys(FiberLibrary))"))
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
function gas_refractive_index(gas::Symbol, lambda_m::Real, pressure_bar::Real; temperature_K::Real=293.15)
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
        throw(ArgumentError("Unsupported gas '$gas'. Supported: :Ar, :Ne, :Kr, :Xe, :H2, :N2, :Air"))
    end

    n_stp_minus_1 = n_minus_1_1e8 * 1e-8
    # Scaling with pressure and temperature: (P / 1.0) * (273.15 / T)
    n_gas = 1.0 + n_stp_minus_1 * pressure_bar * (273.15 / temperature_K)
    return n_gas
end

"""
    gas_n2(gas::Symbol, pressure_bar::Real) -> Float64

Nonlinear index n₂ [m²/W] of gas at pressure `pressure_bar` [bar] (STP values scaled linearly with pressure).
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
    HollowCoreFiber(; radius, gas, pressure, length, lambda0, loss=0.0, temperature=293.15) -> Medium

Construct a [`Medium`](@ref) for a gas-filled Hollow-Core Photonic Crystal Fiber (HC-PCF).

Calculates pressure-dependent dispersion using the Marcatili-Schmeltzer / Zeisberger capillary
anti-resonance guidance model:

    β(ω, P) = (ω/c) · √(n_gas²(ω, P) - (u₀₁ c / (ω R_core))²)

where u₀₁ ≈ 2.40483 is the fundamental HE₁₁ mode Bessel root, and R_core is the hollow core radius [m].

# Arguments
- `radius::Real`: Core radius R_core [m] (e.g. 15e-6 for 30 μm core diameter)
- `gas::Symbol`: Gas species (`:Ar`, `:Ne`, `:Kr`, `:Xe`, `:H2`, `:N2`)
- `pressure::Real`: Gas pressure P [bar]
- `length::Real`: Propagation length [m]
- `lambda0::Real`: Central wavelength [m]
- `loss::Real`: Fiber attenuation [dB/m] (default 0.0)
- `temperature::Real`: Temperature T [K] (default 293.15 K)

# Example
```julia
# 30 μm core HC-PCF filled with 5 bar Argon at 800 nm
hcf = HollowCoreFiber(radius=15e-6, gas=:Ar, pressure=5.0, length=0.5, lambda0=800e-9)
```
"""
function HollowCoreFiber(;
    radius::Real,
    gas::Symbol,
    pressure::Real,
    length::Real,
    lambda0::Real,
    loss::Real=0.0,
    temperature::Real=293.15
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
    b2 = (beta_at_w(w0 + dw) - 2b0 + beta_at_w(w0 - dw)) / (dw^2)
    b3 = (beta_at_w(w0 + 2dw) - 2beta_at_w(w0 + dw) + 2beta_at_w(w0 - dw) - beta_at_w(w0 - 2dw)) / (2dw^3)
    b4 = (beta_at_w(w0 + 2dw) - 4beta_at_w(w0 + dw) + 6b0 - 4beta_at_w(w0 - dw) + beta_at_w(w0 - 2dw)) / (dw^4)

    disp = TaylorDispersion([b2, b3, b4])

    Aeff = 0.84 * π * radius^2
    n2_val = gas_n2(gas, pressure)
    gamma_val = 2π * n2_val / (lambda0 * Aeff)

    return Medium(length, gamma_val, loss, disp, lambda0)
end
