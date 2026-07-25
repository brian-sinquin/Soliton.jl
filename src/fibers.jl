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
