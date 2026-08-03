"""
Optics units & conversion utilities for Soliton.jl.
"""

"""
    dispersion_D_to_beta2(D_ps_nm_km::Real, lambda0_m::Real) -> Float64

Convert chromatic dispersion parameter D [ps/(nm·km)] at wavelength `lambda0_m` [m]
to group-velocity dispersion coefficient β₂ [s²/m].

Formula: β₂ = - λ₀² / (2π c) · D
"""
function dispersion_D_to_beta2(D_ps_nm_km::Real, lambda0_m::Real)
    D_si = D_ps_nm_km * 1e-6 # ps/(nm*km) = (1e-12 s) / (1e-9 m * 1000 m) = 1e-6 s/m²
    return - (lambda0_m^2 / (2π * c)) * D_si
end

"""
    beta2_to_dispersion_D(beta2_s2_m::Real, lambda0_m::Real) -> Float64

Convert group-velocity dispersion coefficient β₂ [s²/m] at wavelength `lambda0_m` [m]
to chromatic dispersion parameter D [ps/(nm·km)].
"""
function beta2_to_dispersion_D(beta2_s2_m::Real, lambda0_m::Real)
    D_si = - (2π * c / (lambda0_m^2)) * beta2_s2_m
    return D_si / 1e-6
end

"""
    dispersion_S_to_beta3(S_ps_nm2_km::Real, D_ps_nm_km::Real, lambda0_m::Real) -> Float64

Convert dispersion slope S [ps/(nm²·km)] and parameter D [ps/(nm·km)] at wavelength `lambda0_m` [m]
to third-order dispersion coefficient β₃ [s³/m].
"""
function dispersion_S_to_beta3(S_ps_nm2_km::Real, D_ps_nm_km::Real, lambda0_m::Real)
    S_si = S_ps_nm2_km * 1e3 # ps/(nm²*km) = 1e-12 / (1e-18 * 1000) = 1e3 s/m³
    D_si = D_ps_nm_km * 1e-6 # ps/(nm*km) = 1e-6 s/m²
    factor = (lambda0_m^2 / (2π * c))^2
    return factor * (S_si + (2.0 / lambda0_m) * D_si)
end

"""
    wavelength_to_frequency(lambda_m::Real) -> Float64

Convert optical wavelength λ [m] to optical frequency f [Hz].
"""
wavelength_to_frequency(lambda_m::Real) = c / lambda_m

"""
    frequency_to_wavelength(f_hz::Real) -> Float64

Convert optical frequency f [Hz] to optical wavelength λ [m].
"""
frequency_to_wavelength(f_hz::Real) = c / f_hz
