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

Convert optical wavelength λ [m] to optical (ordinary) frequency f [Hz].

See also [`wavelength_to_omega`](@ref) for the angular-frequency form used
internally by [`Grid`](@ref)/[`Medium`](@ref) (`ω = 2πf`).
"""
wavelength_to_frequency(lambda_m::Real) = c / lambda_m

"""
    frequency_to_wavelength(f_hz::Real) -> Float64

Convert optical (ordinary) frequency f [Hz] to optical wavelength λ [m].
"""
frequency_to_wavelength(f_hz::Real) = c / f_hz

"""
    wavelength_to_omega(lambda_m::Real) -> Float64

Convert optical wavelength λ [m] to angular frequency ω = 2πc/λ [rad/s], the
convention used throughout the package for [`Grid`](@ref) (`omega0`, `W`, `V`)
and dispersion/nonlinearity models. Equivalent to
`2π * wavelength_to_frequency(lambda_m)`.
"""
wavelength_to_omega(lambda_m::Real) = 2π * c / lambda_m

"""
    omega_to_wavelength(omega::Real) -> Float64

Convert angular frequency ω [rad/s] to wavelength λ = 2πc/ω [m]. Inverse of
[`wavelength_to_omega`](@ref); equivalent to [`wavelength_grid`](@ref) applied
to a single frequency.
"""
omega_to_wavelength(omega::Real) = 2π * c / omega

# ---------------------------------------------------------------------------
# Decibel / power-ratio conversions
# ---------------------------------------------------------------------------
#
# The package works with three distinct "dB-like" quantities internally
# (amplifier/attenuator field gain in `elements.jl`, power-ratio gain in
# `AmplifyingMedium`'s `g0_db`, and the dB/m -> Np/m loss/gain factor
# `log(10)/10` in `dispersion.jl`'s `_eval_loss_or_gain!`). These functions
# give each convention an explicit, testable name so callers (and future
# internal refactors) don't have to re-derive `10^(x/10)` vs `10^(x/20)` by
# hand.

"""
    db_to_linear_power(db::Real) -> Float64

Convert a power ratio in decibels to a linear power ratio: `10^(dB/10)`. Use
for quantities defined as *power* ratios (optical power gain/loss, RIN
power spectral density as in [`rin_rms`](@ref)).

See also [`db_to_linear_amplitude`](@ref) for *field-amplitude* ratios
(a factor of 2 in the exponent's denominator, since power ∝ amplitude²).
"""
db_to_linear_power(db::Real) = 10.0^(db / 10.0)

"""
    linear_power_to_db(ratio::Real) -> Float64

Convert a linear power ratio to decibels: `10·log10(ratio)`. Inverse of
[`db_to_linear_power`](@ref). `ratio` must be positive.
"""
function linear_power_to_db(ratio::Real)
    ratio > 0 || throw(ArgumentError("power ratio must be positive"))
    return 10.0 * log10(ratio)
end

"""
    db_to_linear_amplitude(db::Real) -> Float64

Convert a field-amplitude ratio in decibels to a linear amplitude ratio:
`10^(dB/20)`. Use for quantities defined as *amplitude* ratios (e.g. the
field-gain convention used by [`Amplifier`](@ref)/[`Attenuator`](@ref)).

See also [`db_to_linear_power`](@ref) for *power* ratios.
"""
db_to_linear_amplitude(db::Real) = 10.0^(db / 20.0)

"""
    linear_amplitude_to_db(ratio::Real) -> Float64

Convert a linear field-amplitude ratio to decibels: `20·log10(ratio)`. Inverse
of [`db_to_linear_amplitude`](@ref). `ratio` must be positive.
"""
function linear_amplitude_to_db(ratio::Real)
    ratio > 0 || throw(ArgumentError("amplitude ratio must be positive"))
    return 20.0 * log10(ratio)
end

"""
    db_to_np(db::Real) -> Float64

Convert a power-ratio decibel value to Nepers: `dB · ln(10)/10`. This is the
dB/m → Np/m conversion the package applies internally to `Medium.loss` and
gain fields (a fiber specified as α dB/m has field-amplitude attenuation
`exp(-α_Np·z/2)` and power attenuation `exp(-α_Np·z)`).
"""
db_to_np(db::Real) = db * (log(10.0) / 10.0)

"""
    np_to_db(np::Real) -> Float64

Convert Nepers to a power-ratio decibel value: `Np · 10/ln(10)`. Inverse of
[`db_to_np`](@ref).
"""
np_to_db(np::Real) = np * (10.0 / log(10.0))

"""
    dbm_to_watt(p_dbm::Real) -> Float64

Convert optical power in dBm (decibels relative to 1 mW) to watts:
`P[W] = 1mW · 10^(P_dBm/10)`.
"""
dbm_to_watt(p_dbm::Real) = 1e-3 * db_to_linear_power(p_dbm)

"""
    watt_to_dbm(p_watt::Real) -> Float64

Convert optical power in watts to dBm (decibels relative to 1 mW):
`P_dBm = 10·log10(P[W]/1mW)`. `p_watt` must be positive.
"""
function watt_to_dbm(p_watt::Real)
    p_watt > 0 || throw(ArgumentError("power must be positive"))
    return linear_power_to_db(p_watt / 1e-3)
end

# ---------------------------------------------------------------------------
# Photon energy & nonlinear-coefficient conversions
# ---------------------------------------------------------------------------

"""
    photon_energy(lambda_m::Real) -> Float64

Energy E = ħω = hc/λ [J] of a single photon at wavelength `lambda_m` [m].
Useful for converting between photon-counting quantities (e.g.
[`add_noise`](@ref)'s `photons_per_mode`) and physical energies/powers.

Reuses the package's reduced Planck constant `ħ` (see [`add_noise`](@ref)).
"""
photon_energy(lambda_m::Real) = ħ * wavelength_to_omega(lambda_m)

"""
    n2_aeff_to_gamma(n2::Real, lambda0_m::Real, Aeff::Real) -> Float64

Convert nonlinear refractive index n₂ [m²/W] and effective mode area A_eff
[m²] at wavelength `lambda0_m` [m] to the waveguide nonlinear coefficient
γ [1/(W·m)] used by [`Medium`](@ref):

    γ = 2π·n₂ / (λ₀·A_eff)

This is the same relation used internally to build γ for
[`HollowCoreFiber`](@ref); use it directly for solid-core fibers/waveguides
when n₂ and A_eff (rather than a pre-computed γ) are known. For a
frequency-dependent A_eff(ω), use [`NonlinearityFromEffectiveArea`](@ref)
instead of evaluating this at a single wavelength.
"""
function n2_aeff_to_gamma(n2::Real, lambda0_m::Real, Aeff::Real)
    lambda0_m > 0 || throw(ArgumentError("lambda0_m must be positive"))
    Aeff > 0 || throw(ArgumentError("Aeff must be positive"))
    return 2π * n2 / (lambda0_m * Aeff)
end

"""
    gamma_aeff_to_n2(gamma::Real, lambda0_m::Real, Aeff::Real) -> Float64

Convert nonlinear coefficient γ [1/(W·m)] and effective mode area A_eff [m²]
at wavelength `lambda0_m` [m] to nonlinear refractive index n₂ [m²/W].
Inverse of [`n2_aeff_to_gamma`](@ref): `n₂ = γ·λ₀·A_eff / (2π)`.
"""
function gamma_aeff_to_n2(gamma::Real, lambda0_m::Real, Aeff::Real)
    lambda0_m > 0 || throw(ArgumentError("lambda0_m must be positive"))
    Aeff > 0 || throw(ArgumentError("Aeff must be positive"))
    return gamma * lambda0_m * Aeff / (2π)
end
