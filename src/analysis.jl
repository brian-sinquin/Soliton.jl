module Analysis

using ..Types: Pulse, Solution
using FFTW
using Random: Random, AbstractRNG, default_rng

export pulse_energy, peak_power, fwhm, spectral_bandwidth, time_bandwidth_product
export photon_number, spectral_centroid
export dispersion_length, nonlinear_length, soliton_number
export add_noise, rin_rms, spectral_coherence

# Reduced Planck constant [J·s]
const ħ = 1.054571817e-34

"""
    pulse_energy(pulse::Pulse)

Pulse energy E = ∫|A(t)|²dt [J].
"""
function pulse_energy(pulse::Pulse)
    return sum(abs2, pulse.At) * pulse.grid.dt
end

"""
    peak_power(pulse::Pulse)

Peak power P_peak = max(|A(t)|²) [W].
"""
function peak_power(pulse::Pulse)
    return maximum(abs2, pulse.At)
end

"""
    _fwhm(intensity, axis)

Full width at half maximum of `intensity` sampled on monotonic `axis`,
using linear interpolation at the half-maximum crossings.
"""
function _fwhm(intensity::AbstractVector, axis::AbstractVector)
    peak = maximum(intensity)
    peak > 0 || return 0.0
    half = 0.5 * peak
    above = findall(>=(half), intensity)
    length(above) < 2 && return 0.0
    lo, hi = first(above), last(above)

    left = if lo > 1
        x1, x2 = axis[lo - 1], axis[lo]
        y1, y2 = intensity[lo - 1], intensity[lo]
        x1 + (half - y1) * (x2 - x1) / (y2 - y1)
    else
        axis[lo]
    end

    right = if hi < length(axis)
        x1, x2 = axis[hi], axis[hi + 1]
        y1, y2 = intensity[hi], intensity[hi + 1]
        x1 + (half - y1) * (x2 - x1) / (y2 - y1)
    else
        axis[hi]
    end

    return abs(right - left)
end

"""
    fwhm(pulse::Pulse; domain::Symbol=:time)

Full width at half maximum of the pulse.

`domain = :time` returns the temporal width [s]; `domain = :frequency`
returns the spectral width as an angular-frequency width [rad/s].
"""
function fwhm(pulse::Pulse; domain::Symbol=:time)
    if domain === :time
        return _fwhm(abs2.(pulse.At), pulse.grid.t)
    elseif domain === :frequency
        # grid.V is monotonic; fftshift brings AW to the same ordering.
        return _fwhm(abs2.(fftshift(pulse.AW)), pulse.grid.V)
    else
        throw(ArgumentError("domain must be :time or :frequency"))
    end
end

"""
    spectral_bandwidth(pulse::Pulse; level::Float64=0.5)

Spectral width at the given intensity `level` (0.5 = FWHM), returned in Hz.
"""
function spectral_bandwidth(pulse::Pulse; level::Float64=0.5)
    spectrum = abs2.(fftshift(pulse.AW))
    peak = maximum(spectrum)
    peak > 0 || return 0.0
    above = findall(>=(level * peak), spectrum)
    length(above) < 2 && return 0.0
    V = pulse.grid.V
    return abs(V[last(above)] - V[first(above)]) / (2π)
end

"""
    time_bandwidth_product(pulse::Pulse)

Time-bandwidth product Δt·Δν (dimensionless). Transform-limited references:
≈ 0.441 (Gaussian), ≈ 0.315 (sech²).
"""
function time_bandwidth_product(pulse::Pulse)
    dt = fwhm(pulse; domain=:time)
    dnu = fwhm(pulse; domain=:frequency) / (2π)
    return dt * dnu
end

"""
    spectral_centroid(pulse::Pulse)

Intensity-weighted center frequency of the pulse spectrum relative to the
carrier: ⟨ω - ω₀⟩ [rad/s]. Returns zero for a spectrum centered at the carrier;
positive/negative for red/blue shifts. Useful for tracking spectral drift during
nonlinear propagation.
"""
function spectral_centroid(pulse::Pulse)
    spectrum = abs2.(fftshift(pulse.AW))   # aligned with monotonic grid.V
    return sum(pulse.grid.V .* spectrum) / sum(spectrum)
end

"""
    photon_number(pulse::Pulse)
    photon_number(solution::Solution)

Dimensionless photon count ∝ ∫|A(ω)|²/ω dω — the quantity conserved by the
GNLSE (including self-steepening) in the absence of loss. The energy per photon
scales as ħω, making this count independent of the optical carrier frequency.
This metric is useful for verifying energy conservation and for understanding the
role of ħω in the [`add_noise`](@ref) quantum-noise seeding. For a `Solution`,
returns one value per saved distance.

# Returns

  - Float64 (for Pulse) or Vector (for Solution): photon count
"""
function photon_number(pulse::Pulse)
    # pulse.AW = ifft(At) is in FFT order; align the absolute-frequency grid.
    return sum(abs2.(pulse.AW) ./ ifftshift(pulse.grid.W))
end

function photon_number(solution::Solution)
    # solution.AW columns and solution.W are both in monotonic order
    return [sum(abs2.(view(solution.AW, :, j)) ./ solution.W)
            for j in axes(solution.AW, 2)]
end

"""
    dispersion_length(beta2, T0)

Dispersion length L_D = T₀² / |β₂| [m], the distance over which a pulse of
characteristic width T₀ disperses significantly due to chromatic dispersion.
Compares to nonlinear length to determine whether dispersion or nonlinearity
dominates the pulse evolution. See [`soliton_number`](@ref).
"""
dispersion_length(beta2::Real, T0::Real) = T0^2 / abs(beta2)

"""
    nonlinear_length(gamma, P0)

Nonlinear length L_NL = 1 / (γ P₀) [m], the distance over which a pulse of peak
power P₀ undergoes significant nonlinear phase modulation. Compares to
dispersion length to determine the dominant physics. See [`soliton_number`](@ref).
"""
nonlinear_length(gamma::Real, P0::Real) = 1 / (gamma * P0)

"""
    soliton_number(beta2, gamma, T0, P0)

Soliton number N = √(L_D / L_NL) = √(γ P₀ T₀² / |β₂|) (dimensionless). This
parameter predicts the number of fundamental solitons that comprise the initial
pulse and governs nonlinear-dispersive dynamics:
  - N ≪ 1: weakly nonlinear, dispersion dominates
  - N ≈ 1: fundamental soliton (stable in anomalous dispersion)
  - N > 1: higher-order soliton exhibiting periodic breathing; also indicates
    soliton-fission regime where multiple solitons emerge

The higher-order soliton period is approximately Tfission ≈ π L_D / 2 ≈ π T₀² / (2|β₂|).
"""
function soliton_number(beta2::Real, gamma::Real, T0::Real, P0::Real)
    return sqrt(gamma * P0 * T0^2 / abs(beta2))
end

"""
    rin_rms(psd_dbc_hz, bandwidth) -> Float64

RMS relative intensity fluctuation σ_P/P obtained by integrating a (flat)
relative-intensity-noise power spectral density `psd_dbc_hz` [dBc/Hz] over a
one-sided detection `bandwidth` [Hz]:

    σ² = ∫₀^B S(f) df = 10^(RIN/10) · B

Use the result as the `rin` argument of [`add_noise`](@ref). Example: a laser
with −150 dBc/Hz RIN observed over a 1 GHz bandwidth gives
`rin_rms(-150, 1e9) ≈ 3.2e-4` (0.03 % RMS power fluctuation).
"""
function rin_rms(psd_dbc_hz::Real, bandwidth::Real)
    bandwidth > 0 || throw(ArgumentError("bandwidth must be positive"))
    return sqrt(10.0^(psd_dbc_hz / 10) * bandwidth)
end

"""
    add_noise(pulse::Pulse; kwargs...) -> Pulse

Return a copy of `pulse` with a physically motivated realization of input noise
added. Three independent contributions can be enabled and tuned separately:

 1. **Quantum noise** — vacuum fluctuations of the optical field, modelled as
    `photons_per_mode` photons per spectral mode. This is the fundamental seed
    for noise-driven dynamics (modulation instability, supercontinuum
    decoherence) and is the only term enabled by default.
 2. **Relative intensity noise (RIN)** — classical shot-to-shot fluctuation of
    the laser output power, applied as a multiplicative amplitude scaling.
 3. **Phase noise** — shot-to-shot common-mode optical phase jitter.

Independent `rng` draws give statistically independent realizations, so calling
`add_noise` repeatedly on the same clean pulse builds the ensemble needed for a
[`spectral_coherence`](@ref) study.

# Keyword arguments

  - `rng::AbstractRNG = default_rng()`: random source.
  - `photons_per_mode::Real = 1.0`: quantum-noise level. `1.0` is the standard
    one-photon-per-mode seed (Dudley & Coen); `0.5` corresponds to the vacuum
    zero-point energy ħω/2; `0.0` disables quantum noise.
  - `quantum_model::Symbol = :gaussian`: `:gaussian` draws each field quadrature
    from an independent normal distribution (Rayleigh-distributed amplitude,
    uniform phase) — the physically faithful model of a vacuum/coherent state.
    `:phase_only` uses a fixed per-mode amplitude with a uniformly random phase,
    i.e. the classic Dudley & Coen seed.
  - `rin::Real = 0.0`: RMS relative intensity noise σ_P/P (fractional, e.g.
    `0.01` = 1 % RMS power fluctuation). Convert a dBc/Hz spec with
    [`rin_rms`](@ref).
  - `phase_rms::Real = 0.0`: RMS optical phase jitter [rad].

# Physics

In the package FFT convention the energy of spectral mode `m` is
`N·dt·|AW[m]|²`, so a mode carrying `nₚ` photons of energy ħω satisfies
`N·dt·⟨|δAW|²⟩ = nₚ·ħω`. RIN scales the field by `√(1 + δ)` with
`δ ~ 𝒩(0, rin²)`; phase noise multiplies it by `exp(iφ)` with
`φ ~ 𝒩(0, phase_rms²)`.

Reference: J. M. Dudley & S. Coen, Opt. Lett. 27, 1180 (2002);
J. M. Dudley, G. Genty & S. Coen, Rev. Mod. Phys. 78, 1135 (2006).
"""
function add_noise(
    pulse::Pulse;
    rng::AbstractRNG=default_rng(),
    photons_per_mode::Real=1.0,
    quantum_model::Symbol=:gaussian,
    rin::Real=0.0,
    phase_rms::Real=0.0,
)
    photons_per_mode >= 0 ||
        throw(ArgumentError("photons_per_mode must be non-negative"))
    rin >= 0 || throw(ArgumentError("rin must be non-negative"))
    quantum_model in (:gaussian, :phase_only) ||
        throw(ArgumentError("quantum_model must be :gaussian or :phase_only"))

    grid = pulse.grid
    N, dt = grid.N, grid.dt

    # --- Classical laser noise: RIN amplitude scaling + common-mode phase jitter
    At = copy(pulse.At)
    if rin > 0 || phase_rms > 0
        # δ is the fractional power fluctuation; the field scales as √(1 + δ).
        amp = rin > 0 ? sqrt(max(0.0, 1.0 + rin * randn(rng))) : 1.0
        ϕ = phase_rms > 0 ? phase_rms * randn(rng) : 0.0
        @. At *= amp * cis(ϕ)
    end

    AW = ifft(At)

    # --- Quantum noise: photons_per_mode photons per spectral mode -------------
    if photons_per_mode > 0
        Wabs = ifftshift(grid.W)        # absolute frequency per FFT bin [rad/s]
        if quantum_model === :gaussian
            # Complex-Gaussian quadratures: var per quadrature = nₚ·ħω/(2·N·dt),
            # so N·dt·⟨|δAW|²⟩ = nₚ·ħω.
            scale = sqrt(photons_per_mode * ħ / (2 * N * dt))
            @inbounds for m in eachindex(AW)
                σ = scale * sqrt(abs(Wabs[m]))
                AW[m] += σ * complex(randn(rng), randn(rng))
            end
        else  # :phase_only — fixed amplitude, uniform random phase
            scale = sqrt(photons_per_mode * ħ / (N * dt))
            @inbounds for m in eachindex(AW)
                AW[m] += scale * sqrt(abs(Wabs[m])) * cis(2π * rand(rng))
            end
        end
    end

    return Pulse(fft(AW), AW, grid)
end

"""
    spectral_coherence(spectra) -> Vector{Float64}

Modulus of the complex degree of first-order coherence |g₁₂⁽¹⁾(ω)| at zero path
delay, evaluated bin-by-bin across an ensemble of independent spectra:

    g(ω) = |⟨Aᵢ*(ω) Aⱼ(ω)⟩_{i≠j}| / ⟨|A(ω)|²⟩

`spectra` may be a vector of complex frequency-domain fields (all equal length),
a vector of [`Pulse`](@ref)s, or a vector of [`Solution`](@ref)s (the spectrum
at the final distance is used). Returns g ∈ [0, 1]: 1 = fully coherent (the
supercontinuum is reproducible shot-to-shot), 0 = incoherent (noise-dominated).

The estimator uses the algebraic identity `Σ_{i≠j} Aᵢ*Aⱼ = |ΣAᵢ|² - Σ|Aᵢ|²`,
which averages over all `M(M-1)` ordered pairs without an explicit double loop.

!!! note "Finite-ensemble bias"
    For a truly incoherent field the pairwise estimator does not vanish but
    fluctuates around a positive floor `≈ 1/√(M(M-1)) ≈ 1/M`. Use a sufficiently
    large ensemble (`M ≳ 20`, ideally 50–100) so that this bias stays well below
    the coherence features of interest.

Reference: J. M. Dudley & S. Coen, Opt. Lett. 27, 1180 (2002).
"""
function spectral_coherence(spectra::AbstractVector{<:AbstractVector{<:Complex}})
    M = length(spectra)
    M >= 2 || throw(ArgumentError("need at least two spectra for an ensemble"))
    N = length(first(spectra))
    all(s -> length(s) == N, spectra) ||
        throw(ArgumentError("all spectra must have equal length"))

    S = zeros(ComplexF64, N)   # Σᵢ Aᵢ
    P = zeros(Float64, N)      # Σᵢ |Aᵢ|²
    for s in spectra
        @. S += s
        @. P += abs2(s)
    end

    g = similar(P)
    @inbounds for k in eachindex(g)
        # Σ_{i≠j} Aᵢ*Aⱼ = |ΣAᵢ|² - Σ|Aᵢ|²; normalize by M(M-1) and ⟨|A|²⟩
        denom = (M - 1) * P[k]
        g[k] = denom > 0 ? abs(abs2(S[k]) - P[k]) / denom : 0.0
    end
    return g
end

"""
    spectral_coherence(pulses::AbstractVector{<:Pulse})

Convenience overload: accepts a vector of [`Pulse`](@ref) objects and extracts
their frequency-domain envelopes (AW fields) before computing coherence.
"""
spectral_coherence(pulses::AbstractVector{<:Pulse}) =
    spectral_coherence([p.AW for p in pulses])

"""
    spectral_coherence(solutions::AbstractVector{<:Solution})

Convenience overload: accepts a vector of [`Solution`](@ref) objects and extracts
the final spectrum (AW field at the last propagation distance) from each,
then computes coherence across the ensemble.
"""
spectral_coherence(solutions::AbstractVector{<:Solution}) =
    spectral_coherence([@view(sol.AW[:, end]) for sol in solutions])

end # module
