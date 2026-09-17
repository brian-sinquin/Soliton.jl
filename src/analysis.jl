"""
Analysis functions for pulse characterization (natural SI units).

Provides energy, peak power, bandwidth, pulse-duration, soliton, noise-seeding
and coherence metrics.
"""

using FFTW
using Random: Random, AbstractRNG, default_rng

# Reduced Planck constant [J·s]
const ħ = 1.054571817e-34

"""
    pulse_energy(pulse::Pulse)

Pulse energy E = ∫|A(t)|²dt [J].
"""
function pulse_energy(pulse::Pulse)
    return sum(abs2, pulse.At) * pulse.grid.dt
end

function pulse_energy(vpulse::VectorialPulse)
    return (sum(abs2, @view(vpulse.At[:, 1])) + sum(abs2, @view(vpulse.At[:, 2]))) *
           vpulse.grid.dt
end

"""
    peak_power(pulse::Pulse)

Peak power P_peak = max(|A(t)|²) [W].
"""
function peak_power(pulse::Pulse)
    return maximum(abs2, pulse.At)
end

function peak_power(vpulse::VectorialPulse)
    return maximum(abs2.(vpulse.At[:, 1]) .+ abs2.(vpulse.At[:, 2]))
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

Spectral width at the given intensity `level` (0.5 = FWHM), returned as ordinary
frequency ν [Hz] (i.e. divided by 2π; not angular frequency ω [rad/s]).
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
    pulse_energy_estimate(Pmax, FWHM; shape=:sech) -> Float64

Analytic pulse energy E [J] for an *ideal* transform-limited pulse of peak
power `Pmax` [W] and intensity FWHM [s], given the envelope `shape`
(`:sech`, `:gaussian`, or `:lorentzian`, matching [`sech_pulse`](@ref),
[`gaussian_pulse`](@ref), [`lorentzian_pulse`](@ref)):

    E = Pmax · FWHM · k(shape)

with `k(:sech) = 2/m ≈ 1.1346`, `k(:gaussian) = √(π/(4ln2)) ≈ 1.0645`,
`k(:lorentzian) = (π/2)/m ≈ 1.2203`, where each shape's `m` matches the
corresponding pulse-generator function. Inverse of
[`peak_power_estimate`](@ref).

Use this to relate a datasheet or measured (autocorrelator/scope) peak power
and FWHM to pulse energy without constructing a [`Pulse`](@ref)/[`Grid`](@ref)
— e.g. for a quick back-of-envelope estimate before setting up a simulation.
For the exact numerical energy of an actual `Pulse` (which may not be a pure
transform-limited shape), use [`pulse_energy`](@ref) instead.
"""
function pulse_energy_estimate(Pmax::Real, FWHM::Real; shape::Symbol=:sech)
    Pmax >= 0 || throw(ArgumentError("Pmax must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))
    return Pmax * FWHM * _energy_fwhm_factor(shape)
end

"""
    peak_power_estimate(energy, FWHM; shape=:sech) -> Float64

Analytic peak power P_max [W] for a transform-limited pulse of energy
`energy` [J], intensity FWHM [s], and envelope `shape` (see
[`pulse_energy_estimate`](@ref) for the supported shapes and the underlying
`E = Pmax·FWHM·k(shape)` relation). Inverse of [`pulse_energy_estimate`](@ref):
`Pmax = E / (FWHM·k(shape))`.
"""
function peak_power_estimate(energy::Real, FWHM::Real; shape::Symbol=:sech)
    energy >= 0 || throw(ArgumentError("energy must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))
    return energy / (FWHM * _energy_fwhm_factor(shape))
end

"""
    _energy_fwhm_factor(shape::Symbol) -> Float64

Dimensionless factor `k` in `E = Pmax·FWHM·k` for the intensity profile of
`shape` (see [`sech_pulse`](@ref), [`gaussian_pulse`](@ref),
[`lorentzian_pulse`](@ref) for the corresponding field envelopes), derived by
integrating `∫|A(T)|²dT` in closed form.
"""
function _energy_fwhm_factor(shape::Symbol)
    if shape === :sech
        m = 2 * log(1 + sqrt(2))
        return 2 / m
    elseif shape === :gaussian
        m = 4 * log(2)
        return sqrt(π / m)
    elseif shape === :lorentzian
        m = 2 * sqrt(sqrt(2) - 1)
        return (π / 2) / m
    else
        throw(ArgumentError("shape must be :sech, :gaussian, or :lorentzian"))
    end
end

"""
    average_power(energy, rep_rate) -> Float64

Average optical power P_avg [W] of a periodic pulse train with per-pulse
energy `energy` [J] and repetition rate `rep_rate` [Hz]: `P_avg = E·f_rep`.
Inverse of [`pulse_energy_from_average`](@ref).
"""
function average_power(energy::Real, rep_rate::Real)
    energy >= 0 || throw(ArgumentError("energy must be non-negative"))
    rep_rate > 0 || throw(ArgumentError("rep_rate must be positive"))
    return energy * rep_rate
end

"""
    pulse_energy_from_average(P_avg, rep_rate) -> Float64

Per-pulse energy E [J] of a periodic pulse train given average power `P_avg`
[W] and repetition rate `rep_rate` [Hz]: `E = P_avg / f_rep`. Inverse of
[`average_power`](@ref).
"""
function pulse_energy_from_average(P_avg::Real, rep_rate::Real)
    P_avg >= 0 || throw(ArgumentError("P_avg must be non-negative"))
    rep_rate > 0 || throw(ArgumentError("rep_rate must be positive"))
    return P_avg / rep_rate
end

"""
    peak_power_from_average(P_avg, rep_rate, FWHM; shape=:sech) -> Float64

Estimate the peak power P_max [W] of a mode-locked pulse train from the
quantities usually read off a power meter and a scope/spectrum analyzer:
average power `P_avg` [W], repetition rate `rep_rate` [Hz], and intensity
FWHM [s], assuming a transform-limited envelope `shape` (see
[`pulse_energy_estimate`](@ref)).

Composes [`pulse_energy_from_average`](@ref) and [`peak_power_estimate`](@ref):
first recovers the per-pulse energy from `P_avg` and `rep_rate`, then converts
to peak power via the shape's energy/FWHM relation. This is a shape-aware
replacement for the common lab estimate `P_peak ≈ P_avg / (f_rep·FWHM)`, which
implicitly (and only approximately) assumes a rectangular pulse.
"""
function peak_power_from_average(
    P_avg::Real, rep_rate::Real, FWHM::Real; shape::Symbol=:sech
)
    E = pulse_energy_from_average(P_avg, rep_rate)
    return peak_power_estimate(E, FWHM; shape=shape)
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

Conserved quantity ∝ ∫|A(ω)|²/ω dω used to monitor numerical accuracy of the
GNLSE integration. For a lossless fiber this quantity is conserved by the GNLSE
(including self-steepening); a drift indicates the step-size tolerance is too
loose. Note: the returned value is *not* an absolute photon count — it lacks the
ℏ and dω normalization factors and should only be compared *relative* to itself
along the propagation axis. For a `Solution`, returns one value per saved distance.
"""
function photon_number(pulse::Pulse)
    # pulse.AW = ifft(At) is in FFT order; align the absolute-frequency grid.
    return sum(abs2.(pulse.AW) ./ ifftshift(pulse.grid.W))
end

function photon_number(solution::Solution)
    if isempty(solution.AW)
        # Reconstruct FFT-natural spectra and align the frequency denominator.
        W = ifftshift(solution.W)
        return [sum(abs2.(ifft(view(solution.At, :, j))) ./ W)
                for j in axes(solution.At, 2)]
    end
    # solution.AW columns and solution.W are both in monotonic order
    return [sum(abs2.(view(solution.AW, :, j)) ./ solution.W) for j in axes(solution.AW, 2)]
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
    soliton_period(beta2, T0) -> Float64

Fundamental soliton period `z₀ = (π/2)·L_D = π·T₀²/(2|β₂|)` [m]: the
propagation distance over which a fundamental soliton (N ≈ 1, see
[`soliton_number`](@ref)) recovers its input shape after one cycle of
periodic phase evolution, and the characteristic breathing period for
higher-order solitons (N > 1) referenced in [`soliton_number`](@ref)'s
docstring as `Tfission ≈ π·L_D/2`.

Reference: G. P. Agrawal, "Nonlinear Fiber Optics," 6th ed., Eq. (5.2.9).
"""
soliton_period(beta2::Real, T0::Real) = (π / 2) * dispersion_length(beta2, T0)

"""
    modulation_instability_gain(beta2, gamma, P0, Omega) -> Float64
    modulation_instability_gain(beta2, gamma, P0, Omega::AbstractVector) -> Vector{Float64}

Power-gain coefficient g(Ω) [1/m] for scalar modulation instability (MI) of a
continuous-wave/quasi-CW pump of power `P0` [W] in a fiber with GVD `beta2`
[s²/m] and nonlinear coefficient `gamma` [1/(W·m)], at angular-frequency
detuning `Omega` [rad/s] from the pump. A weak perturbation at detuning Ω
grows as `exp(g(Ω)·z)` along the fiber:

    g(Ω) = |β₂Ω|·√(Ω_c² − Ω²),   Ω_c = √(4γP₀/|β₂|)

Only anomalous dispersion (β₂ < 0) supports scalar MI; `g(Ω) = 0` is returned
for normal dispersion (β₂ ≥ 0) or for `|Ω| ≥ Ω_c`. The peak gain
`g_max = 2γP₀` occurs at `Ω = Ω_c/√2` (see [`mi_peak_frequency`](@ref)); the
full gain bandwidth is `2Ω_c` (see [`mi_bandwidth`](@ref)).

MI is the mechanism that seeds soliton fission and supercontinuum generation
under quasi-CW or long-pulse pumping. Compare `g(Ω)` to `1/L` (fiber length)
to gauge whether sidebands grow appreciably over the propagation length; for
short-pulse pumping use [`soliton_number`](@ref)/[`nonlinear_length`](@ref)
instead.

Reference: J. M. Dudley, G. Genty & S. Coen, Rev. Mod. Phys. 78, 1135 (2006),
Eq. (3); G. P. Agrawal, "Nonlinear Fiber Optics," 6th ed., §5.1.
"""
function modulation_instability_gain(beta2::Real, gamma::Real, P0::Real, Omega::Real)
    gamma >= 0 || throw(ArgumentError("gamma must be non-negative"))
    P0 >= 0 || throw(ArgumentError("P0 must be non-negative"))
    beta2 < 0 || return 0.0
    Omega_c2 = 4 * gamma * P0 / abs(beta2)
    Omega2 = Omega^2
    Omega2 < Omega_c2 || return 0.0
    return abs(beta2 * Omega) * sqrt(Omega_c2 - Omega2)
end

modulation_instability_gain(
    beta2::Real, gamma::Real, P0::Real, Omega::AbstractVector{<:Real}
) = modulation_instability_gain.(beta2, gamma, P0, Omega)

"""
    mi_peak_frequency(beta2, gamma, P0) -> Float64

Angular-frequency detuning `Ω_max = √(2γP₀/|β₂|)` [rad/s] at which
[`modulation_instability_gain`](@ref) is maximal, giving peak gain
`g_max = 2γP₀` [1/m]. Requires anomalous dispersion (`beta2 < 0`).
"""
function mi_peak_frequency(beta2::Real, gamma::Real, P0::Real)
    beta2 < 0 ||
        throw(ArgumentError("modulation instability requires anomalous dispersion (beta2 < 0)"))
    gamma >= 0 || throw(ArgumentError("gamma must be non-negative"))
    P0 >= 0 || throw(ArgumentError("P0 must be non-negative"))
    return sqrt(2 * gamma * P0 / abs(beta2))
end

"""
    mi_bandwidth(beta2, gamma, P0) -> Float64

Full modulation-instability gain bandwidth `2Ω_c` [rad/s], the width of the
interval `|Ω| < Ω_c` over which [`modulation_instability_gain`](@ref) is
nonzero, with `Ω_c = √(4γP₀/|β₂|)`. Requires anomalous dispersion
(`beta2 < 0`).
"""
function mi_bandwidth(beta2::Real, gamma::Real, P0::Real)
    beta2 < 0 ||
        throw(ArgumentError("modulation instability requires anomalous dispersion (beta2 < 0)"))
    gamma >= 0 || throw(ArgumentError("gamma must be non-negative"))
    P0 >= 0 || throw(ArgumentError("P0 must be non-negative"))
    return 2 * sqrt(4 * gamma * P0 / abs(beta2))
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
    return sqrt(db_to_linear_power(psd_dbc_hz) * bandwidth)
end

"""
    add_noise(pulse::Pulse; kwargs...) -> Pulse

Return a copy of `pulse` with a physically motivated realization of input noise
added. Four independent contributions can be enabled and tuned separately:

 1. **Quantum noise** — vacuum fluctuations of the optical field, modelled as
    `photons_per_mode` photons per spectral mode. This is the fundamental seed
    for noise-driven dynamics (modulation instability, supercontinuum
    decoherence) and is the only term enabled by default.
 2. **Relative intensity noise (RIN)** — classical shot-to-shot fluctuation of
    the laser output power, applied as a multiplicative amplitude scaling.
 3. **Phase noise (shot-to-shot)** — common-mode optical phase jitter drawn
    from a single Gaussian deviate (white phase noise).
 4. **Laser linewidth** — frequency-domain colored phase noise with a
    Lorentzian power spectrum corresponding to a laser of linewidth `linewidth_hz`.
    Modeled as a Wiener (random-walk) phase process: the phase evolves as
    Brownian motion in time, giving a Lorentzian electric-field spectrum with
    FWHM = `linewidth_hz`.

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
  - `phase_rms::Real = 0.0`: RMS common-mode optical phase jitter [rad]
    (shot-to-shot; white phase noise).
  - `linewidth_hz::Real = 0.0`: Laser linewidth [Hz] (half-maximum of the
    Lorentzian power spectrum). Generates a time-domain Wiener phase process
    with diffusion coefficient `D = 2π · linewidth_hz`. Typical values:
    < 1 kHz (narrow-linewidth CW), 1–100 MHz (standard DFB), > 10 GHz (free-running).

# Physics

In the package FFT convention the energy of spectral mode `m` is
`N·dt·|AW[m]|²`, so a mode carrying `nₚ` photons of energy ħω satisfies
`N·dt·⟨|δAW|²⟩ = nₚ·ħω`. RIN scales the field by `√(1 + δ)` with
`δ ~ 𝒩(0, rin²)`; the shot-to-shot phase noise multiplies it by `exp(iφ)` with
`φ ~ 𝒩(0, phase_rms²)`.

The Wiener phase process satisfies `φ(t+dt) = φ(t) + 𝒩(0, 2π·Δν·dt)`, giving
a Lorentzian electric-field autocorrelation `⟨E*(0)E(τ)⟩ ∝ exp(−π|Δν||τ|)` and
power spectrum FWHM = Δν (Schawlow–Townes white-frequency-noise limit).

Reference: J. M. Dudley & S. Coen, Opt. Lett. 27, 1180 (2002);
J. M. Dudley, G. Genty & S. Coen, Rev. Mod. Phys. 78, 1135 (2006);
A. L. Schawlow & C. H. Townes, Phys. Rev. 112, 1940 (1958).
"""
function add_noise(
    pulse::Pulse;
    rng::AbstractRNG=default_rng(),
    photons_per_mode::Real=1.0,
    quantum_model::Symbol=:gaussian,
    rin::Real=0.0,
    phase_rms::Real=0.0,
    linewidth_hz::Real=0.0,
)
    photons_per_mode >= 0 || throw(ArgumentError("photons_per_mode must be non-negative"))
    rin >= 0 || throw(ArgumentError("rin must be non-negative"))
    linewidth_hz >= 0 || throw(ArgumentError("linewidth_hz must be non-negative"))
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

    # --- Laser linewidth: Wiener phase process (colored phase noise) -----------
    # φ(t) is a Brownian-motion phase with diffusion D = 2π · Δν [rad²/s].
    # Per-step variance: Var[dφ] = 2π · Δν · dt.
    # This gives a Lorentzian electric-field PSD with FWHM = Δν [Hz].
    if linewidth_hz > 0
        σ_step = sqrt(2π * linewidth_hz * pulse.grid.dt)
        φ = cumsum(σ_step .* randn(rng, pulse.grid.N))
        @. At *= cis(φ)
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
their frequency-domain envelopes in monotonic grid.W order before computing coherence.
"""
spectral_coherence(pulses::AbstractVector{<:Pulse}) =
    spectral_coherence([fftshift(p.AW) for p in pulses])

"""
    spectral_coherence(solutions::AbstractVector{<:Solution})

Convenience overload: accepts a vector of [`Solution`](@ref) objects and extracts
the final spectrum (AW field at the last propagation distance) from each,
then computes coherence across the ensemble in monotonic sol.W order.
When spectra were not saved, reconstructs them from the time-domain fields.
"""
spectral_coherence(solutions::AbstractVector{<:Solution}) =
    spectral_coherence([isempty(sol.AW) ? fftshift(ifft(sol.At[:, end])) :
                        sol.AW[:, end] for sol in solutions])

"""
    _unwrap(phase::AbstractVector{<:Real}) -> Vector{Float64}

Unwrap a sequence of phase samples (radians) by removing 2π discontinuities
between consecutive points, so that the result is a continuous phase ramp
suitable for differentiation. Standard sequential unwrap algorithm.
"""
function _unwrap(phase::AbstractVector{<:Real})
    unwrapped = zeros(Float64, length(phase))
    isempty(phase) && return unwrapped
    unwrapped[1] = phase[1]
    @inbounds for i in 2:length(phase)
        d = phase[i] - phase[i - 1]
        d -= 2π * round(d / (2π))
        unwrapped[i] = unwrapped[i - 1] + d
    end
    return unwrapped
end

"""
    instantaneous_frequency(pulse::Pulse) -> Vector{Float64}

Instantaneous angular-frequency deviation δω(t) = -dφ/dt [rad/s] from the
carrier, where φ(t) = arg(A(t)) is the unwrapped temporal phase of the pulse
envelope. This is the time-domain chirp: a linear chirp appears as
`δω(t) ∝ t`, and self-phase modulation on an unchirped pulse produces the
characteristic S-shaped chirp (red-shifted leading edge, blue-shifted
trailing edge, for `γ, P₀ > 0`) that seeds spectral broadening.

Computed via central finite differences of the unwrapped phase on the
uniform grid `pulse.grid.t` (one-sided differences at the endpoints).

!!! note "Low-intensity wings"

    Where `|A(t)|² ≈ 0`, the phase is dominated by numerical noise and its
    derivative can show large spurious excursions unrelated to the physical
    chirp. Mask the result with `abs2.(pulse.At)` (e.g. only trust
    `instantaneous_frequency` where intensity exceeds a few percent of the
    peak) before interpreting it.

# See also

[`spectrogram`](@ref) and [`shg_frog_trace`](@ref) for time-frequency
visualizations that do not require explicit phase unwrapping.
"""
function instantaneous_frequency(pulse::Pulse)
    phase = _unwrap(angle.(pulse.At))
    dt = pulse.grid.dt
    N = length(phase)
    domega = zeros(Float64, N)
    N >= 2 || return domega
    domega[1] = -(phase[2] - phase[1]) / dt
    domega[N] = -(phase[N] - phase[N - 1]) / dt
    @inbounds for i in 2:(N - 1)
        domega[i] = -(phase[i + 1] - phase[i - 1]) / (2dt)
    end
    return domega
end

"""
    spectrogram(pulse::Pulse; n_delay=200, gate_fwhm=nothing) -> (t_delays, V_grid, S_matrix)

Compute Short-Time Fourier Transform (STFT) spectrogram of pulse:
S(t, ω) = |∫ A(t') exp(-(t' - t)² / (2 τ_g²)) exp(-i ω t') dt'|²
"""
function spectrogram(
    pulse::Pulse; n_delay::Int=200, gate_fwhm::Union{Real, Nothing}=nothing
)
    t = pulse.grid.t
    At = pulse.At
    dt = pulse.grid.dt
    N = pulse.grid.N
    V = pulse.grid.V

    tau_g = gate_fwhm === nothing ? fwhm(pulse; domain=:time) : Float64(gate_fwhm)
    tau_g = max(tau_g, 10 * dt)
    sigma_g = tau_g / (2.0 * sqrt(2.0 * log(2.0)))

    t_delays = collect(range(t[1], t[end]; length=n_delay))
    S_matrix = zeros(Float64, N, n_delay)

    gate_buf = zeros(ComplexF64, N)
    freq_buf = zeros(ComplexF64, N)

    for (j, tau) in enumerate(t_delays)
        @. gate_buf = At * exp(-((t - tau)^2) / (2.0 * sigma_g^2))
        freq_buf .= fftshift(ifft(gate_buf))
        @. S_matrix[:, j] = abs2(freq_buf)
    end

    return t_delays, V, S_matrix
end

"""
    shg_frog_trace(pulse::Pulse; n_delay=200) -> (delays, V_shg, I_frog)

Compute Second-Harmonic Generation (SHG) FROG trace:
I_FROG(ω, τ) = |∫ A(t) A(t - τ) exp(-i ω t) dt|²
"""
function shg_frog_trace(pulse::Pulse; n_delay::Int=200)
    t = pulse.grid.t
    At = pulse.At
    N = pulse.grid.N
    dt = pulse.grid.dt

    t_delays = collect(range(t[1] / 2, t[end] / 2; length=n_delay))
    I_frog = zeros(Float64, N, n_delay)
    signal = zeros(ComplexF64, N)

    for (j, tau) in enumerate(t_delays)
        for i in 1:N
            t_shifted = t[i] - tau
            if t[1] <= t_shifted <= t[end]
                idx = round(Int, (t_shifted - t[1]) / dt) + 1
                idx_clamped = clamp(idx, 1, N)
                signal[i] = At[i] * At[idx_clamped]
            else
                signal[i] = 0.0 + 0.0im
            end
        end
        I_frog[:, j] .= abs2.(fftshift(ifft(signal)))
    end

    # The carrier doubles, but the DFT detuning-bin spacing is unchanged.
    V_shg = pulse.grid.V
    return t_delays, V_shg, I_frog
end

"""
    track_solitons(sol::Solution) -> (z_fiss, peak_power_z, centroid_w_z)

Automated tracking of soliton fission and Soliton Self-Frequency Shift (SSFS) red-shift trajectory
across propagation distances `sol.Z`.

# Returns

  - `z_fiss`: Estimated distance of peak compression / soliton fission [m]
  - `peak_power_z`: Peak power P_max(z) [W] along propagation
  - `centroid_w_z`: Spectral centroid ⟨ω - ω₀⟩(z) [rad/s] along propagation
"""
function track_solitons(sol::Solution)
    n_saves = length(sol.Z)
    peak_power_z = zeros(Float64, n_saves)
    centroid_w_z = zeros(Float64, n_saves)

    V = sol.W .- sol.omega0

    for j in 1:n_saves
        intensity_t = abs2.(@view(sol.At[:, j]))
        peak_power_z[j] = maximum(intensity_t)

        spectrum_w = abs2.(@view(sol.AW[:, j]))
        sum_spec = sum(spectrum_w)
        centroid_w_z[j] = sum_spec > 0 ? sum(V .* spectrum_w) / sum_spec : 0.0
    end

    idx_fiss = argmax(peak_power_z)
    z_fiss = sol.Z[idx_fiss]

    return z_fiss, peak_power_z, centroid_w_z
end

"""
    dispersive_wave_wavelength(medium::Medium, pulse::Pulse; P0::Real=peak_power(pulse)) -> Float64

Calculate the predicted resonant Cherenkov dispersive wave emission wavelength λ_DW [m]
emitted during soliton fission in `medium`.

Solves the phase-matching condition:
Δβ(Δω) = β(ω₀ + Δω) - β(ω₀) - β₁(ω₀)·Δω - ½·γ·P₀ = 0
for Δω ≠ 0.
"""
function dispersive_wave_wavelength(
    medium::Medium, pulse::Pulse; P0::Real=peak_power(pulse)
)
    lambda0 = medium.lambda0
    omega0 = 2π * c / lambda0
    gamma_val = medium.gamma isa Real ? Float64(medium.gamma) : 0.0

    V = pulse.grid.V
    D_op = dispersion_operator(pulse.grid, medium)

    # B_comoving: propagation-constant deviation in the co-moving frame,
    # i.e. imag(D_op) = β(ω) - β₀ - β₁·(ω-ω₀). This already has β₁·V removed,
    # which is exactly the form needed for the Cherenkov phase-matching condition.
    B_comoving = imag.(D_op)
    phase_mismatch = @. B_comoving - 0.5 * gamma_val * P0

    N = length(V)
    zero_crossings = Float64[]

    for i in 1:(N - 1)
        if abs(V[i]) > 1e12 # Skip trivial root at pump frequency V ≈ 0
            if phase_mismatch[i] * phase_mismatch[i + 1] <= 0
                dw_zero =
                    V[i] -
                    phase_mismatch[i] * (V[i + 1] - V[i]) /
                    (phase_mismatch[i + 1] - phase_mismatch[i])
                push!(zero_crossings, dw_zero)
            end
        end
    end

    if isempty(zero_crossings)
        if medium.dispersion isa TaylorDispersion && length(medium.dispersion.betas) >= 2
            b2 = medium.dispersion.betas[1]
            b3 = medium.dispersion.betas[2]
            if b3 != 0
                dw_anal = -3.0 * b2 / b3
                omega_dw = omega0 + dw_anal
                return 2π * c / omega_dw
            end
        end
        return lambda0
    end

    pos_crossings = filter(dw -> dw > 1e12, zero_crossings)
    # Select the nearest positive-detuning crossing (physically closest to the pump)
    dw_primary = if !isempty(pos_crossings)
        pos_crossings[argmin(pos_crossings)]
    else
        (!isempty(zero_crossings) ? zero_crossings[argmax(abs.(zero_crossings))] : 0.0)
    end
    omega_dw = omega0 + dw_primary
    return 2π * c / omega_dw
end
