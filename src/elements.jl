
using FFTW
using Random: AbstractRNG, default_rng

"""
    LumpedElement

Abstract base type for point-like stages in a cascaded simulation.
"""
abstract type LumpedElement end

"""
    Amplifier(gain_db)

Represent a lumped amplifier that boosts field amplitudes by a specified decibel gain.
"""
struct Amplifier <: LumpedElement
    gain_db::Float64
end

"""
    Attenuator(loss_db)

Represent a lumped attenuator that reduces field amplitudes by a specified decibel loss.
"""
struct Attenuator <: LumpedElement
    loss_db::Float64
end

"""
    Filter(transfer_function)

Represent a lumped filter that applies a frequency-domain transfer function to the pulse.
"""
struct Filter{F} <: LumpedElement
    transfer_function::F
end

"""
    PMDElement(mean_dgd; rng=default_rng())

Lumped first-order PMD element with mean differential group delay `mean_dgd` [s].

At each call to [`apply`](@ref), draws an independent random realization of:

  - **DGD** Δτ from a Maxwell–Boltzmann distribution with mean `mean_dgd`
  - **PSP orientation** angle θ ∈ [0, π) drawn uniformly

and applies the resulting frequency-domain Jones matrix to a [`VectorialPulse`](@ref):

    J(V) = R(θ) · diag(exp(+i V Δτ/2), exp(−i V Δτ/2)) · R(−θ)

where V = ω − ω₀ [rad/s] is the relative angular frequency and R(θ) is a 2×2 rotation.

The Maxwellian distribution is the correct first-order PMD statistics for a fiber
composed of many random birefringent segments (ITU-T G.650.2 model).

# Arguments

  - `mean_dgd::Real`: Mean differential group delay ⟨Δτ⟩ [s].
    Typical values: 0.5–5 ps for long-haul SMF, < 0.1 ps for short/PM fibers.
  - `rng`: Random number generator for reproducible noise realizations.

# Notes

  - Only applicable to [`VectorialPulse`](@ref). Applying to a scalar [`Pulse`](@ref)
    throws an `ArgumentError`.
  - Independent calls produce statistically independent PMD realizations.
  - For cascaded fibers with multiple PMD stages, pipe multiple `PMDElement`s.

# Example

```julia
pmd = PMDElement(1e-12)   # 1 ps mean DGD
noisy_vpulse = apply(vpulse, pmd)
# or with the pipe operator:
result = vpulse |> PMDElement(0.5e-12) |> fiber_params
```
"""
struct PMDElement <: LumpedElement
    mean_dgd::Float64
    rng::AbstractRNG
end

PMDElement(mean_dgd::Real; rng::AbstractRNG=default_rng()) =
    PMDElement(Float64(mean_dgd), rng)

"""
    apply(pulse::Pulse, element::LumpedElement)

Apply the lumped element to the optical pulse, returning a new `Pulse`.
"""
function apply(pulse::Pulse, amp::Amplifier)
    factor = 10.0^(amp.gain_db / 20.0)
    At = pulse.At .* factor
    AW = pulse.AW .* factor
    return Pulse(At, AW, pulse.grid)
end

function apply(vpulse::VectorialPulse, amp::Amplifier)
    factor = 10.0^(amp.gain_db / 20.0)
    At = vpulse.At .* factor
    AW = vpulse.AW .* factor
    return VectorialPulse(At, AW, vpulse.grid)
end

function apply(pulse::Pulse, att::Attenuator)
    factor = 10.0^(-att.loss_db / 20.0)
    At = pulse.At .* factor
    AW = pulse.AW .* factor
    return Pulse(At, AW, pulse.grid)
end

function apply(vpulse::VectorialPulse, att::Attenuator)
    factor = 10.0^(-att.loss_db / 20.0)
    At = vpulse.At .* factor
    AW = vpulse.AW .* factor
    return VectorialPulse(At, AW, vpulse.grid)
end

function apply(pulse::Pulse, filt::Filter)
    AW = pulse.AW .* filt.transfer_function.(pulse.grid.W)
    At = fft(AW) # fft is standard optics convention: At = fft(AW)
    return Pulse(At, AW, pulse.grid)
end

function apply(vpulse::VectorialPulse, filt::Filter)
    tf = filt.transfer_function.(vpulse.grid.W)
    AW = vpulse.AW .* tf
    At = similar(vpulse.At)
    At[:, 1] = fft(AW[:, 1])
    At[:, 2] = fft(AW[:, 2])
    return VectorialPulse(At, AW, vpulse.grid)
end

function apply(::Pulse, ::PMDElement)
    throw(ArgumentError(
        "PMDElement requires a VectorialPulse. " *
        "PMD is a polarization effect and has no meaning for a scalar (single-polarization) pulse."
    ))
end

"""
    apply(vpulse::VectorialPulse, pmd::PMDElement) -> VectorialPulse

Apply a random first-order PMD realization to `vpulse`.

Draws a Maxwellian DGD and uniform PSP angle, then applies the frequency-domain
Jones matrix. Each call produces an independent realization.
"""
function apply(vpulse::VectorialPulse, pmd::PMDElement)
    # ── 1. Draw DGD from Maxwell–Boltzmann distribution ─────────────────────
    # If g ~ N(0, σ²I₃), then |g| ~ Maxwell(σ), with E[|g|] = 2σ√(2/π).
    # Setting 2σ√(2/π) = mean_dgd  →  σ = mean_dgd · √(π/8)
    σ_maxwell = pmd.mean_dgd * sqrt(π / 8)
    g  = randn(pmd.rng, 3)
    Δτ = σ_maxwell * sqrt(g[1]^2 + g[2]^2 + g[3]^2)   # [s]

    # ── 2. Random PSP orientation θ ∈ [0, π) ─────────────────────────────────
    θ = π * rand(pmd.rng)
    cosθ, sinθ = cos(θ), sin(θ)

    # ── 3. FFT-ordered relative frequency V [rad/s] ──────────────────────────
    # grid.V is monotonic; ifftshift converts to FFT order matching AW columns.
    V_fft = ifftshift(vpulse.grid.V)   # FFT-ordered V = ω − ω₀ [rad/s]

    AW_out = similar(vpulse.AW)

    @inbounds for k in eachindex(V_fft)
        φ  = V_fft[k] * (Δτ / 2)         # half-DGD phase at frequency bin k
        ep = cis(+φ)                       # fast-PSP phasor
        em = cis(-φ)                       # slow-PSP phasor

        Ex = vpulse.AW[k, 1]
        Ey = vpulse.AW[k, 2]

        # Rotate to PSP basis: R(−θ) · [Ex; Ey]
        u =  cosθ * Ex + sinθ * Ey        # projection onto fast PSP
        v = -sinθ * Ex + cosθ * Ey        # projection onto slow PSP

        # Apply differential delay
        u_d = ep * u
        v_d = em * v

        # Rotate back to lab basis: R(+θ) · [u_d; v_d]
        AW_out[k, 1] = cosθ * u_d - sinθ * v_d
        AW_out[k, 2] = sinθ * u_d + cosθ * v_d
    end

    # Recompute time-domain fields via fft (optics convention: At = fft(AW))
    At_out = similar(AW_out)
    At_out[:, 1] = fft(AW_out[:, 1])
    At_out[:, 2] = fft(AW_out[:, 2])

    return VectorialPulse(At_out, AW_out, vpulse.grid)
end

# ── Callable dispatch for piping ─────────────────────────────────────────────
(element::LumpedElement)(pulse::AbstractPulse)     = apply(pulse, element)
(element::LumpedElement)(sol::Solution)            = apply(Pulse(sol), element)
(element::LumpedElement)(vsol::VectorialSolution)  = apply(VectorialPulse(vsol), element)
