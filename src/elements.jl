using FFTW

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
    apply(pulse::Pulse, element::LumpedElement)

Apply the lumped element to the optical pulse, returning a new `Pulse`.
"""
function apply(pulse::Pulse, amp::Amplifier)
    factor = 10.0^(amp.gain_db / 20.0)
    At = pulse.At .* factor
    AW = pulse.AW .* factor
    return Pulse(At, AW, pulse.grid)
end

function apply(pulse::Pulse, att::Attenuator)
    factor = 10.0^(-att.loss_db / 20.0)
    At = pulse.At .* factor
    AW = pulse.AW .* factor
    return Pulse(At, AW, pulse.grid)
end

function apply(pulse::Pulse, filt::Filter)
    AW = pulse.AW .* filt.transfer_function.(pulse.grid.W)
    At = fft(AW) # fft is standard optics convention: At = fft(AW)
    return Pulse(At, AW, pulse.grid)
end

# Make LumpedElement callable for piping support
(element::LumpedElement)(pulse::Pulse) = apply(pulse, element)
(element::LumpedElement)(sol::Solution) = apply(Pulse(sol), element)
