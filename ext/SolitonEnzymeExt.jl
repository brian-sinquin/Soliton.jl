"""
Enzyme.jl reverse-mode AD support for Soliton.jl's FFTW boundary.

`plan_fft`/`plan_ifft` (`AbstractFFTs.Plan`) call into the FFTW C library via
`ccall`, which Enzyme cannot differentiate through automatically (see
`docs/src/dev/adjoint_ad.md`). This registers the `EnzymeRules` pair for
`LinearAlgebra.mul!(y, plan, x)` that makes every nonlinear-step function in
`src/nonlinearity.jl` (they all route through `PhysicsModel.to_freq`/
`to_time`) differentiable with `Enzyme.autodiff`.

# Math

For an FFT plan `P` (a C-linear map from `ℂⁿ → ℂⁿ`), the reverse-mode
pullback of `y = P*x` needs `x̄ += Pᴴx̄` where `Pᴴ` is `P`'s Hermitian
adjoint — this is the standard convention used by ChainRules/Zygote-style
AD for complex-linear operators driving a real-valued loss, and matches
Enzyme's own treatment of `Complex` activity as a pair of reals.

`AbstractFFTs.inv(P)` gives the exact inverse of `P`, and for the DFT:

  - An unnormalized transform (`plan_fft`/`plan_bfft`, no scale factor)
    satisfies `Pᴴ*P == N*I`, so `Pᴴ = N * inv(P)`.
  - An `AbstractFFTs.ScaledPlan` (`plan_ifft`, already carrying the `1/N`
    normalization) satisfies `Pᴴ*P == (1/N)*I`, so `Pᴴ = inv(P) / N`.

Both cases were verified against finite differences (relative error at the
1e-7 level) on both `model.to_freq` and `model.to_time` in isolation, and on
the full `_spm` nonlinear step end-to-end.

# Usage note

Differentiating a function that takes a `PhysicsModel` requires marking the
model `Duplicated(model, Enzyme.make_zero(model))`, not `Const(model)`:
`PhysicsModel`'s buffers (`buf_t1`, `buf_f1`, ...) are mutated as scratch
space for the active pulse envelope, and Enzyme's static activity analysis
cannot prove that use non-differentiable when the whole struct is `Const`
(it either throws `EnzymeRuntimeActivityError`, or — if that's papered over
with `set_runtime_activity` instead of `Duplicated` — silently returns an
all-zero gradient with no error at all).
"""
module SolitonEnzymeExt

using Enzyme
using AbstractFFTs
import LinearAlgebra: mul!

const EnzymeRules = Enzyme.EnzymeRules
const EC = Enzyme.EnzymeCore

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    y::EC.Annotation{<:AbstractVector},
    plan::EC.Annotation{<:AbstractFFTs.Plan},
    x::EC.Annotation{<:AbstractVector},
) where {RT}
    mul!(y.val, plan.val, x.val)
    primal = EnzymeRules.needs_primal(config) ? y.val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    ::Any,
    y::EC.Annotation{<:AbstractVector},
    plan::EC.Annotation{<:AbstractFFTs.Plan},
    x::EC.Annotation{<:AbstractVector},
) where {RT}
    if !(x isa EC.Const)
        ybar = y.dval
        N = length(ybar)
        p = plan.val
        adjoint_plan = p isa AbstractFFTs.ScaledPlan ? inv(p) * (1 / N) : inv(p) * N
        x.dval .+= adjoint_plan * ybar
        ybar .= 0
    end
    return (nothing, nothing, nothing)
end

end # module SolitonEnzymeExt
