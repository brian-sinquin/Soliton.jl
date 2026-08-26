"""
Enzyme.jl reverse-mode AD support for Soliton.jl's FFTW boundary.

`plan_fft`/`plan_ifft` (`AbstractFFTs.Plan`) call into the FFTW C library via
`ccall`, which Enzyme cannot differentiate through automatically (see
`docs/src/dev/adjoint_ad.md`). This registers the `EnzymeRules` pair for
`LinearAlgebra.mul!(y, plan, x)` that makes every nonlinear-step function in
`src/nonlinearity.jl` (they all route through `PhysicsModel.to_freq`/
`to_time`) differentiable with `Enzyme.autodiff`.

# Math

An FFT plan is a complex-linear map `ℂⁿ → ℂⁿ`, so the reverse-mode pullback of
`y = P*x` is `x̄ += Pᴴ ȳ` — the standard ChainRules/Zygote convention for
complex-linear operators driving a real-valued loss, and what Enzyme's
treatment of `Complex` activity as a pair of reals expects.

`Pᴴ` needs no explicit `N` anywhere. For the unnormalized DFT matrix `F`
(`F[j,k] = exp(-2πi·jk/N)`), `Fᴴ = F̄ᵀ = B`, the unnormalized *backward*
transform: conjugating and transposing a symmetric DFT matrix only flips its
sign convention. See [`_adjoint_plan`](@ref) for how that is recovered from an
arbitrary (possibly `ScaledPlan`-wrapped) plan.

This is the same quantity `AbstractFFTs` exposes as `p'` (its `AdjointPlan`),
factored so it can be applied with `mul!` into an existing buffer — `p'` itself
allocates, since `AdjointPlan` does not support `mul!`. `test/test_enzyme.jl`
checks the two agree, which makes upstream a maintained oracle for our
normalization.

Forward mode is supported too: an FFT plan is linear, so a tangent obeys the
same map as the primal (`ẏ = P ẋ`). Reverse mode tapes the whole propagation to
produce a gradient, so for a design problem with only a handful of parameters —
e.g. optimizing `β₂` alone — forward mode is cheaper in both time and memory.

Verified against finite differences on `model.to_freq` and `model.to_time` in
isolation, on the full `_spm` nonlinear step, and end-to-end through a whole
fixed-step `SSFM` propagation — see `test/test_enzyme.jl`.

# Usage note

Differentiating a function that takes a `PhysicsModel` requires marking the
model `Duplicated(model, Enzyme.make_zero(model))`, not `Const(model)` — even
when the model's own gradient is never read. `PhysicsModel`'s buffers
(`buf_t1`, `buf_t2`, `buf_f1`) are mutated as scratch space for the active
pulse envelope, and a `Const` model gives Enzyme no shadow memory to carry
those intermediate adjoints: it either throws `EnzymeRuntimeActivityError`, or
— if that is papered over with `set_runtime_activity` — silently returns an
all-zero gradient with no error at all.
"""
module SolitonEnzymeExt

using Enzyme
using AbstractFFTs: AbstractFFTs, Plan, ScaledPlan
import LinearAlgebra: mul!

const EnzymeRules = Enzyme.EnzymeRules
const EC = Enzyme.EnzymeCore

"""
    _unscale(p) -> (base, scale)

Split a plan into an unnormalized `base` plan and a scalar `scale` with
`p == scale * base`. Plain plans are already unnormalized; `ScaledPlan`s can
nest (`inv` of a `ScaledPlan` produces one), so the scales multiply.

Dispatch resolves the recursion at compile time, so this is allocation-free and
type-stable for any concrete plan type.
"""
_unscale(p::Plan) = (p, 1)
function _unscale(p::ScaledPlan)
    base, scale = _unscale(p.p)
    return base, scale * p.scale
end

"""
    _adjoint_plan(p) -> (padj, scale)

Factor the Hermitian adjoint of the linear map `p` into a plan application and
a real rescaling: `p' * v == scale * (padj * v)`.

`p` splits as `scale * base` with `base` unnormalized, and `pᴴ == scale * baseᴴ`
because `scale` is real. `baseᴴ` is the unnormalized transform in the opposite
direction (see this module's docstring), which is exactly what
`AbstractFFTs.inv(base)` is once its `1/N` normalization is stripped back off.

Deriving the factor this way — rather than from `length(y)` — keeps the rule
correct for plans that transform only *some* dimensions of a multidimensional
array, such as the vectorial solver's `plan_ifft(tmp, 1)` over an `N x 2`
field, where the normalization is the transform length rather than the array
length. `AbstractFFTs` caches `inv` on the plan itself, so repeated calls cost
a field read rather than a new plan.

!!! warning "Complex-to-complex plans only"
    Assumes `p` maps `ℂⁿ → ℂⁿ` (`plan_fft`/`plan_ifft` and their scaled
    inverses), which is all `PhysicsModel` ever builds. Real-input plans
    (`plan_rfft`/`plan_irfft`) change the array size and need boundary-aware
    scaling at the Nyquist bin — `AbstractFFTs`' `RFFTAdjointStyle` handles
    that, this does not, and would be silently wrong rather than erroring.
    Adding real-transform support means deferring to `p'` for those styles.
"""
function _adjoint_plan(p::Plan)
    base, scale = _unscale(p)
    padj, _ = _unscale(inv(base))
    return padj, scale
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    y::EC.Annotation{<:AbstractVecOrMat},
    plan::EC.Annotation{<:Plan},
    x::EC.Annotation{<:AbstractVecOrMat},
) where {RT}
    mul!(y.val, plan.val, x.val)

    primal = EnzymeRules.needs_primal(config) ? y.val : nothing
    shadow = if EnzymeRules.needs_shadow(config) && !(y isa EC.Const)
        y.dval
    else
        nothing
    end

    # No tape. An earlier version allocated the reverse pass's scratch buffer
    # here, which removed one allocation per adjoint — but a tape entry is held
    # from `augmented_primal` until its matching `reverse`, so that traded a
    # transient allocation for one retained array *per taped `mul!`*. Invisible
    # at test sizes; at N=2^14 over 10k steps it is several GB of extra live
    # tape, and tape memory is already the binding constraint on differentiating
    # realistic runs. `reverse` allocates its own scratch instead.
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    ::Nothing,
    y::EC.Annotation{<:AbstractVecOrMat},
    plan::EC.Annotation{<:Plan},
    x::EC.Annotation{<:AbstractVecOrMat},
) where {RT}
    if !(x isa EC.Const) && !(y isa EC.Const)
        ybar = y.dval
        padj, scale = _adjoint_plan(plan.val)
        # Transient scratch: freed as soon as this adjoint returns, so peak
        # extra memory is one array rather than one per taped call. Applying
        # `padj` with `mul!` and folding `scale` into the accumulation still
        # saves a `ScaledPlan` allocation and a full pass over the array
        # compared with `x̄ .+= (scale * padj) * ȳ`.
        tmp = similar(x.dval)
        mul!(tmp, padj, ybar)
        x.dval .+= scale .* tmp
        # `mul!` overwrites `y` in full, so `y`'s incoming adjoint is entirely
        # consumed here and must not leak back into any earlier use of `y`.
        ybar .= 0
    end
    return (nothing, nothing, nothing)
end

"""
Forward-mode rule. An FFT plan is a linear map, so the tangent propagates
through the identical transform: `ẏ = P ẋ`. A `Const` `x` carries no tangent,
so `y`'s tangent is zeroed rather than left stale from a previous call.
"""
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    y::EC.Annotation{<:AbstractVecOrMat},
    plan::EC.Annotation{<:Plan},
    x::EC.Annotation{<:AbstractVecOrMat},
) where {RT}
    mul!(y.val, plan.val, x.val)

    if !(y isa EC.Const)
        if x isa EC.Const
            fill!(y.dval, zero(eltype(y.dval)))
        else
            mul!(y.dval, plan.val, x.dval)
        end
    end

    # Return shape is dictated by `EnzymeRules.forward_rule_return_type`.
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? y.val : nothing
    elseif EnzymeRules.needs_primal(config)
        return EC.Duplicated(y.val, y.dval)
    else
        return y.dval
    end
end

# Only width-1 (`Duplicated`, not `BatchDuplicated`) reverse mode is covered;
# batched seeds fall through to Enzyme's own handling of the FFTW `ccall`.

end # module SolitonEnzymeExt
