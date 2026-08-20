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

Assumes `p` maps `ℂⁿ → ℂⁿ` (`plan_fft`/`plan_ifft` and their scaled inverses),
which is all `PhysicsModel` ever builds; real-input plans (`plan_rfft`) change
the array size and would need a different adjoint.
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

    # Scratch for the reverse pass's `padj * ȳ`, so `reverse` can `mul!` into a
    # buffer instead of allocating a fresh result array per adjoint. `x`'s
    # activity is known statically here, so a `Const` `x` — nothing to
    # accumulate into — allocates nothing at all.
    tape = x isa EC.Const ? nothing : similar(x.val)

    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::EC.Const{typeof(mul!)},
    ::Type{RT},
    tape,
    y::EC.Annotation{<:AbstractVecOrMat},
    plan::EC.Annotation{<:Plan},
    x::EC.Annotation{<:AbstractVecOrMat},
) where {RT}
    if !(x isa EC.Const) && !(y isa EC.Const)
        ybar = y.dval
        padj, scale = _adjoint_plan(plan.val)
        mul!(tape, padj, ybar)
        x.dval .+= scale .* tape
        # `mul!` overwrites `y` in full, so `y`'s incoming adjoint is entirely
        # consumed here and must not leak back into any earlier use of `y`.
        ybar .= 0
    end
    return (nothing, nothing, nothing)
end

# Only width-1 (`Duplicated`, not `BatchDuplicated`) reverse mode is covered;
# batched seeds fall through to Enzyme's own handling of the FFTW `ccall`.

end # module SolitonEnzymeExt
