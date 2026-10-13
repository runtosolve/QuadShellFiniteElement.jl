# --------------------------------------------------------------------------------------------
# J2 (von Mises) plane-stress plasticity with isotropic hardening and layered through-thickness
# integration of the shell section.
#
# This file is shared verbatim between QuadShellFiniteElement.jl and TriShellFiniteElement.jl.
#
# Stress and strain are plane-stress Voigt vectors [σx, σy, τxy] and [εx, εy, γxy] (engineering
# shear strain). The yield function is f = σ_vm − σ_y(ε̄p) with σ_vm² = σᵀ P σ,
# P = [1 −1/2 0; −1/2 1 0; 0 0 3]; the associated flow rule Δεp = Δγ P σ = Δγ (3/2) s (s the
# deviatoric stress) gives the equivalent plastic strain increment Δε̄p = Δγ σ_vm. The return mapping is the plane-stress projected
# algorithm of Simo & Taylor (1986) / Simo & Hughes (1998, Sec. 3.5) written for a general
# piecewise-linear hardening curve, with the algorithmically consistent tangent.
# --------------------------------------------------------------------------------------------

using StaticArrays

const P_VM = @SMatrix [1.0 -0.5 0.0; -0.5 1.0 0.0; 0.0 0.0 3.0]
const I3 = SMatrix{3,3,Float64,9}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

"""
    IsotropicHardening(εp, σ)
    IsotropicHardening(σy)
    IsotropicHardening(σy, H; εp_max = 10.0)

Piecewise-linear isotropic hardening curve: true (von Mises) yield stress `σ[i]` at equivalent
plastic strain `εp[i]`, with `εp[1] == 0` so that `σ[1]` is the initial yield stress. Between the
points the yield stress is interpolated linearly; beyond the last point it is constant (perfectly
plastic), the same convention as the ABAQUS `*PLASTIC` table. `IsotropicHardening(σy)` is
elastic–perfectly plastic and `IsotropicHardening(σy, H)` is linear hardening with plastic modulus
`H` up to `εp_max`.
"""
struct IsotropicHardening
    εp::Vector{Float64}
    σ::Vector{Float64}
    function IsotropicHardening(εp, σ)
        n = length(εp)
        n == length(σ) && n ≥ 1 || throw(ArgumentError("εp and σ must have the same nonzero length"))
        εp[1] == 0 || throw(ArgumentError("the first plastic strain must be 0 (initial yield)"))
        all(diff(εp) .> 0) || throw(ArgumentError("plastic strains must be strictly increasing"))
        all(σ .> 0) || throw(ArgumentError("yield stresses must be positive"))
        return new(collect(Float64, εp), collect(Float64, σ))
    end
end

IsotropicHardening(σy::Real) = IsotropicHardening([0.0], [float(σy)])

function IsotropicHardening(σy::Real, H::Real; εp_max = 10.0)
    H == 0 && return IsotropicHardening(σy)
    H > 0 || throw(ArgumentError("the plastic modulus H must be nonnegative"))
    return IsotropicHardening([0.0, float(εp_max)], [float(σy), float(σy + H * εp_max)])
end

"""
    yield_stress(h::IsotropicHardening, ε̄p) -> (σy, H)

Yield stress and plastic modulus `H = dσy/dε̄p` at the equivalent plastic strain `ε̄p`.
"""
function yield_stress(h::IsotropicHardening, ε̄p)
    εp = h.εp; σ = h.σ
    n = length(εp)
    (n == 1 || ε̄p >= εp[end]) && return σ[end], 0.0
    i = max(searchsortedlast(εp, ε̄p), 1)
    H = (σ[i+1] - σ[i]) / (εp[i+1] - εp[i])
    return σ[i] + H * (ε̄p - εp[i]), H
end

"""
    J2PlaneStress(E, ν, hardening::IsotropicHardening)
    J2PlaneStress(E, ν, σy)

Linear elastic – von Mises plastic plane-stress material with isotropic hardening.
"""
struct J2PlaneStress
    E::Float64
    ν::Float64
    hardening::IsotropicHardening
    C::SMatrix{3,3,Float64,9}
end

function J2PlaneStress(E, ν, hardening::IsotropicHardening)
    C = E / (1 - ν^2) * @SMatrix [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1-ν)/2]
    return J2PlaneStress(E, ν, hardening, C)
end

J2PlaneStress(E, ν, σy::Real) = J2PlaneStress(E, ν, IsotropicHardening(σy))

"""
    LayerState(εp, ε̄p)

Plastic state of one through-thickness layer at one integration point: plastic strain `εp`
(Voigt, engineering shear) and equivalent plastic strain `ε̄p`.
"""
struct LayerState
    εp::SVector{3,Float64}
    ε̄p::Float64
end

LayerState() = LayerState(zero(SVector{3,Float64}), 0.0)

von_mises(σ::SVector{3}) = sqrt(max(dot(σ, P_VM * σ), 0.0))

"""
    return_mapping(mat::J2PlaneStress, ε, state::LayerState) -> (σ, Cep, new_state)

Stress, consistent elastic–plastic tangent `dσ/dε` and updated state for the total strain `ε`
(Voigt `SVector{3}`) starting from the converged `state`. Elastic steps return the elastic `C`.
"""
function return_mapping(mat::J2PlaneStress, ε::SVector{3,Float64}, st::LayerState; tol = 1e-10, maxiter = 60)
    C = mat.C
    P = P_VM
    σtr = C * (ε - st.εp)
    svm_tr = von_mises(σtr)
    σy0, _ = yield_stress(mat.hardening, st.ε̄p)
    if svm_tr <= σy0 * (1 + 1e-12)
        return σtr, C, st
    end

    CP = C * P
    # residual r(Δγ) = σ_vm(σ(Δγ)) − σ_y(ε̄p_n + Δγ σ_vm) and its derivative
    function residual(Δγ)
        A = inv(I3 + Δγ * CP)
        σ = A * σtr
        svm = von_mises(σ)
        ε̄p = st.ε̄p + Δγ * svm
        σy, H = yield_stress(mat.hardening, ε̄p)
        Ξ = A * C
        Pσ = P * σ
        dσ = -(Ξ * Pσ)
        dsvm = svm > 0 ? dot(Pσ, dσ) / svm : 0.0
        rγ = dsvm - H * (svm + Δγ * dsvm)
        return svm - σy, rγ, σ, svm, ε̄p, H, Ξ, Pσ
    end

    # bracket: r(0) > 0, find Δγ_hi with r < 0
    lo = 0.0
    hi = 1.5 * (svm_tr - σy0) / (mat.E * σy0)      # estimate of the plastic multiplier (units 1/stress)
    hi = max(hi, 1e-12 / σy0)
    r_hi = residual(hi)[1]
    nb = 0
    while r_hi > 0
        hi *= 4
        r_hi = residual(hi)[1]
        nb += 1
        nb > 60 && error("return_mapping: could not bracket the plastic multiplier")
    end

    Δγ = hi / 2
    out = residual(Δγ)
    r = out[1]
    for iter in 1:maxiter
        r, rγ = out[1], out[2]
        abs(r) <= tol * σy0 && break
        # maintain the bracket and take a safeguarded Newton step
        r > 0 ? (lo = Δγ) : (hi = Δγ)
        Δγ_new = Δγ - r / rγ
        if !(lo < Δγ_new < hi) || !isfinite(Δγ_new)
            Δγ_new = 0.5 * (lo + hi)
        end
        Δγ = Δγ_new
        out = residual(Δγ)
        iter == maxiter && @warn "return_mapping: no convergence, |r|/σy = $(abs(out[1]) / σy0)"
    end
    r, rγ, σ, svm, ε̄p, H, Ξ, Pσ = out

    ΞPσ = Ξ * Pσ
    rε = (1 - Δγ * H) / svm * ΞPσ          # ∂r/∂ε at fixed Δγ  (= (1 − Δγ H)(Pσ)ᵀΞ/σ_vm, Ξ symmetric)
    Cep = Ξ + (ΞPσ * rε') / rγ
    return σ, Cep, LayerState(st.εp + Δγ * Pσ, ε̄p)
end

"""
    thickness_layers(t, n; rule = :gauss) -> (z, w)

Through-thickness integration points `z ∈ (−t/2, t/2)` and weights `w` (summing to `t`).
`rule = :gauss` is the `n`-point Gauss–Legendre rule, `rule = :simpson` the `n`-point Simpson rule
(`n` odd, includes the surfaces and the midplane, the ABAQUS default for shells with `n = 5`).
"""
function thickness_layers(t, n; rule = :gauss)
    if rule == :gauss
        qr = QuadratureRule{RefLine}(n)
        z = [t / 2 * p[1] for p in Ferrite.getpoints(qr)]
        w = [t / 2 * wq for wq in Ferrite.getweights(qr)]
    elseif rule == :simpson
        (n ≥ 3 && isodd(n)) || throw(ArgumentError("Simpson's rule needs an odd number of layers ≥ 3"))
        h = t / (n - 1)
        z = [-t / 2 + h * (k - 1) for k in 1:n]
        w = [h / 3 * (k == 1 || k == n ? 1.0 : (iseven(k) ? 4.0 : 2.0)) for k in 1:n]
    else
        throw(ArgumentError("rule must be :gauss or :simpson, got $rule"))
    end
    return z, w
end

"""
    ShellSection(mat::J2PlaneStress, t; nlayers = 5, rule = :gauss)

Elastic–plastic shell section: material, thickness and the through-thickness layers used to
integrate the stress resultants, see [`thickness_layers`](@ref).
"""
struct ShellSection
    mat::J2PlaneStress
    t::Float64
    z::Vector{Float64}
    w::Vector{Float64}
end

function ShellSection(mat::J2PlaneStress, t; nlayers = 5, rule = :gauss)
    z, w = thickness_layers(t, nlayers, rule = rule)
    return ShellSection(mat, float(t), z, w)
end

nlayers(sec::ShellSection) = length(sec.z)

"""
    section_response(sec, εm, κ, states, new_states) -> (N, M, Dmm, Dmb, Dbb)

Integrate the layer stresses through the thickness for the membrane strain `εm` and curvature `κ`
(strain at level `z` is `εm + z κ`): membrane forces `N = ∫σ dz`, moments `M = ∫zσ dz` and the
section tangent blocks `dN/dεm`, `dN/dκ = dM/dεm`, `dM/dκ`. The updated layer states are written
to `new_states`; `states` is left untouched.
"""
function section_response(sec::ShellSection, εm::SVector{3,Float64}, κ::SVector{3,Float64},
                          states::AbstractVector{LayerState}, new_states::AbstractVector{LayerState})
    N = zero(SVector{3,Float64}); M = zero(SVector{3,Float64})
    Dmm = zero(SMatrix{3,3,Float64,9}); Dmb = zero(SMatrix{3,3,Float64,9}); Dbb = zero(SMatrix{3,3,Float64,9})
    @inbounds for k in eachindex(sec.z)
        z = sec.z[k]; w = sec.w[k]
        σ, Cep, st = return_mapping(sec.mat, εm + z * κ, states[k])
        new_states[k] = st
        N += w * σ
        M += (w * z) * σ
        Dmm += w * Cep
        Dmb += (w * z) * Cep
        Dbb += (w * z^2) * Cep
    end
    return N, M, Dmm, Dmb, Dbb
end

"""
    layer_stresses(sec, εm, κ, states) -> Vector{SVector{3}}

Stresses in each layer for the given section strains and converged states (no state update).
"""
function layer_stresses(sec::ShellSection, εm::SVector{3,Float64}, κ::SVector{3,Float64}, states::AbstractVector{LayerState})
    return [return_mapping(sec.mat, εm + sec.z[k] * κ, states[k])[1] for k in eachindex(sec.z)]
end
