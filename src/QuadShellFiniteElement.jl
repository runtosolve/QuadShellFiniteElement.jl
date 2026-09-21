module QuadShellFiniteElement

# Four-node quadrilateral Mindlin shell finite element with drilling degrees of freedom.
#
# The element follows the "4+2" formulation of S. Ádány (MATLAB routines
# ke_uv_4n_condens_from_12to8dof_num.m and ke_wt_4n_condens_from_18to12dof_num.m, 2026) and
# Moen & Ádány, "Thin shell finite element formulations implemented in open-source software",
# SSRC Annual Stability Conference, Louisville, 2025. The five shell displacement functions
# (u, v, w, θx, θy) are interpolated with the four bilinear corner shape functions plus two
# quadratic bubble functions N5 = 1 - ξ², N6 = 1 - η² whose amplitudes are condensed out
# statically, leaving 5 dof per corner node. A drilling rotation θz with a small penalty stiffness
# is then added, giving a 24 × 24 element in the local frame. Element matrices are formed in a
# local planar frame and rotated to global 3D coordinates, so the element can be used for
# arbitrarily oriented (folded, warped) shell meshes managed by Ferrite.jl.

using Ferrite, LinearAlgebra, Tensors

export IP4, IP6

"""
    DEFAULT_SHEAR_RELAXATION

Default shear relaxation factor `Cs` used by [`local_elastic_stiffness_matrix!`](@ref) and
[`assemble_global_Ke!`](@ref). The transverse shear stiffness of the element is scaled by
`1 / (1 + Cs * alpha)`, where `alpha` is the ratio of the element's shear to bending rotational
stiffness (Tessler–Hughes type relaxation, as in `TriShellFiniteElement.jl`). With `Cs = 0` the
element is exactly S. Ádány's MATLAB q42 element; its two condensed bubble modes remove the shear
locking in beam-like bending but not in general plate bending, so for thin plates the buckling
coefficient of a simply supported square plate meshed 10 elements across is 14% high at b/t = 105
and 3.6 times too high at b/t = 460. `Cs = 0.1` was calibrated on simply supported and clamped
plate buckling benchmarks (see `test/runtests.jl`): the buckling coefficient is within 1.5% of the
classical value from 8 to 24 elements across the plate width, independent of thickness, and the
beam-like bending benchmarks of Moen & Ádány (2025) are matched to 0.2% even on a 20 × 2 mesh.
"""
const DEFAULT_SHEAR_RELAXATION = 0.1

"""
    DEFAULT_DRILLING
    DEFAULT_DRILLING_GAMMA

Default treatment of the drilling (in-plane rotation) dof `θz`, used by
[`local_elastic_stiffness_matrix!`](@ref) and [`assemble_global_Ke!`](@ref):

- `:hughes_brezzi` (default): the Hughes & Brezzi (1989) drilling term
  `γ ∫ (θz − ½(∂v/∂x − ∂u/∂y))² dA`, `γ = drilling_gamma × G × t`, integrated with the membrane
  quadrature on the bilinear corner functions. It ties `θz` to the in-plane rotation of the
  membrane displacement field, is energy-free for rigid rotations and for the true in-plane rotation
  of a strip in Saint-Venant torsion, and lets the twisting moment of one plate strip pass across a
  fold line into the next strip. `DEFAULT_DRILLING_GAMMA = 1.0`; results are insensitive to `γ`
  between about 0.1 and 10 times `G t`.
- `:penalty`: S. Ádány's MATLAB q42 treatment, a diagonal penalty of 1/100 of the smallest
  rotational diagonal term on each `θz`.

The penalty on the absolute `θz` is fine for flat plates but wrong for folded or curved shells in
torsion: in Saint-Venant torsion every plate strip rotates in-plane at the rate `β′ h`, which the
penalty resists, so the torsion constant of a lipped C comes out 2 to 3 times too large and grows
with corner refinement. Removing the penalty instead makes every fold line behave as a free edge
for the twisting moment (J ≈ 12% low for a lipped C). The Hughes–Brezzi term gives J within 1% of
the exact Saint-Venant value (see `test/runtests.jl`, "torsion of folded strips").
"""
const DEFAULT_DRILLING = :hughes_brezzi
const DEFAULT_DRILLING_GAMMA = 1.0

# --------------------------------------------------------------------------------------------
# Interpolations on the reference quadrilateral ξ, η ∈ [-1, 1]
# --------------------------------------------------------------------------------------------

"""
    IP4()

Bilinear Lagrange interpolation on the reference quadrilateral (ξ, η ∈ [-1, 1]) with the Ferrite
corner ordering (-1,-1), (1,-1), (1,1), (-1,1). Used as the geometric interpolation of the
element and as the displacement interpolation for the geometric stiffness matrix.
"""
struct IP4 <: ScalarInterpolation{RefQuadrilateral, 1}
end

function Ferrite.reference_shape_value(ip::IP4, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number == 1 && return (1 - ξ₁) * (1 - ξ₂) / 4
    shape_number == 2 && return (1 + ξ₁) * (1 - ξ₂) / 4
    shape_number == 3 && return (1 + ξ₁) * (1 + ξ₂) / 4
    shape_number == 4 && return (1 - ξ₁) * (1 + ξ₂) / 4

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end

Ferrite.getnbasefunctions(::IP4) = 4
Ferrite.adjust_dofs_during_distribution(::IP4) = false

"""
    IP6()

The four bilinear corner shape functions of [`IP4`](@ref) plus the two quadratic bubble functions
`N5 = 1 - ξ²` and `N6 = 1 - η²` (Moen & Ádány 2025, Eq. 18–19). The bubble amplitudes are
internal element dofs that are removed by static condensation.
"""
struct IP6 <: ScalarInterpolation{RefQuadrilateral, 2}
end

function Ferrite.reference_shape_value(ip::IP6, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]

    shape_number <= 4 && return Ferrite.reference_shape_value(IP4(), ξ, shape_number)
    shape_number == 5 && return 1 - ξ₁^2
    shape_number == 6 && return 1 - ξ₂^2

    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end

Ferrite.getnbasefunctions(::IP6) = 6
Ferrite.adjust_dofs_during_distribution(::IP6) = false

# --------------------------------------------------------------------------------------------
# Constitutive matrices (isotropic, linear elastic)
# --------------------------------------------------------------------------------------------

function calculate_membrane_constitutive_matrix(E, ν, t)

    G = E / (2 * (1 + ν))

    D = [   E/(1-ν^2)   ν*E/(1-ν^2)   0.0
          ν*E/(1-ν^2)     E/(1-ν^2)   0.0
                  0.0           0.0     G]

    return D .* t

end

function calculate_bending_constitutive_matrix(E, ν, t)

    D_const = E * t^3 / (12 * (1 - ν^2))
    D = D_const * [1.0   ν    0.0
                    ν   1.0   0.0
                   0.0  0.0  (1-ν)/2]

    return D

end

function calculate_shear_constitutive_matrix(E, ν, t)

    G = E / (2 * (1 + ν))

    D = [5/6*G*t      0.0
             0.0  5/6*G*t]

    return D

end

# --------------------------------------------------------------------------------------------
# Element matrices in the local planar frame
#
# Node dof blocks: membrane [u, v]; plate [w, θx, θy] with
#   γxz = ∂w/∂x + θy,  γyz = ∂w/∂y - θx,  κx = ∂θy/∂x,  κy = -∂θx/∂y,  κxy = ∂θy/∂y - ∂θx/∂x
# i.e. the same conventions as TriShellFiniteElement.jl and S. Ádány's MATLAB code.
# The matrices are returned uncondensed for however many shape functions the CellValues carry
# (4 or 6); condensation happens in `local_elastic_stiffness_matrix!`.
# --------------------------------------------------------------------------------------------

"""
    calculate_element_membrane_stiffness_matrix(D, cv)

Membrane stiffness `∫ Bᵀ D B dA` for the `n` shape functions in `cv`, size `2n × 2n`, dof order
`u1 v1 u2 v2 …`. `cv` must be `reinit!`ed on the planar element coordinates.
"""
function calculate_element_membrane_stiffness_matrix(D, cv)

    n = getnbasefunctions(cv)
    ke = zeros(Float64, 2n, 2n)
    B = zeros(Float64, 3, 2n)

    for q_point in 1:getnquadpoints(cv)

        for i in 1:n
            dN = shape_gradient(cv, q_point, i)
            B[1, 2i-1] = dN[1]
            B[2, 2i]   = dN[2]
            B[3, 2i-1] = dN[2]
            B[3, 2i]   = dN[1]
        end

        ke += B' * D * B .* getdetJdV(cv, q_point)

    end

    return ke

end

"""
    calculate_element_bending_stiffness_matrix(D, cv)

Mindlin plate bending stiffness for the `n` shape functions in `cv`, size `3n × 3n`, dof order
`w1 θx1 θy1 w2 …`. The `w` rows and columns are zero.
"""
function calculate_element_bending_stiffness_matrix(D, cv)

    n = getnbasefunctions(cv)
    ke = zeros(Float64, 3n, 3n)
    B = zeros(Float64, 3, 3n)

    for q_point in 1:getnquadpoints(cv)

        for i in 1:n
            dN = shape_gradient(cv, q_point, i)
            # columns: w, θx, θy
            B[1, 3i]   =  dN[1]     # κx  =  ∂θy/∂x
            B[2, 3i-1] = -dN[2]     # κy  = -∂θx/∂y
            B[3, 3i-1] = -dN[1]     # κxy = -∂θx/∂x + ∂θy/∂y
            B[3, 3i]   =  dN[2]
        end

        ke += B' * D * B .* getdetJdV(cv, q_point)

    end

    return ke

end

"""
    calculate_element_shear_stiffness_matrix(D, cv)

Transverse shear stiffness for the `n` shape functions in `cv`, size `3n × 3n`, dof order
`w1 θx1 θy1 w2 …`.
"""
function calculate_element_shear_stiffness_matrix(D, cv)

    n = getnbasefunctions(cv)
    ke = zeros(Float64, 3n, 3n)
    B = zeros(Float64, 2, 3n)

    for q_point in 1:getnquadpoints(cv)

        for i in 1:n
            dN = shape_gradient(cv, q_point, i)
            N = shape_value(cv, q_point, i)
            # columns: w, θx, θy
            B[1, 3i-2] = dN[1]      # γxz = ∂w/∂x + θy
            B[1, 3i]   = N
            B[2, 3i-2] = dN[2]      # γyz = ∂w/∂y - θx
            B[2, 3i-1] = -N
        end

        ke += B' * D * B .* getdetJdV(cv, q_point)

    end

    return ke

end

"""
    static_condensation(k, n_active)

Eliminate the trailing internal dofs `n_active+1:end` of the symmetric matrix `k`:
`kaa - kai * kii⁻¹ * kia`.
"""
function static_condensation(k, n_active)

    inda = 1:n_active
    indi = n_active+1:size(k, 1)

    return k[inda, inda] - k[inda, indi] * (k[indi, indi] \ k[indi, inda])

end

# dof bookkeeping for the 4-node element
const NNODES = 4
const IND_UV_20 = [1, 2, 6, 7, 11, 12, 16, 17]                       # u, v in the 20-dof (5/node) matrix
const IND_WT_20 = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20]        # w, θx, θy in the 20-dof matrix
const IND_ROT_20 = [4, 5, 9, 10, 14, 15, 19, 20]                     # θx, θy in the 20-dof matrix
const MAP_20_TO_24 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
const IND_DRILL_24 = [6, 12, 18, 24]
# nodal-block order [u v w θx θy θz]×4  →  Ferrite field order (:u 12 dofs, then :θ 12 dofs)
const FIELD_ORDER_24 = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21,
                        4, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23, 24]

"""
    local_elastic_stiffness_matrix!(cv_m, cv_b, E, ν, t, x; Cs, drilling, drilling_gamma)
    local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, t, x; Cs, drilling, drilling_gamma)

24×24 element elastic stiffness matrix in the element local frame (node dofs ordered
`[u, v, w, θx, θy, θz]` per node). `x` holds the four planar node coordinates
(`Vector{Vec{2}}`, counter-clockwise). `cv_m` and `cv_b` are `CellValues` built with the six-function
interpolation [`IP6`](@ref) and the geometric interpolation [`IP4`](@ref); `cv_m` carries the
membrane quadrature (2×2 Gauss recommended) and `cv_b` the plate quadrature (3×3 Gauss
recommended). The second form builds them from quadrature rules and interpolations.

The membrane matrix is condensed from 12 to 8 dofs and the plate (bending + shear) matrix from 18
to 12 dofs, then the drilling dof `θz` is added: `drilling = :hughes_brezzi` (default) adds the
Hughes–Brezzi term with `γ = drilling_gamma × G t`, `drilling = :penalty` adds the MATLAB q42 diagonal
penalty, see [`DEFAULT_DRILLING`](@ref). `Cs` is the shear relaxation factor, see
[`DEFAULT_SHEAR_RELAXATION`](@ref).
"""
function local_elastic_stiffness_matrix!(cv_m::CellValues, cv_b::CellValues, E, ν, t, x; Cs = DEFAULT_SHEAR_RELAXATION,
                                         drilling = DEFAULT_DRILLING, drilling_gamma = DEFAULT_DRILLING_GAMMA)

    ##### membrane: 4 + 2 shape functions, 12 dof → 8 dof
    reinit!(cv_m, x)
    Dm = calculate_membrane_constitutive_matrix(E, ν, t)
    ke_m = calculate_element_membrane_stiffness_matrix(Dm, cv_m)
    ke_m = static_condensation(ke_m, 2 * NNODES)

    ##### bending + transverse shear: 4 + 2 shape functions, 18 dof → 12 dof
    reinit!(cv_b, x)
    Db = calculate_bending_constitutive_matrix(E, ν, t)
    Ds = calculate_shear_constitutive_matrix(E, ν, t)
    ke_b = calculate_element_bending_stiffness_matrix(Db, cv_b)
    ke_s = calculate_element_shear_stiffness_matrix(Ds, cv_b)

    if Cs != 0.0
        # shear / bending rotational stiffness ratio on the corner-node rotations
        rot = [2, 3, 5, 6, 8, 9, 11, 12]
        alpha = sum(diag(ke_s[rot, rot])) / sum(diag(ke_b[rot, rot]))
        ke_s = ke_s .* (1 / (1 + Cs * alpha))
    end

    ke_wt = static_condensation(ke_b + ke_s, 3 * NNODES)

    ##### 5 dof per node, 20 × 20
    k20 = zeros(Float64, 5 * NNODES, 5 * NNODES)
    k20[IND_UV_20, IND_UV_20] = ke_m
    k20[IND_WT_20, IND_WT_20] = ke_wt

    ##### add drilling dof, 24 × 24
    k24 = zeros(Float64, 6 * NNODES, 6 * NNODES)
    k24[MAP_20_TO_24, MAP_20_TO_24] = k20

    add_drilling_stiffness!(k24, cv_m, k20, E, ν, t, drilling, drilling_gamma)

    return k24

end

function local_elastic_stiffness_matrix!(qr_m, qr_b, ip4::IP4, ip6::IP6, E, ν, t, x; Cs = DEFAULT_SHEAR_RELAXATION,
                                         drilling = DEFAULT_DRILLING, drilling_gamma = DEFAULT_DRILLING_GAMMA)

    cv_m = CellValues(qr_m, ip6, ip4)
    cv_b = CellValues(qr_b, ip6, ip4)

    return local_elastic_stiffness_matrix!(cv_m, cv_b, E, ν, t, x; Cs, drilling, drilling_gamma)

end

"""
    add_drilling_stiffness!(k24, cv_m, k20, E, ν, t, drilling, drilling_gamma)

Add the drilling dof stiffness to the 24×24 local element matrix `k24` (node dofs `[u v w θx θy θz]`),
see [`DEFAULT_DRILLING`](@ref). `cv_m` must be `reinit!`ed on the planar element coordinates; its
first four shape functions are the bilinear corner functions used for the Hughes–Brezzi term.
`k20` is the 20×20 matrix without drilling dofs (used by the `:penalty` option).
"""
function add_drilling_stiffness!(k24, cv_m, k20, E, ν, t, drilling, drilling_gamma)

    if drilling == :penalty
        stif = minimum(diag(k20)[IND_ROT_20]) / 100
        for i in IND_DRILL_24
            k24[i, i] = stif
        end
    elseif drilling == :hughes_brezzi
        G = E / (2 * (1 + ν))
        γ = drilling_gamma * G * t
        row = zeros(Float64, 6 * NNODES)
        for q_point in 1:getnquadpoints(cv_m)
            dΩ = getdetJdV(cv_m, q_point)
            fill!(row, 0.0)
            for i in 1:NNODES
                dN = shape_gradient(cv_m, q_point, i)
                N = shape_value(cv_m, q_point, i)
                row[6(i - 1) + 1] = -0.5 * dN[2]      # u_i:  ½(∂v/∂x − ∂u/∂y)
                row[6(i - 1) + 2] = 0.5 * dN[1]       # v_i
                row[6(i - 1) + 6] = -N                # θz_i
            end
            k24 .+= γ .* (row * row') .* dΩ
        end
    else
        throw(ArgumentError("drilling must be :hughes_brezzi or :penalty, got $drilling"))
    end

    return k24

end

"""
    assemble_global_Ke!(Ke, dh, qr_m, qr_b, ip4, ip6, E, ν, t; Cs, drilling, drilling_gamma)

Assemble the global elastic stiffness matrix for a shell mesh with fields `:u` (3 translations)
and `:θ` (3 rotations) on `Quadrilateral` cells with 3D node coordinates. Element matrices are
formed in each element's local frame and rotated to global coordinates. `qr_m` and `qr_b` are the
membrane and plate quadrature rules (`QuadratureRule{RefQuadrilateral}(2)` and `(3)` recommended).
`Cs` is the shear relaxation factor, see [`DEFAULT_SHEAR_RELAXATION`](@ref); `drilling` and
`drilling_gamma` select the drilling dof treatment, see [`DEFAULT_DRILLING`](@ref).
"""
function assemble_global_Ke!(Ke, dh, qr_m, qr_b, ip4::IP4, ip6::IP6, E, ν, t; Cs = DEFAULT_SHEAR_RELAXATION,
                             drilling = DEFAULT_DRILLING, drilling_gamma = DEFAULT_DRILLING_GAMMA)

    cv_m = CellValues(qr_m, ip6, ip4)
    cv_b = CellValues(qr_b, ip6, ip4)

    assembler = start_assemble(Ke)
    for cell in CellIterator(dh)

        x_global = getcoordinates(cell)
        T = calculation_rotation_matrix(x_global)
        x_local = global_nodal_coords_to_planar_coords(x_global, T)

        ke_local = local_elastic_stiffness_matrix!(cv_m, cv_b, E, ν, t, x_local; Cs, drilling, drilling_gamma)

        # rotate element stiffness matrix back to global coordinates
        Te = rotation_matrix_for_element_stiffness_drilling(T)
        ke_global = Te * ke_local * Te'

        # reorder from nodal blocks to fields, Ferrite default
        ke_global = ke_global[FIELD_ORDER_24, FIELD_ORDER_24]

        assemble!(assembler, celldofs(cell), ke_global)
    end
    return Ke
end

# --------------------------------------------------------------------------------------------
# Geometric stiffness
# --------------------------------------------------------------------------------------------

_stress_at(s::Number, q_point) = s
_stress_at(s, q_point) = s[q_point]

"""
    calculate_element_geometric_stiffness_matrix(cv, σxx, σyy, τxy)

20×20 element geometric stiffness matrix in the local frame (node dofs `[u, v, w, θx, θy]`) from
the membrane stress resultants `σxx, σyy, τxy` (stress × thickness, force per length, positive in
tension). Each stress may be a scalar (constant over the element) or indexable by quadrature
point. `cv` is a `CellValues` with the four-node interpolation [`IP4`](@ref), already `reinit!`ed.
Follows Moen & Ádány 2025 Eq. 11–13: the nonlinear strains include the gradients of `u`, `v`
and `w` (Visy & Ádány 2017).
"""
function calculate_element_geometric_stiffness_matrix(cv, σxx, σyy, τxy)

    n = getnbasefunctions(cv)
    kuv = zeros(Float64, 2n, 2n)
    kwt = zeros(Float64, 3n, 3n)
    kg  = zeros(Float64, 5n, 5n)

    dNx_uv = zeros(Float64, 2, 2n)
    dNy_uv = zeros(Float64, 2, 2n)
    dNx_w  = zeros(Float64, 1, 3n)
    dNy_w  = zeros(Float64, 1, 3n)

    for q_point in 1:getnquadpoints(cv)

        for i in 1:n
            dN = shape_gradient(cv, q_point, i)
            dNx_uv[1, 2i-1] = dN[1]; dNx_uv[2, 2i] = dN[1]
            dNy_uv[1, 2i-1] = dN[2]; dNy_uv[2, 2i] = dN[2]
            dNx_w[1, 3i-2] = dN[1]
            dNy_w[1, 3i-2] = dN[2]
        end

        GGuvx  = dNx_uv' * dNx_uv
        GGuvy  = dNy_uv' * dNy_uv
        GGuvxy = dNx_uv' * dNy_uv + dNy_uv' * dNx_uv

        GGwtx  = dNx_w' * dNx_w
        GGwty  = dNy_w' * dNy_w
        GGwtxy = dNx_w' * dNy_w + dNy_w' * dNx_w

        # stresses are already in the element's local frame (same frame as dNdx)
        sx  = _stress_at(σxx, q_point)
        sy  = _stress_at(σyy, q_point)
        sxy = _stress_at(τxy, q_point)

        dV = getdetJdV(cv, q_point)
        kuv += (GGuvx * sx + GGuvy * sy + GGuvxy * sxy) * dV
        kwt += (GGwtx * sx + GGwty * sy + GGwtxy * sxy) * dV

    end

    induv = [5(i-1) + j for i in 1:n for j in 1:2]      # 1,2, 6,7, 11,12, 16,17
    indwt = [5(i-1) + j for i in 1:n for j in 3:5]      # 3,4,5, 8,9,10, ...
    kg[induv, induv] = kuv
    kg[indwt, indwt] = kwt

    return kg

end

"""
    assemble_global_Kg!(Kg, dh, qr, ip4, σXX, σYY, τXY)

Assemble the global geometric stiffness matrix. `σXX[i], σYY[i], τXY[i]` are the membrane stress
resultants (stress × thickness) of cell `i` in that cell's local frame, either scalars or
indexable by quadrature point of `qr` (`QuadratureRule{RefQuadrilateral}(2)` recommended). The
drilling rows and columns are zero. Sign convention: with compressive (negative) stresses the
buckling problem is `K φ = λ (-Kg) φ`.
"""
function assemble_global_Kg!(Kg, dh, qr, ip4::IP4, σXX, σYY, τXY)

    cv = CellValues(qr, ip4, ip4)
    assembler = start_assemble(Kg)
    i = 1
    for cell in CellIterator(dh)

        x_global = getcoordinates(cell)
        T = calculation_rotation_matrix(x_global)
        x_local = global_nodal_coords_to_planar_coords(x_global, T)

        reinit!(cv, x_local)

        kg_local = calculate_element_geometric_stiffness_matrix(cv, σXX[i], σYY[i], τXY[i])

        # expand 20×20 (no drilling) to 24×24 with zero rows/cols for θz
        kg_24 = zeros(Float64, 24, 24)
        kg_24[MAP_20_TO_24, MAP_20_TO_24] = kg_local

        # rotate element stiffness matrix back to global coordinates
        Te = rotation_matrix_for_element_stiffness_drilling(T)
        kg_global = Te * kg_24 * Te'

        # reorder from nodal blocks to fields, Ferrite default
        kg_global = kg_global[FIELD_ORDER_24, FIELD_ORDER_24]

        assemble!(assembler, celldofs(cell), kg_global)
        i += 1
    end
    return Kg
end

# --------------------------------------------------------------------------------------------
# Membrane stress recovery
# --------------------------------------------------------------------------------------------

"""
    element_membrane_stresses(dh, u, ip4, E, ν, t; qr = QuadratureRule{RefQuadrilateral}(1))

Membrane stresses `σxx, σyy, τxy` (force per area, positive in tension) in each cell's local
frame, evaluated from the global solution vector `u` with the bilinear interpolation at the
points of `qr`. With the default one-point rule each entry is a scalar at the element centre;
with more points each entry is a vector indexed by quadrature point, ready for
[`assemble_global_Kg!`](@ref) after multiplying by the thickness. Returns `(σXX, σYY, τXY)`.
"""
function element_membrane_stresses(dh, u, ip4::IP4, E, ν, t; qr = QuadratureRule{RefQuadrilateral}(1))

    D = calculate_membrane_constitutive_matrix(E, ν, t) ./ t
    cv = CellValues(qr, ip4, ip4)
    nq = getnquadpoints(cv)
    ncells = getncells(dh.grid)

    S = nq == 1 ? Float64 : Vector{Float64}
    σXX = Vector{S}(undef, ncells)
    σYY = Vector{S}(undef, ncells)
    τXY = Vector{S}(undef, ncells)

    B = zeros(Float64, 3, 8)
    ul = zeros(Float64, 8)

    for cell in CellIterator(dh)

        x_global = getcoordinates(cell)
        T = calculation_rotation_matrix(x_global)
        x_local = global_nodal_coords_to_planar_coords(x_global, T)
        reinit!(cv, x_local)

        # in-plane nodal translations in the local frame (:u field dofs come first)
        cd = celldofs(cell)
        for i in 1:NNODES
            ug = [u[cd[3(i-1) + k]] for k in 1:3]
            uloc = T' * ug
            ul[2i-1] = uloc[1]
            ul[2i]   = uloc[2]
        end

        sx = zeros(nq); sy = zeros(nq); sxy = zeros(nq)
        for q_point in 1:nq
            for i in 1:NNODES
                dN = shape_gradient(cv, q_point, i)
                B[1, 2i-1] = dN[1]
                B[2, 2i]   = dN[2]
                B[3, 2i-1] = dN[2]
                B[3, 2i]   = dN[1]
            end
            σ = D * B * ul
            sx[q_point] = σ[1]; sy[q_point] = σ[2]; sxy[q_point] = σ[3]
        end

        c = cellid(cell)
        if nq == 1
            σXX[c] = sx[1]; σYY[c] = sy[1]; τXY[c] = sxy[1]
        else
            σXX[c] = sx; σYY[c] = sy; τXY[c] = sxy
        end
    end

    return σXX, σYY, τXY

end

# --------------------------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------------------------

"""
    assemble_pressure_load!(f, dh, qr, ip4, p)

Add the consistent nodal forces of a uniform pressure `p` (force per area) acting along each
element's local normal `j3` (see [`calculation_rotation_matrix`](@ref)) to the global load
vector `f`. `qr` is a `QuadratureRule{RefQuadrilateral}` (2×2 integrates the bilinear
shape functions exactly). Only the `:u` translational dofs receive load.
"""
function assemble_pressure_load!(f, dh, qr, ip4::IP4, p)

    cv = CellValues(qr, ip4, ip4)

    for cell in CellIterator(dh)

        x_global = getcoordinates(cell)
        T = calculation_rotation_matrix(x_global)
        x_local = global_nodal_coords_to_planar_coords(x_global, T)
        reinit!(cv, x_local)

        normal = T[:, 3]
        cd = celldofs(cell)

        for q_point in 1:getnquadpoints(cv)
            dV = getdetJdV(cv, q_point)
            for i in 1:NNODES
                N = shape_value(cv, q_point, i)
                for k in 1:3
                    f[cd[3(i-1) + k]] += N * p * normal[k] * dV
                end
            end
        end
    end

    return f

end

# --------------------------------------------------------------------------------------------
# Local ↔ global geometry
# --------------------------------------------------------------------------------------------

"""
    calculation_rotation_matrix(node)

3×3 rotation matrix `T = [j1 j2 j3]` of the element local frame for the four global node
coordinates `node` (counter-clockwise), following `ct_4node_g2e.m`: `j3` is normal to the average
plane spanned by the two mid-side connecting vectors (P23 - P41) and (P34 - P12), `j1` is along
(P23 - P41), and `j2 = j3 × j1`. Columns of `T` are the local axes in global components, so a
global vector `g` has local components `T' * g`.
"""
function calculation_rotation_matrix(node)

    P1, P2, P3, P4 = node[1], node[2], node[3], node[4]

    P12 = (P1 + P2) / 2; P34 = (P3 + P4) / 2
    P23 = (P2 + P3) / 2; P41 = (P4 + P1) / 2

    norm_vec = cross(P23 - P41, P34 - P12)
    j3 = norm_vec / norm(norm_vec)
    j1 = (P23 - P41) / norm(P23 - P41)      # already perpendicular to j3
    j2 = cross(j3, j1)

    T = [j1 j2 j3]

    return T

end

"""
    global_nodal_coords_to_planar_coords(cell_nodes_global, T)

Planar (local x, y) coordinates of the four nodes relative to the element centroid, i.e. the
nodes projected onto the element's average plane. Returns a `Vector{Vec{2}}` for `reinit!`.
"""
function global_nodal_coords_to_planar_coords(cell_nodes_global, T)

    P0 = sum(cell_nodes_global) / length(cell_nodes_global)

    cell_nodes_local = [begin
                            Pl = T' * (P - P0)
                            Tensors.Vec((Pl[1], Pl[2]))
                        end for P in cell_nodes_global]

    return cell_nodes_local

end

"""
    rotation_matrix_for_element_stiffness_drilling(T3)

24×24 block-diagonal transformation with `T3` on every translation and rotation block
(nodal dof order `[u v w θx θy θz]`), so that `ke_global = Te * ke_local * Te'`.
"""
function rotation_matrix_for_element_stiffness_drilling(T3)

    T = Matrix(1.0I, 24, 24)

    for i in 1:NNODES
        ind = 6(i-1) .+ (1:3)
        T[ind, ind] = T3
        ind = 6(i-1) .+ (4:6)
        T[ind, ind] = T3
    end

    return T

end

"""
    rotation_matrix_for_element_stiffness_no_drilling(T3)

20×20 transformation for the 5-dof-per-node matrices (`[u v w θx θy]`): `T3` on the
translations and its upper-left 2×2 block on the in-plane rotations.
"""
function rotation_matrix_for_element_stiffness_no_drilling(T3)

    T2 = T3[1:2, 1:2]
    T = Matrix(1.0I, 20, 20)

    for i in 1:NNODES
        ind = 5(i-1) .+ (1:3)
        T[ind, ind] = T3
        ind = 5(i-1) .+ (4:5)
        T[ind, ind] = T2
    end

    return T

end

end # module QuadShellFiniteElement
