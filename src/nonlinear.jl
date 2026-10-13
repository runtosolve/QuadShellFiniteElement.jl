# --------------------------------------------------------------------------------------------
# Geometrically nonlinear, elastic–plastic quadrilateral shell element.
#
# Total Lagrangian formulation in each element's initial local frame (the frame of the undeformed,
# possibly imperfect, mesh): the membrane strain is the Green–Lagrange strain of the midsurface
#   εx  = u,x + ½ (u,x² + v,x² + w,x²)
#   εy  = v,y + ½ (u,y² + v,y² + w,y²)
#   γxy = u,y + v,x + (u,x u,y + v,x v,y + w,x w,y)
# (gradients of all three displacement components, as in the geometric stiffness matrix of
# Visy & Ádány 2017 / Moen & Ádány 2025), the curvatures are linear in the rotations
# (κx = θy,x, κy = −θx,y, κxy = θy,y − θx,x) and the transverse shear strains and the drilling
# term are linear elastic. The membrane and bending stresses are coupled through the layered
# elastic–plastic section (plasticity.jl).
#
# All five displacement functions keep the 4 + 2 interpolation (bilinear corner functions plus
# the two bubbles N5 = 1 − ξ², N6 = 1 − η²). The ten bubble amplitudes are element-internal
# variables: they are condensed out of the element tangent and residual at every evaluation and
# updated from the stored linearization before the next evaluation, as in enhanced-assumed-strain
# elements (Simo & Rifai 1990). At zero displacement the condensed tangent is exactly the linear
# element stiffness of `local_elastic_stiffness_matrix!` on rectangular elements (the membrane
# block is integrated with the plate quadrature `qr_b`, which is exact for rectangles, instead of
# the 2×2 membrane rule).
#
# Element dof order in the local frame: [u v w θx θy θz] per node (24 dofs); internal dofs
# [u v w θx θy] for N5 then N6 (10 dofs). The uncondensed working vector has 30 dofs:
# [u v w θx θy] for each of the six functions.
# --------------------------------------------------------------------------------------------

const NQ_INTERNAL = 10

"""
    QuadShellElementData

Per-element data of a [`PlasticShellModel`](@ref): initial local frame, shape function gradients
and weights at the plate quadrature points, the linear elastic transverse shear and drilling
matrices, the converged and pending layer states (`nlayers × nquadpoints`), the bubble dofs `α`
and the stored linearization used to update them (with committed copies for step cutbacks).
"""
mutable struct QuadShellElementData
    T::Matrix{Float64}                 # 3×3 local frame
    Te::Matrix{Float64}                # 24×24 block rotation
    dNdx::Matrix{Vec{2,Float64}}       # 6 × nq gradients of the 4 + 2 functions
    dV::Vector{Float64}                # nq
    ks::Matrix{Float64}                # 30×30 relaxed transverse shear (uncondensed)
    kd::Matrix{Float64}                # 24×24 drilling
    states::Matrix{LayerState}         # converged layer states, nlayers × nq
    pending::Matrix{LayerState}
    α::Vector{Float64}                 # 10 bubble dofs
    d_last::Vector{Float64}            # 24 local nodal dofs of the latest evaluation
    rα::Vector{Float64}                # 10 internal residual of the latest evaluation
    Kαd::Matrix{Float64}               # 10×24
    Kαα_inv::Matrix{Float64}           # 10×10
    α_c::Vector{Float64}               # committed copies
    d_last_c::Vector{Float64}
    rα_c::Vector{Float64}
    Kαd_c::Matrix{Float64}
    Kαα_inv_c::Matrix{Float64}
end

# preallocated buffers for the element evaluation
struct QuadWorkspace
    U::Vector{Float64}        # 30
    f30::Vector{Float64}
    K30::Matrix{Float64}
    Bm::Matrix{Float64}       # 3×30
    Bb::Matrix{Float64}
    DB::Matrix{Float64}       # 3×30
    D3::Matrix{Float64}       # 3×3
    Nv::Vector{Float64}       # 3
    f24::Vector{Float64}
    k24::Matrix{Float64}
    ug::Vector{Float64}
    ul::Vector{Float64}
    fg::Vector{Float64}
    kg::Matrix{Float64}
    tmp24::Matrix{Float64}
    fe_f::Vector{Float64}
    ke_f::Matrix{Float64}
end

QuadWorkspace() = QuadWorkspace(zeros(30), zeros(30), zeros(30, 30), zeros(3, 30), zeros(3, 30), zeros(3, 30),
                                zeros(3, 3), zeros(3), zeros(24), zeros(24, 24), zeros(24), zeros(24), zeros(24),
                                zeros(24, 24), zeros(24, 24), zeros(24), zeros(24, 24))

"""
    PlasticShellModel(dh, section::ShellSection; Cs, drilling, drilling_gamma, qr_b)

Geometrically nonlinear elastic–plastic shell model on the mesh of the DofHandler `dh` (fields
`:u` and `:θ` on `Quadrilateral` cells with 3D node coordinates; an initial imperfection is simply
part of the node coordinates). `section` holds the material, thickness and through-thickness
layers. `Cs`, `drilling` and `drilling_gamma` have the same meaning as in
[`assemble_global_Ke!`](@ref); `qr_b` is the in-plane quadrature (3×3 Gauss by default) used for
the membrane, bending and transverse shear terms. Use [`assemble_internal!`](@ref) for the
internal force vector and tangent stiffness, [`solve_nonlinear`](@ref) to run an incremental
analysis, and [`element_results`](@ref) for stresses.
"""
struct PlasticShellModel
    dh::DofHandler
    section::ShellSection
    elements::Vector{QuadShellElementData}
    Cs::Float64
    drilling::Symbol
    drilling_gamma::Float64
    nq::Int
    ws::QuadWorkspace
end

function PlasticShellModel(dh::DofHandler, section::ShellSection; Cs = DEFAULT_SHEAR_RELAXATION,
                           drilling = DEFAULT_DRILLING, drilling_gamma = DEFAULT_DRILLING_GAMMA,
                           qr_b = QuadratureRule{RefQuadrilateral}(3))
    E = section.mat.E; ν = section.mat.ν; t = section.t
    ip4 = IP4(); ip6 = IP6()
    cv_m = CellValues(QuadratureRule{RefQuadrilateral}(2), ip6, ip4)
    cv_b = CellValues(qr_b, ip6, ip4)
    nq = getnquadpoints(cv_b)
    nl = nlayers(section)
    elements = Vector{QuadShellElementData}(undef, getncells(dh.grid))
    for cell in CellIterator(dh)
        x_global = getcoordinates(cell)
        T = calculation_rotation_matrix(x_global)
        x_local = global_nodal_coords_to_planar_coords(x_global, T)

        ke_m, ke_b, ke_s = local_elastic_pieces(cv_m, cv_b, E, ν, t, x_local; Cs)   # reinit!s cv_m and cv_b
        dNdx = [shape_gradient(cv_b, q, i) for i in 1:6, q in 1:nq]
        dV = [getdetJdV(cv_b, q) for q in 1:nq]

        # relaxed transverse shear on the 30-dof ordering
        ks = zeros(30, 30)
        iwt = [5 * (i - 1) + c for i in 1:6 for c in 3:5]       # w, θx, θy of each function
        ks[iwt, iwt] = ke_s

        # drilling on the 24 nodal dofs (the :penalty option needs the condensed linear matrix)
        k20 = zeros(20, 20)
        k20[IND_UV_20, IND_UV_20] = static_condensation(ke_m, 2 * NNODES)
        k20[IND_WT_20, IND_WT_20] = static_condensation(ke_b + ke_s, 3 * NNODES)
        kd = zeros(24, 24)
        add_drilling_stiffness!(kd, cv_m, k20, E, ν, t, drilling, drilling_gamma)

        Te = rotation_matrix_for_element_stiffness_drilling(T)
        states = [LayerState() for _ in 1:nl, _ in 1:nq]
        elements[cellid(cell)] = QuadShellElementData(T, Te, dNdx, dV, ks, kd, states, copy(states),
                                                      zeros(10), zeros(24), zeros(10), zeros(10, 24), zeros(10, 10),
                                                      zeros(10), zeros(24), zeros(10), zeros(10, 24), zeros(10, 10))
    end
    return PlasticShellModel(dh, section, elements, Cs, drilling, drilling_gamma, nq, QuadWorkspace())
end

"""
    quad_internal_force!(fe, ke, el, sec, ws, d)

Condensed internal force `fe` (24) and tangent `ke` (24×24) of one element in its local frame for
the local nodal displacements `d` (block order `[u v w θx θy θz]` per node). Updates the bubble
dofs from the stored linearization, evaluates the layered section at every quadrature point
(writing `el.pending`) and stores the new linearization of the bubble dofs.
"""
function quad_internal_force!(fe, ke, el::QuadShellElementData, sec::ShellSection, ws::QuadWorkspace, d::AbstractVector)
    # update the internal dofs from the linearization of the previous evaluation
    Δd = d .- el.d_last
    el.α .-= el.Kαα_inv * (el.rα .+ el.Kαd * Δd)

    U = ws.U
    @inbounds for k in 1:20
        U[k] = d[MAP_20_TO_24[k]]
    end
    @inbounds for k in 1:10
        U[20 + k] = el.α[k]
    end

    f30 = ws.f30; K30 = ws.K30; Bm = ws.Bm; Bb = ws.Bb; DB = ws.DB; D3 = ws.D3
    fill!(f30, 0.0); fill!(K30, 0.0)
    dN = el.dNdx

    for q in 1:length(el.dV)
        ux = uy = vx = vy = wx = wy = 0.0
        κx = κy = κxy = 0.0
        @inbounds for i in 1:6
            o = 5 * (i - 1)
            gx = dN[i, q][1]; gy = dN[i, q][2]
            ux += gx * U[o+1]; uy += gy * U[o+1]
            vx += gx * U[o+2]; vy += gy * U[o+2]
            wx += gx * U[o+3]; wy += gy * U[o+3]
            κx += gx * U[o+5]
            κy -= gy * U[o+4]
            κxy += gy * U[o+5] - gx * U[o+4]
        end
        εm = SVector(ux + 0.5 * (ux^2 + vx^2 + wx^2),
                     vy + 0.5 * (uy^2 + vy^2 + wy^2),
                     uy + vx + ux * uy + vx * vy + wx * wy)
        κ = SVector(κx, κy, κxy)

        N, M, Dmm, Dmb, Dbb = section_response(sec, εm, κ, view(el.states, :, q), view(el.pending, :, q))

        fill!(Bm, 0.0); fill!(Bb, 0.0)
        @inbounds for i in 1:6
            o = 5 * (i - 1)
            gx = dN[i, q][1]; gy = dN[i, q][2]
            Bm[1, o+1] = gx * (1 + ux); Bm[1, o+2] = gx * vx;       Bm[1, o+3] = gx * wx
            Bm[2, o+1] = gy * uy;       Bm[2, o+2] = gy * (1 + vy); Bm[2, o+3] = gy * wy
            Bm[3, o+1] = gy * (1 + ux) + gx * uy
            Bm[3, o+2] = gx * (1 + vy) + gy * vx
            Bm[3, o+3] = gy * wx + gx * wy
            Bb[1, o+5] = gx
            Bb[2, o+4] = -gy
            Bb[3, o+4] = -gx; Bb[3, o+5] = gy
        end
        dV = el.dV[q]

        # internal forces
        @inbounds for k in 1:30
            f30[k] += dV * (Bm[1, k] * N[1] + Bm[2, k] * N[2] + Bm[3, k] * N[3] +
                            Bb[1, k] * M[1] + Bb[2, k] * M[2] + Bb[3, k] * M[3])
        end

        # material tangent: Bmᵀ Dmm Bm + Bmᵀ Dmb Bb + Bbᵀ Dmbᵀ Bm + Bbᵀ Dbb Bb
        _copy3!(D3, Dmm); mul!(DB, D3, Bm); mul!(K30, Bm', DB, dV, 1.0)
        _copy3!(D3, Dmb); mul!(DB, D3, Bb); mul!(K30, Bm', DB, dV, 1.0)
        mul!(DB, D3', Bm);                  mul!(K30, Bb', DB, dV, 1.0)
        _copy3!(D3, Dbb); mul!(DB, D3, Bb); mul!(K30, Bb', DB, dV, 1.0)

        # geometric stiffness of the membrane forces on the three translations
        @inbounds for j in 1:6, i in 1:6
            gxi = dN[i, q][1]; gyi = dN[i, q][2]; gxj = dN[j, q][1]; gyj = dN[j, q][2]
            g = dV * (N[1] * gxi * gxj + N[2] * gyi * gyj + N[3] * (gxi * gyj + gyi * gxj))
            for c in 1:3
                K30[5*(i-1)+c, 5*(j-1)+c] += g
            end
        end
    end

    # linear elastic transverse shear
    mul!(f30, el.ks, U, 1.0, 1.0)
    K30 .+= el.ks

    # condense the ten bubble dofs
    Kdd = @view K30[1:20, 1:20]
    Kdα = @view K30[1:20, 21:30]
    Kαα = K30[21:30, 21:30]
    fα = f30[21:30]
    Kαα_inv = inv(Kαα)
    C = Kdα * Kαα_inv                       # 20×10
    f20 = f30[1:20] .- C * fα
    K20 = Kdd .- C * (Kdα')

    fill!(fe, 0.0); fill!(ke, 0.0)
    @inbounds for k in 1:20
        fe[MAP_20_TO_24[k]] = f20[k]
        for l in 1:20
            ke[MAP_20_TO_24[l], MAP_20_TO_24[k]] = K20[l, k]
        end
    end
    mul!(fe, el.kd, d, 1.0, 1.0)
    ke .+= el.kd

    # store the linearization of the internal dofs for the next update
    copyto!(el.d_last, d)
    copyto!(el.rα, fα)
    fill!(el.Kαd, 0.0)
    @inbounds for k in 1:20, l in 1:10
        el.Kαd[l, MAP_20_TO_24[k]] = K30[20 + l, k]
    end
    copyto!(el.Kαα_inv, Kαα_inv)
    return fe, ke
end

@inline function _copy3!(D::Matrix{Float64}, S::SMatrix{3,3,Float64,9})
    @inbounds for j in 1:3, i in 1:3
        D[i, j] = S[i, j]
    end
    return D
end

"""
    assemble_internal!(K, f, model::PlasticShellModel, u)

Assemble the global internal force vector `f` and tangent stiffness `K` for the global solution
vector `u` (Ferrite dof order of `model.dh`). The layer states and bubble-dof linearizations of
this evaluation are kept as pending until [`commit!`](@ref) is called.
"""
function assemble_internal!(K, f, model::PlasticShellModel, u::AbstractVector)
    assembler = start_assemble(K, f)
    ws = model.ws
    fe = ws.f24; ke = ws.k24; ug = ws.ug; ul = ws.ul; fg = ws.fg; kg = ws.kg; tmp = ws.tmp24
    fe_f = ws.fe_f; ke_f = ws.ke_f
    for cell in CellIterator(model.dh)
        el = model.elements[cellid(cell)]
        cd = celldofs(cell)
        @inbounds for k in 1:24
            ug[FIELD_ORDER_24[k]] = u[cd[k]]      # Ferrite field order → nodal block order
        end
        mul!(ul, el.Te', ug)
        quad_internal_force!(fe, ke, el, model.section, ws, ul)
        mul!(fg, el.Te, fe)
        mul!(tmp, ke, el.Te')
        mul!(kg, el.Te, tmp)
        @inbounds for j in 1:24
            fe_f[j] = fg[FIELD_ORDER_24[j]]
            for i in 1:24
                ke_f[i, j] = kg[FIELD_ORDER_24[i], FIELD_ORDER_24[j]]
            end
        end
        assemble!(assembler, cd, ke_f, fe_f)
    end
    return K, f
end

"""
    commit!(model)

Accept the pending layer states and bubble-dof linearizations of the latest
[`assemble_internal!`](@ref) as converged.
"""
function commit!(model::PlasticShellModel)
    for el in model.elements
        copyto!(el.states, el.pending)
        copyto!(el.α_c, el.α); copyto!(el.d_last_c, el.d_last); copyto!(el.rα_c, el.rα)
        copyto!(el.Kαd_c, el.Kαd); copyto!(el.Kαα_inv_c, el.Kαα_inv)
    end
    return model
end

"""
    restore!(model)

Discard the pending states and reset the bubble dofs and their linearization to the last
committed values (used after a step cutback).
"""
function restore!(model::PlasticShellModel)
    for el in model.elements
        copyto!(el.pending, el.states)
        copyto!(el.α, el.α_c); copyto!(el.d_last, el.d_last_c); copyto!(el.rα, el.rα_c)
        copyto!(el.Kαd, el.Kαd_c); copyto!(el.Kαα_inv, el.Kαα_inv_c)
    end
    return model
end

"""
    element_results(model::PlasticShellModel, u)

Section results at the quadrature points of every element for the solution `u`: a vector over
cells of vectors over quadrature points of named tuples `(x, N, M, εm, κ, ε̄p, σ_layers)` with the
global coordinates of the point, the membrane forces and moments (force/length, moment/length) in
the element local frame, the section strains, the maximum equivalent plastic strain over the
layers and the layer stresses (ordered as `model.section.z`). Uses the committed bubble dofs and
layer states; nothing is modified.
"""
function element_results(model::PlasticShellModel, u::AbstractVector)
    sec = model.section
    out = Vector{Vector{NamedTuple}}(undef, getncells(model.dh.grid))
    ug = zeros(24); d = zeros(24); U = zeros(30)
    scratch = [LayerState() for _ in 1:nlayers(sec)]
    ip4 = IP4()
    cv = CellValues(QuadratureRule{RefQuadrilateral}(3), ip4, ip4)
    for cell in CellIterator(model.dh)
        el = model.elements[cellid(cell)]
        cd = celldofs(cell)
        for k in 1:24
            ug[FIELD_ORDER_24[k]] = u[cd[k]]
        end
        mul!(d, el.Te', ug)
        for k in 1:20
            U[k] = d[MAP_20_TO_24[k]]
        end
        U[21:30] .= el.α_c
        xg = getcoordinates(cell)
        res = Vector{NamedTuple}(undef, length(el.dV))
        for q in 1:length(el.dV)
            ux = uy = vx = vy = wx = wy = 0.0; κx = κy = κxy = 0.0
            for i in 1:6
                o = 5 * (i - 1); gx = el.dNdx[i, q][1]; gy = el.dNdx[i, q][2]
                ux += gx * U[o+1]; uy += gy * U[o+1]; vx += gx * U[o+2]; vy += gy * U[o+2]
                wx += gx * U[o+3]; wy += gy * U[o+3]
                κx += gx * U[o+5]; κy -= gy * U[o+4]; κxy += gy * U[o+5] - gx * U[o+4]
            end
            εm = SVector(ux + 0.5 * (ux^2 + vx^2 + wx^2), vy + 0.5 * (uy^2 + vy^2 + wy^2),
                         uy + vx + ux * uy + vx * vy + wx * wy)
            κ = SVector(κx, κy, κxy)
            st = view(el.states, :, q)
            N, M, _, _, _ = section_response(sec, εm, κ, st, scratch)
            σl = layer_stresses(sec, εm, κ, st)
            # global position of the quadrature point (bilinear map of the corner nodes)
            ξ = Ferrite.getpoints(cv.qr)[q]
            x = sum(Ferrite.reference_shape_value(ip4, ξ, i) * xg[i] for i in 1:4)
            res[q] = (; x, N, M, εm, κ, ε̄p = maximum(s.ε̄p for s in st), σ_layers = σl)
        end
        out[cellid(cell)] = res
    end
    return out
end

"""
    lumped_mass(model::PlasticShellModel)

Lumped mass vector (Ferrite dof order) of the shell with unit density: each element's mass
`t × area` is split equally between its nodes on the three translations, with the rotary inertia
`t² / 12` of a plate per unit mass on the three rotations. Used by the artificial damping of
[`solve_nonlinear`](@ref).
"""
function lumped_mass(model::PlasticShellModel)
    m = zeros(ndofs(model.dh))
    t = model.section.t
    for cell in CellIterator(model.dh)
        el = model.elements[cellid(cell)]
        me = t * sum(el.dV) / 4
        cd = celldofs(cell)
        for i in 1:4, k in 1:3
            m[cd[3(i-1)+k]] += me
            m[cd[12+3(i-1)+k]] += me * t^2 / 12
        end
    end
    return m
end

"""
    displaced_grid(grid, Δx)

Copy of `grid` with every node `i` moved by `Δx[i]` (a `Vec{3}`), e.g. to impose an initial
geometric imperfection from a scaled buckling mode.
"""
function displaced_grid(grid::Grid, Δx::AbstractVector)
    nodes = [Ferrite.Node(n.x + Δx[i]) for (i, n) in enumerate(grid.nodes)]
    return Grid(copy(grid.cells), nodes; cellsets = copy(Ferrite.getcellsets(grid)),
                nodesets = copy(Ferrite.getnodesets(grid)), facetsets = copy(Ferrite.getfacetsets(grid)))
end
