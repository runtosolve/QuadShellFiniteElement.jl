# Shared benchmark drivers for the QuadShellFiniteElement.jl tests (included by runtests.jl).
# Builds structured quadrilateral shell meshes with 3D nodes and solves the SSRC 2025 beam-like
# plate problems and the rectangular plate buckling problem with the element itself.

using QuadShellFiniteElement, Ferrite, LinearAlgebra, Tensors



const Q = QuadShellFiniteElement
const E = 200000.0
const ν = 0.30

ip4 = IP4(); ip6 = IP6()
qr_m = QuadratureRule{RefQuadrilateral}(2)      # membrane, 2×2
qr_b = QuadratureRule{RefQuadrilateral}(3)      # bending + shear, 3×3
qr_g = QuadratureRule{RefQuadrilateral}(2)      # geometric stiffness, 2×2

# --------------------------------------------------------------------------------------------
# Structured quadrilateral mesh of an a × b plate with 3D nodes. `plane = (cξ, cη, cn)` gives the
# global component carrying the length direction ξ, the width direction η and the normal:
# (1, 2, 3) is the XY plane, (3, 2, 1) the YZ plane (ξ = Z), (3, 1, 2) the paper's XZ plane
# (Z along the 1000 mm length, X across the 100 mm width, normal Y).
# --------------------------------------------------------------------------------------------
function plate_grid(a, b, nx, ny; plane = (1, 2, 3))
    cξ, cη, cn = plane
    nodes = Ferrite.Node{3,Float64}[]
    for j in 0:ny, i in 0:nx
        c = zeros(3); c[cξ] = a * i / nx; c[cη] = b * j / ny
        push!(nodes, Ferrite.Node(Ferrite.Vec(c[1], c[2], c[3])))
    end
    id(i, j) = j * (nx + 1) + i + 1
    cells = Ferrite.Quadrilateral[]
    for j in 0:ny-1, i in 0:nx-1
        push!(cells, Ferrite.Quadrilateral((id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1))))
    end
    return Ferrite.Grid(cells, nodes)
end

ξη(node, plane) = (node.x[plane[1]], node.x[plane[2]])

function shell_dofhandler(grid)
    ip = Lagrange{RefQuadrilateral,1}()
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^3); close!(dh)
    node_to_dofs = Dict{Int,Vector{Int}}()
    for cell in CellIterator(dh)
        cd = celldofs(cell)
        for (i, n) in enumerate(cell.nodes)
            node_to_dofs[n] = [cd[3(i-1)+1], cd[3(i-1)+2], cd[3(i-1)+3],
                               cd[12+3(i-1)+1], cd[12+3(i-1)+2], cd[12+3(i-1)+3]]
        end
    end
    return dh, node_to_dofs
end

nodes_where(grid, plane, f; tol = 1e-8) = Set(i for (i, n) in enumerate(grid.nodes) if f(ξη(n, plane)...))

# distribute a line load intensity q (force per length along η) over the nodes on a line, tributary
tributary(η, b, ny) = (abs(η) < 1e-8 || abs(η - b) < 1e-8) ? 0.5 * b / ny : b / ny

# --------------------------------------------------------------------------------------------
# SSRC 2025 beam-like plate: a = 1000 (ξ), b = 100 (η), simply supported at ξ = 0 and ξ = a.
# Boundary conditions of Fig. 3: normal translation fixed on the two end lines, ξ translation
# fixed at the centre node, η translation fixed at the two corner nodes (η = 0, ξ = 0 and ξ = a).
# --------------------------------------------------------------------------------------------
function ssrc_plate(t, nx, ny; plane = (3, 1, 2), load = :line_normal, Cs = Q.DEFAULT_SHEAR_RELAXATION)
    a = 1000.0; b = 100.0
    grid = plate_grid(a, b, nx, ny; plane)
    dh, node_to_dofs = shell_dofhandler(grid)
    cξ, cη, cn = plane

    K = allocate_matrix(dh)
    K = Q.assemble_global_Ke!(K, dh, qr_m, qr_b, ip4, ip6, E, ν, t; Cs)

    ends    = nodes_where(grid, plane, (ξ, η) -> abs(ξ) < 1e-8 || abs(ξ - a) < 1e-8)
    centre  = nodes_where(grid, plane, (ξ, η) -> abs(ξ - a/2) < 1e-8 && abs(η - b/2) < 1e-8)
    corners = nodes_where(grid, plane, (ξ, η) -> abs(η) < 1e-8 && (abs(ξ) < 1e-8 || abs(ξ - a) < 1e-8))
    midline = nodes_where(grid, plane, (ξ, η) -> abs(ξ - a/2) < 1e-8)
    @assert !isempty(centre) "nx and ny must be even"

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, ends,    (x, t_) -> [0.0], [cn]))
    add!(ch, Dirichlet(:u, centre,  (x, t_) -> [0.0], [cξ]))
    add!(ch, Dirichlet(:u, corners, (x, t_) -> [0.0], [cη]))
    close!(ch)

    f = zeros(ndofs(dh))
    if load == :line_normal            # 1 N/mm across the width at midspan, along the normal
        for n in midline
            f[node_to_dofs[n][cn]] += 1.0 * tributary(ξη(grid.nodes[n], plane)[2], b, ny)
        end
    elseif load == :pressure_normal    # 0.001 N/mm² over the surface, along the element normal
        Q.assemble_pressure_load!(f, dh, qr_m, ip4, 0.001)
    elseif load == :line_inplane       # 1000 N/mm across the width at midspan, in-plane (η)
        for n in midline
            f[node_to_dofs[n][cη]] += 1000.0 * tributary(ξη(grid.nodes[n], plane)[2], b, ny)
        end
    end

    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)

    comp = load == :line_inplane ? cη : cn
    δ = [u[node_to_dofs[n][comp]] for n in midline]
    return (; δmax = maximum(abs, δ), δavg = abs(sum(δ) / length(δ)), u, dh, node_to_dofs, grid, ch, K)
end

# --------------------------------------------------------------------------------------------
# Plate buckling: a × b × t in uniform compression along ξ, all four edges simply supported
# (w = 0) or clamped; unit total load; returns the buckling coefficient k of the first mode.
# With `column = true` only the loaded end edges are supported (flexural column buckling).
# --------------------------------------------------------------------------------------------
function plate_buckling(a, b, t, nx, ny; Cs = Q.DEFAULT_SHEAR_RELAXATION, plane = (1, 2, 3),
                        clamped = false, column = false)
    grid = plate_grid(a, b, nx, ny; plane)
    dh, node_to_dofs = shell_dofhandler(grid)
    cξ, cη, cn = plane

    K = allocate_matrix(dh)
    K = Q.assemble_global_Ke!(K, dh, qr_m, qr_b, ip4, ip6, E, ν, t; Cs)

    tol = 1e-8
    edge0   = nodes_where(grid, plane, (ξ, η) -> abs(ξ) < tol)
    edgeA   = nodes_where(grid, plane, (ξ, η) -> abs(ξ - a) < tol)
    edgeB0  = nodes_where(grid, plane, (ξ, η) -> abs(η) < tol)
    edgeBb  = nodes_where(grid, plane, (ξ, η) -> abs(η - b) < tol)
    midline = nodes_where(grid, plane, (ξ, η) -> abs(ξ - a / 2) < tol)   # u_ξ datum (symmetric loading)
    centerl = nodes_where(grid, plane, (ξ, η) -> abs(η - b / 2) < tol)   # u_η datum (symmetric Poisson expansion)
    @assert !isempty(midline) && !isempty(centerl) "nx and ny must be even"
    supported = column ? union(edge0, edgeA) : union(edge0, edgeA, edgeB0, edgeBb)

    # unit total compression at each loaded edge, tributary-length weighted (self-equilibrated)
    F = zeros(ndofs(dh))
    for (edge, sgn) in ((edgeA, -1.0), (edge0, +1.0)), n in edge
        η = ξη(grid.nodes[n], plane)[2]
        F[node_to_dofs[n][cξ]] += sgn * tributary(η, b, ny) / b
    end

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, supported, (x, t_) -> [0.0], [cn]))                  # w = 0
    clamped && add!(ch, Dirichlet(:θ, supported, (x, t_) -> [0.0, 0.0, 0.0], [1, 2, 3]))
    add!(ch, Dirichlet(:u, centerl, (x, t_) -> [0.0], [cη]))
    add!(ch, Dirichlet(:u, midline, (x, t_) -> [0.0], [cξ]))
    close!(ch)

    apply!(K, F, ch)
    u = K \ F
    apply!(u, ch)
    shortening = sum(u[node_to_dofs[n][cξ]] for n in edge0) / length(edge0) -
                 sum(u[node_to_dofs[n][cξ]] for n in edgeA) / length(edgeA)

    # membrane stresses in each element's local frame (at the 2×2 Gauss points), × t
    σXX, σYY, τXY = Q.element_membrane_stresses(dh, u, ip4, E, ν, t; qr = qr_g)
    Kg = allocate_matrix(dh)
    Kg = Q.assemble_global_Kg!(Kg, dh, qr_g, ip4, σXX .* t, σYY .* t, τXY .* t)

    free = setdiff(1:ndofs(dh), ch.prescribed_dofs)
    Kff  = Symmetric(Matrix(K[free, free]))
    Kgff = Symmetric(Matrix(Kg[free, free]))
    μ = eigvals(-Kgff, Kff)                    # μ = 1 / P  for  K φ = P (-Kg) φ
    Pcr = 1 / maximum(μ)
    k = Pcr / (b * t) * 12 * (1 - ν^2) * b^2 / (π^2 * E * t^2)
    return (; k, Pcr, σcr = Pcr / (b * t), shortening, σXX)
end

