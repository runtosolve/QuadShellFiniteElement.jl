# Tests of the elastic–plastic, geometrically nonlinear capability (plasticity.jl, nonlinear.jl,
# solver.jl). Included by runtests.jl.

using QuadShellFiniteElement, Ferrite, LinearAlgebra, StaticArrays, Random, Test
const TS = QuadShellFiniteElement

# true stress – plastic strain table of Moen (2008) Fig. 7.1 (Yu 2005 coupon), ksi
const HARD = TS.IsotropicHardening([0.0, 0.0016, 0.0259, 0.07, 0.1407, 0.1988], [33.0, 34.31, 42.14, 51.35, 60.86, 65.5])
const MAT = TS.J2PlaneStress(29500.0, 0.3, HARD)

function quad_plate_grid(a, b, nx, ny)
    nodes = Ferrite.Node{3,Float64}[]
    for j in 0:ny, i in 0:nx
        push!(nodes, Ferrite.Node(Ferrite.Vec(a * i / nx, b * j / ny, 0.0)))
    end
    id(i, j) = j * (nx + 1) + i + 1
    cells = Ferrite.Quadrilateral[]
    for j in 0:ny-1, i in 0:nx-1
        push!(cells, Ferrite.Quadrilateral((id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1))))
    end
    return Ferrite.Grid(cells, nodes)
end

function quad_dofhandler(grid)
    ip = Lagrange{RefQuadrilateral,1}()
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^3); close!(dh)
    return dh
end

node_dofs(dh, n, comp) = begin
    for cell in CellIterator(dh)
        for (i, nn) in enumerate(cell.nodes)
            nn == n && return celldofs(cell)[comp <= 3 ? 3(i-1)+comp : 12+3(i-1)+comp-3]
        end
    end
    error("node not found")
end

@testset "plane-stress return mapping" begin
    # hardening curve interpolation
    @test TS.yield_stress(HARD, 0.0) == (33.0, (34.31 - 33.0) / 0.0016)
    @test TS.yield_stress(HARD, 0.0259)[1] ≈ 42.14
    @test TS.yield_stress(HARD, 0.5) == (65.5, 0.0)
    @test TS.yield_stress(TS.IsotropicHardening(50.0), 0.1) == (50.0, 0.0)
    @test TS.yield_stress(TS.IsotropicHardening(50.0, 1000.0), 0.01) == (60.0, 1000.0)

    # elastic step returns the elastic tangent
    σ, C, st = TS.return_mapping(MAT, SVector(5e-4, -1.5e-4, 0.0), TS.LayerState())
    @test C == MAT.C
    @test st.ε̄p == 0.0
    @test σ ≈ MAT.C * SVector(5e-4, -1.5e-4, 0.0)

    # plastic step: stress on the hardened yield surface, consistent tangent = finite difference
    st0 = TS.LayerState(SVector(0.004, -0.002, 0.001), 0.004)
    ε0 = SVector(0.006, -0.001, 0.003)
    σ0, Cep, st1 = TS.return_mapping(MAT, ε0, st0)
    @test TS.von_mises(σ0) ≈ TS.yield_stress(HARD, st1.ε̄p)[1] rtol = 1e-9
    @test st1.ε̄p > st0.ε̄p
    @test Cep ≈ Cep' atol = 1e-9
    h = 1e-7
    for j in 1:3
        e = SVector{3}(i == j ? h : 0.0 for i in 1:3)
        col = (TS.return_mapping(MAT, ε0 + e, st0)[1] - TS.return_mapping(MAT, ε0 - e, st0)[1]) / (2h)
        @test col ≈ Cep[:, j] rtol = 1e-6
    end

    # uniaxial loading: σ = σy(ε̄p) and ε = σ/E + ε̄p along the whole curve
    st = TS.LayerState(); ε = SVector(0.0, 0.0, 0.0)
    for εx in range(0.0005, 0.05, 100)
        ε = SVector(εx, ε[2] + (εx - ε[1]) * (st.ε̄p > 0 ? -0.5 : -0.3), 0.0)
        local σ, C, s
        for it in 1:30
            σ, C, s = TS.return_mapping(MAT, ε, st)
            r = SVector(σ[2], σ[3])
            norm(r) < 1e-10 && break
            J = SMatrix{2,2}(C[2, 2], C[3, 2], C[2, 3], C[3, 3])
            Δ = J \ r
            ε = SVector(ε[1], ε[2] - Δ[1], ε[3] - Δ[2])
        end
        st = s
        @test abs(εx - (σ[1] / 29500 + st.ε̄p)) < 1e-9
        st.ε̄p > 0 && @test σ[1] ≈ TS.yield_stress(HARD, st.ε̄p)[1] rtol = 1e-8
    end
    @test st.ε̄p > 0.048

    # layers
    z, w = TS.thickness_layers(2.0, 5; rule = :simpson)
    @test z == [-1.0, -0.5, 0.0, 0.5, 1.0] && sum(w) ≈ 2.0 && sum(w .* z .^ 2) ≈ 2 / 3
    z, w = TS.thickness_layers(2.0, 5)
    @test sum(w) ≈ 2.0 && sum(w .* z .^ 2) ≈ 2 / 3
end

@testset "nonlinear element: linear limit and consistent tangent" begin
    E = 29500.0; ν = 0.3; t = 0.0346
    grid = quad_plate_grid(2.0, 1.0, 4, 2)
    dh = quad_dofhandler(grid)
    sec = TS.ShellSection(MAT, t; nlayers = 5)
    model = TS.PlasticShellModel(dh, sec)
    K = allocate_matrix(dh); f = zeros(ndofs(dh))
    TS.assemble_internal!(K, f, model, zeros(ndofs(dh)))
    ip4 = TS.IP4(); ip6 = TS.IP6(); qr_m = QuadratureRule{RefQuadrilateral}(2); qr_b = QuadratureRule{RefQuadrilateral}(3)
    Klin = allocate_matrix(dh); TS.assemble_global_Ke!(Klin, dh, qr_m, qr_b, ip4, ip6, E, ν, t)
    @test norm(f) == 0
    @test norm(K - Klin) / norm(Klin) < 1e-12      # tangent at zero = linear element (same Cs, drilling)

    # finite-difference check of the condensed element tangent in the elastic and plastic ranges on a
    # distorted element. The bubble dofs are updated by one Newton step per evaluation, so every
    # evaluation is repeated until their residual vanishes and the residual lies on the condensed
    # manifold the tangent is derived on.
    nodes = [Ferrite.Node(Ferrite.Vec(0.0, 0.0, 0.0)), Ferrite.Node(Ferrite.Vec(0.5, 0.02, 0.0)),
             Ferrite.Node(Ferrite.Vec(0.52, 0.45, 0.0)), Ferrite.Node(Ferrite.Vec(-0.03, 0.5, 0.0))]
    grid1 = Ferrite.Grid([Ferrite.Quadrilateral((1, 2, 3, 4))], nodes)
    dh1 = quad_dofhandler(grid1)
    model1 = TS.PlasticShellModel(dh1, sec)
    el = model1.elements[1]; ws = model1.ws
    fe = zeros(24); ke = zeros(24, 24)
    function feval(d)
        for _ in 1:40
            TS.quad_internal_force!(fe, ke, el, sec, ws, d)
        end
        return copy(fe), copy(ke), norm(el.rα)
    end
    function reset!(el)
        for s in eachindex(el.states); el.states[s] = TS.LayerState(); el.pending[s] = TS.LayerState(); end
        fill!(el.α, 0.0); fill!(el.d_last, 0.0); fill!(el.rα, 0.0); fill!(el.Kαd, 0.0); fill!(el.Kαα_inv, 0.0)
    end
    Random.seed!(2)
    for (su, sθ, tol) in ((1e-4, 1e-3, 1e-9), (1.5e-3, 3e-2, 1e-5), (1e-2, 1e-1, 1e-6))
        d = zeros(24)
        for i in 1:4
            d[6(i-1) .+ (1:3)] = su * randn(3); d[6(i-1) .+ (4:6)] = sθ * randn(3)
        end
        f0, K0, rα = feval(d)
        @test rα < 1e-12
        h = 1e-6; err = 0.0; ref = 0.0
        for j in 1:24
            dp = copy(d); dp[j] += h; fp, _, _ = feval(dp)
            dm = copy(d); dm[j] -= h; fm, _, _ = feval(dm)
            col = (fp - fm) / (2h)
            err = max(err, norm(K0[:, j] - col)); ref = max(ref, norm(col))
        end
        @test err / ref < tol
        su > 1e-3 && @test maximum(s.ε̄p for s in el.pending) > 0.005
        reset!(el)
    end
end

@testset "strip in compression follows the hardening curve" begin
    t = 0.0346; a = 4.0; b = 1.0
    grid = quad_plate_grid(a, b, 8, 2)
    dh = quad_dofhandler(grid)
    model = TS.PlasticShellModel(dh, TS.ShellSection(MAT, t; nlayers = 5))
    tol = 1e-8
    left = Set(i for (i, n) in enumerate(grid.nodes) if abs(n.x[1]) < tol)
    right = Set(i for (i, n) in enumerate(grid.nodes) if abs(n.x[1] - a) < tol)
    corner = Set(i for (i, n) in enumerate(grid.nodes) if abs(n.x[1]) < tol && abs(n.x[2]) < tol)
    allnodes = Set(1:getnnodes(grid))
    δmax = 0.04 * a
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, left, (x, t_) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, right, (x, t_) -> [-t_ * δmax], [1]))
    add!(ch, Dirichlet(:u, corner, (x, t_) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, allnodes, (x, t_) -> [0.0], [3]))
    add!(ch, Dirichlet(:θ, allnodes, (x, t_) -> [0.0, 0.0, 0.0], [1, 2, 3]))
    close!(ch)
    rdofs = unique([node_dofs(dh, n, 1) for n in right])
    hist = TS.solve_nonlinear(model, ch; reaction_dofs = rdofs, Δt = 0.02, Δt_max = 0.05)
    @test hist.converged
    @test maximum(hist.iterations) <= 4
    for (tt, P) in zip(hist.t, hist.P)
        ε = -tt * δmax / a                 # nominal strain
        εGL = ε + ε^2 / 2                  # Green–Lagrange strain
        S = P / (b * t) / (1 + ε)          # 2nd Piola–Kirchhoff stress from the nominal reaction
        εp = -(εGL - S / 29500.0)
        εp > 1e-6 && @test -S ≈ TS.yield_stress(HARD, εp)[1] rtol = 1e-6
    end
    @test -hist.P[end] / (b * t) ≈ 42.82 rtol = 1e-3           # (1 + ε) σy(ε̄p = 0.0377)
    res = TS.element_results(model, hist.u)
    @test all(q.ε̄p ≈ res[1][1].ε̄p for r in res for q in r)      # uniform state
end

@testset "elastic–plastic bending: pure bending of a strip by end rotations" begin
    # strip bent by equal and opposite end rotations, ends free to approach (no axial force), v = 0
    # and θx = 0 (plane strain across the width): uniform curvature κ = 8 κy, perfectly plastic.
    # The fully plastic plane-strain moment is M_p = (2/√3) σy t² / 4; at κ = 8 κy the exact
    # rectangular section gives M/M_p = 1 − 1/(3·64) = 0.995 and 9 Simpson layers 0.9994.
    E = 29500.0; ν = 0.3; t = 0.1; σy = 33.0
    mat = TS.J2PlaneStress(E, ν, σy)
    a = 2.0; b = 0.5
    grid = quad_plate_grid(a, b, 8, 2)
    dh = quad_dofhandler(grid)
    sec = TS.ShellSection(mat, t; nlayers = 9, rule = :simpson)
    model = TS.PlasticShellModel(dh, sec)
    tol = 1e-8
    allnodes = Set(1:getnnodes(grid))
    x0 = Set(i for (i, n) in enumerate(grid.nodes) if abs(n.x[1]) < tol)
    xa = Set(i for (i, n) in enumerate(grid.nodes) if abs(n.x[1] - a) < tol)
    κy = 2 * σy / (E * t); κmax = 8 * κy
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, allnodes, (x, t_) -> [0.0], [2]))
    add!(ch, Dirichlet(:θ, allnodes, (x, t_) -> [0.0, 0.0], [1, 3]))
    add!(ch, Dirichlet(:u, x0, (x, t_) -> [0.0, 0.0], [1, 3]))
    add!(ch, Dirichlet(:u, xa, (x, t_) -> [0.0], [3]))
    add!(ch, Dirichlet(:θ, x0, (x, t_) -> [t_ * κmax * a / 2], [2]))      # θy = −κ (x − a/2)
    add!(ch, Dirichlet(:θ, xa, (x, t_) -> [-t_ * κmax * a / 2], [2]))
    close!(ch)
    hist = TS.solve_nonlinear(model, ch; Δt = 0.05, Δt_max = 0.05)
    @test hist.converged
    res = TS.element_results(model, hist.u)
    Mx = [q.M[1] for r in res for q in r]
    Nx = [q.N[1] for r in res for q in r]
    Mp = 2 / sqrt(3) * σy * t^2 / 4
    @test all(abs.(Mx .- Mx[1]) .< 1e-4 * abs(Mx[1]))           # uniform moment
    @test -Mx[1] ≈ 0.9994 * Mp rtol = 2e-3
    # no axial force: the bubble membrane modes leave a self-equilibrated N within each element
    # (point values below 1 % of σy t), the element averages vanish
    @test maximum(abs, Nx) < 1e-2 * σy * t
    @test all(abs(sum(q.N[1] * el.dV[k] for (k, q) in enumerate(r)) / sum(el.dV)) < 1e-6 * σy * t
              for (r, el) in zip(res, model.elements))
    @test maximum(q.ε̄p for r in res for q in r) > 0.005
end
