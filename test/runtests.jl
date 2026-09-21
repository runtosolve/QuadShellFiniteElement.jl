# Tests for QuadShellFiniteElement.jl
#
# 1. Element level: agreement with a line-by-line transcription of S. Ádány's MATLAB q42 element
#    (adany_matlab_reference.jl), symmetry, rigid body modes.
# 2. Benchmarks of Moen & Ádány, SSRC 2025 ("Thin shell finite element formulations implemented in
#    open-source software"): 100 mm × 1000 mm plate, E = 200000 MPa, ν = 0.30, thin case t = 2 mm.
#    Out-of-plane and in-plane bending under a midspan line load and under uniform pressure, and
#    flexural column buckling.
# 3. Plate buckling: rectangular plate a × b × t in uniform compression along a.
#    σ_cr = k π² E / (12 (1-ν²)) (t/b)²,  k = 4.0 simply supported, k ≈ 10.07 clamped square plate.
#    The pre-buckling membrane stress is computed with the element itself, then K φ = P_cr (-Kg) φ is
#    solved densely on the free dofs.

using QuadShellFiniteElement, Ferrite, LinearAlgebra, Tensors, Test

include("adany_matlab_reference.jl")
include("benchmarks.jl")

b = 92.1              # mm (362S162 web flat width)
t = 0.0346 * 25.4     # mm (33 mil), b/t ≈ 105

@testset "QuadShellFiniteElement" begin

    @testset "element matrices vs. Ádány MATLAB q42 element" begin
        # distorted, warped and tilted quadrilateral
        P = [Vec(0.0, 0.0, 0.0), Vec(12.0, 1.0, 3.0), Vec(11.0, 10.0, 4.0), Vec(-1.0, 9.0, 1.5)]
        ax = [1, 2, 3] / norm([1, 2, 3]); ang = 0.7
        Kx = [0 -ax[3] ax[2]; ax[3] 0 -ax[1]; -ax[2] ax[1] 0]
        R = I + sin(ang) * Kx + (1 - cos(ang)) * Kx^2
        Pg = [Vec(Tuple(R * collect(p) .+ [5, -3, 2])) for p in P]
        tt = 1.5

        T = Q.calculation_rotation_matrix(Pg)
        xl = Q.global_nodal_coords_to_planar_coords(Pg, T)
        ke_local = Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, tt, xl; Cs = 0.0, drilling = :penalty)
        Te = Q.rotation_matrix_for_element_stiffness_drilling(T)
        ke_global = Te * ke_local * Te'

        ref_global, ref_local, Tref, Pe = AdanyReference.element_ke_global(collect.(Pg)..., tt, E, ν)
        @test T ≈ Tref atol = 1e-12
        @test all(isapprox.([collect(x) for x in xl], [p[1:2] for p in Pe]; atol = 1e-10))
        @test maximum(abs(p[3]) for p in Pe) < 1e-12                 # projected nodes are planar
        @test ke_local ≈ ref_local rtol = 1e-12
        @test ke_global ≈ ref_global rtol = 1e-12

        @test size(ke_local) == (24, 24)
        @test ke_local ≈ ke_local'
        @test all(diag(ke_local) .> 0)
        @test Te * Te' ≈ I

        # six rigid body modes when the drilling dofs are excluded
        nd = setdiff(1:24, Q.IND_DRILL_24)
        ev = eigvals(Symmetric(ke_local[nd, nd]))
        @test count(abs.(ev) .< 1e-9 * maximum(ev)) == 6

        # Hughes–Brezzi drilling (default): same membrane and plate blocks, and the full 24×24 matrix
        # has exactly six zero-energy modes (the drilling dofs follow a rigid in-plane rotation)
        ke_hb = Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, tt, xl; Cs = 0.0)
        @test Q.DEFAULT_DRILLING == :hughes_brezzi
        @test ke_hb ≈ ke_hb'
        w24 = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]
        @test ke_hb[w24, w24] ≈ ke_local[w24, w24] rtol = 1e-12
        ev_hb = eigvals(Symmetric(ke_hb))
        @test count(abs.(ev_hb) .< 1e-9 * maximum(ev_hb)) == 6
        d = zeros(24)                                        # rigid in-plane rotation ω about the origin
        for i in 1:4
            d[6(i-1)+1] = -xl[i][2]; d[6(i-1)+2] = xl[i][1]; d[6(i-1)+6] = 1.0
        end
        @test norm(ke_hb * d) < 1e-9 * maximum(abs, ke_hb)
        @test_throws ArgumentError Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, tt, xl; drilling = :none)

        # membrane and plate blocks are the uncondensed MATLAB routines' condensed results
        Tm, P1e, P2e, P3e, P4e = AdanyReference.ct_4node_g2e(collect.(Pg)...)
        G = E / (2(1 + ν))
        ke_uv = AdanyReference.ke_uv_4n_condens_from_12to8dof_num(P1e, P2e, P3e, P4e, tt, E, E, ν, ν, G, 4)
        ke_wt = AdanyReference.ke_wt_4n_condens_from_18to12dof_num(P1e, P2e, P3e, P4e, tt, E, E, ν, ν, G, 9)
        m24 = [1, 2, 7, 8, 13, 14, 19, 20]; w24 = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]
        @test ke_local[m24, m24] ≈ ke_uv rtol = 1e-12
        @test ke_local[w24, w24] ≈ ke_wt rtol = 1e-12

        # shear relaxation reduces (never increases) stiffness and leaves the membrane block untouched
        ke_relaxed = Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, tt, xl; Cs = 0.2, drilling = :penalty)
        @test all(eigvals(Symmetric(ke_local - ke_relaxed)) .> -1e-8 * maximum(abs, ke_local))
        @test ke_relaxed[m24, m24] ≈ ke_local[m24, m24]
        @test Q.DEFAULT_SHEAR_RELAXATION == 0.1
    end

    @testset "element matrices vs. Ádány MATLAB q42 element run in Octave" begin
        # test/reference/octave_q42/ holds the output of test/reference/make_reference.m, S. Ádány's
        # unmodified .m files run in GNU Octave for a distorted, warped and tilted quadrilateral
        refdir = joinpath(@__DIR__, "reference", "octave_q42")
        readmat(name) = permutedims(reduce(hcat, [parse.(Float64, split(l)) for l in readlines(joinpath(refdir, name))]))
        Eo, νo, to = vec(readmat("material.txt"))
        Pg = [Vec(Tuple(r)) for r in eachrow(readmat("nodes_global.txt"))]
        T = Q.calculation_rotation_matrix(Pg)
        xl = Q.global_nodal_coords_to_planar_coords(Pg, T)
        @test T ≈ readmat("T.txt") atol = 1e-12
        @test reduce(hcat, collect.(xl))' ≈ readmat("nodes_local.txt")[:, 1:2] atol = 1e-10
        ke_local = Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, Eo, νo, to, xl; Cs = 0.0, drilling = :penalty)
        Te = Q.rotation_matrix_for_element_stiffness_drilling(T)
        @test ke_local ≈ readmat("ke24_local.txt") rtol = 1e-12
        @test Te * ke_local * Te' ≈ readmat("ke24_global.txt") rtol = 1e-12
        m24 = [1, 2, 7, 8, 13, 14, 19, 20]; w24 = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]
        @test ke_local[m24, m24] ≈ readmat("ke_uv_condensed.txt") rtol = 1e-12
        @test ke_local[w24, w24] ≈ readmat("ke_wt_condensed.txt") rtol = 1e-12
        # uncondensed 4-function membrane matrix (ke_uv_4n_nocondens_8dof_num.m)
        cv4 = CellValues(qr_m, ip4, ip4); reinit!(cv4, xl)
        Dm = Q.calculate_membrane_constitutive_matrix(Eo, νo, to)
        @test Q.calculate_element_membrane_stiffness_matrix(Dm, cv4) ≈ readmat("ke_uv_nocondens.txt") rtol = 1e-12
    end

    @testset "membrane patch test: constant strain on a distorted quad" begin
        # u = a1 + a2 x + a3 y,  v = b1 + b2 x + b3 y  is reproduced exactly by the condensed element
        x = [Vec(0.0, 0.0), Vec(10.0, 1.0), Vec(9.0, 8.0), Vec(-1.0, 7.0)]
        ke = Q.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, 1.0, x)   # Cs does not touch the membrane block
        εx, εy, γ = 1e-3, -2e-4, 5e-4
        d = zeros(24)
        for i in 1:4
            d[6(i-1)+1] = εx * x[i][1] + γ / 2 * x[i][2]
            d[6(i-1)+2] = εy * x[i][2] + γ / 2 * x[i][1]
        end
        fint = ke * d
        # nodal forces must equal the consistent forces of the constant stress field on each node
        D = Q.calculate_membrane_constitutive_matrix(E, ν, 1.0)
        σ = D * [εx, εy, γ]
        cv = CellValues(qr_m, ip4, ip4); reinit!(cv, x)
        fref = zeros(24)
        for q in 1:getnquadpoints(cv), i in 1:4
            dN = shape_gradient(cv, q, i); dV = getdetJdV(cv, q)
            fref[6(i-1)+1] += (dN[1] * σ[1] + dN[2] * σ[3]) * dV
            fref[6(i-1)+2] += (dN[2] * σ[2] + dN[1] * σ[3]) * dV
        end
        @test fint ≈ fref rtol = 1e-10
    end

    @testset "membrane: uniform compression" begin
        r = plate_buckling(b, b, t, 6, 6)
        @test r.shortening ≈ 1.0 * b / (E * b * t) rtol = 1e-8      # P L / (E A)
        @test all(s -> all(isapprox.(s, -1.0 / (b * t); rtol = 1e-8)), r.σXX)   # uniform stress
    end

    @testset "SSRC 2025 thin plate benchmarks, t = 2 mm (Moen & Ádány 2025, Sec. 5)" begin
        # out-of-plane bending, midspan line load 1 N/mm: analytical 156.252 mm (156.25 Euler–Bernoulli)
        @test ssrc_plate(2.0, 20, 2; load = :line_normal).δavg ≈ 156.25 rtol = 0.01
        @test ssrc_plate(2.0, 92, 10; load = :line_normal).δavg ≈ 156.25 rtol = 0.005
        # out-of-plane bending, uniform pressure 0.001 N/mm²: analytical 97.657 mm
        @test ssrc_plate(2.0, 20, 2; load = :pressure_normal).δavg ≈ 97.657 rtol = 0.01
        @test ssrc_plate(2.0, 92, 10; load = :pressure_normal).δavg ≈ 97.657 rtol = 0.005
        # in-plane bending, midspan line load 1000 N/mm: analytical 64.45 mm (the paper's FE solutions
        # with point supports tend to ≈ 66.0 mm, Sec. 5.3); this element gives 64.16, 64.65, 64.99 mm on
        # 20 × 2, 40 × 4 and 92 × 10 meshes
        @test ssrc_plate(2.0, 92, 10; load = :line_inplane).δavg ≈ 64.45 rtol = 0.02
        # thick case t = 100 mm, midspan line load 1000 N/mm: analytical 1.289 mm (1.25 Euler–Bernoulli),
        # i.e. the Mindlin shear deformation is captured
        @test ssrc_plate(100.0, 92, 10; load = :line_normal).δavg * 1000 ≈ 1.289 rtol = 0.005
        # the unrelaxed (MATLAB) element converges to the same answers, but needs a finer mesh
        @test ssrc_plate(2.0, 92, 10; load = :line_normal, Cs = 0.0).δavg ≈ 156.25 rtol = 0.005
        @test ssrc_plate(2.0, 20, 2; load = :line_normal, Cs = 0.0).δavg ≈ 151.78 rtol = 1e-3   # 2.9% low
        # orientation invariance: same problem in the XY plane
        @test ssrc_plate(2.0, 20, 2; load = :line_normal, plane = (1, 2, 3)).δavg ≈
              ssrc_plate(2.0, 20, 2; load = :line_normal, plane = (3, 1, 2)).δavg rtol = 1e-8
    end

    @testset "SSRC 2025 column buckling (Sec. 5.5)" begin
        # flexural buckling of the 100 × 1000 plate with shear deformation: 0.65797 N/mm² (t = 2),
        # 1603.78 N/mm² (t = 100)
        @test plate_buckling(1000.0, 100.0, 2.0, 30, 4; column = true).σcr ≈ 0.65797 rtol = 0.01
        @test plate_buckling(1000.0, 100.0, 2.0, 92, 10; column = true).σcr ≈ 0.65797 rtol = 0.002
        @test plate_buckling(1000.0, 100.0, 100.0, 30, 4; column = true).σcr ≈ 1603.78 rtol = 0.01
    end

    @testset "simply supported plate, k = 4 (Cs = 0.1 default)" begin
        @test plate_buckling(b, b, t, 8, 8).k ≈ 4.0 rtol = 0.02
        @test plate_buckling(b, b, t, 10, 10).k ≈ 4.0 rtol = 0.02
        @test plate_buckling(b, b, t, 16, 16).k ≈ 4.0 rtol = 0.02
        @test plate_buckling(b, b, t, 24, 24).k ≈ 4.0 rtol = 0.02
        @test plate_buckling(3b, b, t, 30, 10).k ≈ 4.0 rtol = 0.02   # a/b = 3, three half-waves
    end

    @testset "unrelaxed element (Cs = 0, Ádány MATLAB q42) locks on a coarse mesh" begin
        k0 = plate_buckling(b, b, t, 10, 10; Cs = 0.0).k
        @test k0 ≈ 4.561 rtol = 1e-3                                  # documented value
        @test plate_buckling(b, b, t, 24, 24; Cs = 0.0).k ≈ 4.0 rtol = 0.01   # converges with refinement
        @test plate_buckling(b, b, 0.2, 10, 10; Cs = 0.0).k > 8.0    # thin plate: locking blows up
    end

    @testset "thickness independence with Cs = 0.1" begin
        # b/t = 460 … 46; the thickest case carries a genuine ~1.5% Mindlin shear reduction
        ks = [plate_buckling(b, b, tt, 10, 10).k for tt in (0.2, 0.5, t, 2.0)]
        @test all(isapprox.(ks, 4.0; rtol = 0.03))
        @test maximum(ks) - minimum(ks) < 0.10
        # the unrelaxed element on the same meshes spans k = 4.1 … 14.4
        @test plate_buckling(b, b, 0.2, 10, 10; Cs = 0.0).k - plate_buckling(b, b, 2.0, 10, 10; Cs = 0.0).k > 4.0
    end

    @testset "orientation invariance in 3D" begin
        kxy = plate_buckling(b, b, t, 8, 8; plane = (1, 2, 3)).k
        kyz = plate_buckling(b, b, t, 8, 8; plane = (3, 2, 1)).k
        kzx = plate_buckling(b, b, t, 8, 8; plane = (3, 1, 2)).k
        @test kxy ≈ kyz rtol = 1e-8
        @test kxy ≈ kzx rtol = 1e-8
    end

    @testset "clamped square plate, k ≈ 10.07" begin
        @test plate_buckling(b, b, t, 16, 16; clamped = true).k ≈ 10.07 rtol = 0.02
        @test plate_buckling(b, b, t, 24, 24; clamped = true).k ≈ 10.07 rtol = 0.02
    end

    @testset "torsion of folded strips (drilling dof treatment)" begin
        # Saint-Venant J from a warping-free static twist (Moen 2008, Sec. 4.2.7.3.2.3): T = G J β′.
        # Thin-walled J = Σ b t³/3; the exact rectangle value carries the free-edge factor 1 − 0.63 t/b.
        tt = 1.0
        flat = torsion_J(strip_section(0.0, 76.2), tt)                      # 3 in strip, 8 elements across
        @test flat.J / flat.Jthin ≈ 1 - 0.63 * tt / 76.2 rtol = 0.02
        @test flat.Fnet < 1e-8
        # the flat strip does not care about the drilling treatment
        @test torsion_J(strip_section(0.0, 76.2), tt; drilling = :penalty).J ≈ flat.J rtol = 1e-6
        # folding the strip in two at any angle must not change J (the twisting moment crosses the fold);
        # each half has its own free-edge correction, so the 90° angle is slightly below the flat strip
        for α in (5.0, 30.0, 90.0)
            fold = torsion_J(strip_section(α, 76.2), tt)
            @test fold.J ≈ flat.J rtol = 0.005
            @test fold.Fnet < 1e-8                                          # pure torsion, no net end force
        end
        # the legacy absolute penalty stiffens the fold and leaves a spurious net end force
        pen = torsion_J(strip_section(90.0, 76.2), tt; drilling = :penalty)
        @test pen.Fnet > 1e-4
        # rounded lipped C (Abaqus-like mesh: 4 elements per flat, 5 per corner): J ≈ 0.987 Σbt³/3 with
        # Hughes–Brezzi (exact Saint-Venant value is 0.9956 Σbt³/3). With the penalty J is far too large
        # and, because of the spurious net end force, depends on the point the twist is applied about
        # (1.29 × here about the section's mean point, 2.1 × about the shear center)
        c = torsion_J(lipped_c_section(), 1.88)
        @test c.J / c.Jthin ≈ 0.987 rtol = 0.01
        @test c.Fnet < 1e-6
        @test torsion_J(lipped_c_section(), 1.88; drilling = :penalty).J / c.Jthin > 1.2
        # insensitive to the Hughes–Brezzi factor
        @test torsion_J(lipped_c_section(), 1.88; drilling_gamma = 0.1).J ≈ c.J rtol = 0.02
        @test torsion_J(lipped_c_section(), 1.88; drilling_gamma = 10.0).J ≈ c.J rtol = 0.005
    end
end
