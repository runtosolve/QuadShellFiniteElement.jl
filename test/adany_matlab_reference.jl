# Line-by-line Julia transcription of S. Ádány's MATLAB 4-node element routines
# (to_Cris_Sept15_2026: GL4.m, ke_uv_4n_condens_from_12to8dof_num.m,
# ke_wt_4n_condens_from_18to12dof_num.m, ct_4node_g2e.m, add_drill.m, rotate3d.m).
# Used only by the test suite as an independent reference for QuadShellFiniteElement.jl.
# Explicit shape function derivatives and Gauss data, no Ferrite.

module AdanyReference

using LinearAlgebra

# Gauss-Legendre data on [-1,1]², np = 1…4 points per direction (GL4.m)
function GL4(np)
    if np == 1
        locx = [0.0;;]; locy = [0.0;;]; wei = [4.0;;]
    elseif np == 2
        g = 1 / sqrt(3)
        locx = [-g -g; g g]; locy = [-g g; -g g]; wei = [1.0 1.0; 1.0 1.0]
    elseif np == 3
        g = sqrt(3 / 5)
        locx = [-g -g -g; 0 0 0; g g g]; locy = [-g 0 g; -g 0 g; -g 0 g]
        wei = [5, 8, 5] * [5 8 5] / 81
    elseif np == 4
        g1 = sqrt(3/7 + 2/7 * sqrt(6/5)); g2 = sqrt(3/7 - 2/7 * sqrt(6/5))
        locx = [-g1 -g1 -g1 -g1; -g2 -g2 -g2 -g2; g2 g2 g2 g2; g1 g1 g1 g1]
        locy = [-g1 -g2 g2 g1; -g1 -g2 g2 g1; -g1 -g2 g2 g1; -g1 -g2 g2 g1]
        w = [18 - sqrt(30), 18 + sqrt(30), 18 + sqrt(30), 18 - sqrt(30)]
        wei = w * w' / 36^2
    end
    return locx, locy, wei
end

function Dmat(Ex, Ey, nuxy, nuyx, G)
    E11 = Ex / (1 - nuxy * nuyx)
    E22 = Ey / (1 - nuxy * nuyx)
    E12 = nuyx * Ex / (1 - nuxy * nuyx)
    E21 = E12                                  # mat_sym = 1
    return [E11 E12 0; E21 E22 0; 0 0 G]
end

function jacobian(P, x, y)
    xe1, ye1 = P[1][1], P[1][2]; xe2, ye2 = P[2][1], P[2][2]
    xe3, ye3 = P[3][1], P[3][2]; xe4, ye4 = P[4][1], P[4][2]
    J = [xe1*(y/4 - 1/4) - xe2*(y/4 - 1/4) + xe3*(y/4 + 1/4) - xe4*(y/4 + 1/4)   ye1*(y/4 - 1/4) - ye2*(y/4 - 1/4) + ye3*(y/4 + 1/4) - ye4*(y/4 + 1/4)
         xe1*(x/4 - 1/4) - xe2*(x/4 + 1/4) + xe3*(x/4 + 1/4) - xe4*(x/4 - 1/4)   ye1*(x/4 - 1/4) - ye2*(x/4 + 1/4) + ye3*(x/4 + 1/4) - ye4*(x/4 - 1/4)]
    return J
end

# shape functions and reference derivatives at (x, y), 6 functions
function shapes(x, y)
    N  = [(x-1)*(y-1)/4, -(x+1)*(y-1)/4, (x+1)*(y+1)/4, -(x-1)*(y+1)/4, 1 - x^2, 1 - y^2]
    Nx = [(y-1)/4, -(y-1)/4, (y+1)/4, -(y+1)/4, -2x, 0.0]
    Ny = [(x-1)/4, -(x+1)/4, (x+1)/4, -(x-1)/4, 0.0, -2y]
    return N, Nx, Ny
end

# ke_uv_4n_condens_from_12to8dof_num.m
function ke_uv_4n_condens_from_12to8dof_num(P1, P2, P3, P4, t, Ex, Ey, nuxy, nuyx, G, nGp)
    P = (P1, P2, P3, P4)
    Duv = Dmat(Ex, Ey, nuxy, nuyx, G) * t
    np = round(Int, sqrt(nGp))
    locx, locy, wei = GL4(np)
    kuv = zeros(12, 12)
    for i in 1:np, j in 1:np
        x = locx[i, j]; y = locy[i, j]
        J = jacobian(P, x, y); detJ = det(J); Ji = inv(J)
        Ji11, Ji12, Ji21, Ji22 = Ji[1, 1], Ji[1, 2], Ji[2, 1], Ji[2, 2]
        _, Nx, Ny = shapes(x, y)
        Buv = zeros(3, 12)
        for k in 1:6
            dx = Nx[k]*Ji11 + Ny[k]*Ji12
            dy = Nx[k]*Ji21 + Ny[k]*Ji22
            Buv[1, 2k-1] = dx
            Buv[2, 2k]   = dy
            Buv[3, 2k-1] = dy
            Buv[3, 2k]   = dx
        end
        kuv += Buv' * Duv * Buv * detJ * wei[i, j]
    end
    inda = 1:8; indi = 9:12
    return kuv[inda, inda] - kuv[inda, indi] * inv(kuv[indi, indi]) * kuv[indi, inda]
end

# ke_wt_4n_condens_from_18to12dof_num.m
function ke_wt_4n_condens_from_18to12dof_num(P1, P2, P3, P4, t, Ex, Ey, nuxy, nuyx, G, nGp)
    P = (P1, P2, P3, P4)
    D = Dmat(Ex, Ey, nuxy, nuyx, G)
    Dwtb = D * t^3 / 12
    Dwts = Matrix(1.0I, 2, 2) * 5/6 * G * t
    np = round(Int, sqrt(nGp))
    locx, locy, wei = GL4(np)
    kwts = zeros(18, 18); kwtb = zeros(18, 18)
    for i in 1:np, j in 1:np
        x = locx[i, j]; y = locy[i, j]
        J = jacobian(P, x, y); detJ = det(J); Ji = inv(J)
        Ji11, Ji12, Ji21, Ji22 = Ji[1, 1], Ji[1, 2], Ji[2, 1], Ji[2, 2]
        N, Nx, Ny = shapes(x, y)
        Bwts = zeros(2, 18); Bwtb = zeros(3, 18)
        for k in 1:6
            dx = Nx[k]*Ji11 + Ny[k]*Ji12
            dy = Nx[k]*Ji21 + Ny[k]*Ji22
            Bwts[1, 3k-2] = dx;  Bwts[1, 3k] = N[k]
            Bwts[2, 3k-2] = dy;  Bwts[2, 3k-1] = -N[k]
            Bwtb[1, 3k] = dx
            Bwtb[2, 3k-1] = -dy
        end
        Bwtb[3, 2:3:18] = -Bwtb[1, 3:3:18]
        Bwtb[3, 3:3:18] = -Bwtb[2, 2:3:18]
        kwts += Bwts' * Dwts * Bwts * detJ * wei[i, j]
        kwtb += Bwtb' * Dwtb * Bwtb * detJ * wei[i, j]
    end
    inda = 1:12; indi = 13:18
    kwt1 = kwts + kwtb
    return kwt1[inda, inda] - kwt1[inda, indi] * inv(kwt1[indi, indi]) * kwt1[indi, inda]
end

# ct_4node_g2e.m  (nodes given as 3-vectors)
function ct_4node_g2e(P1g, P2g, P3g, P4g)
    P12 = (P1g + P2g) / 2; P34 = (P3g + P4g) / 2
    P23 = (P2g + P3g) / 2; P41 = (P4g + P1g) / 2
    norm_vec = cross(P23 - P41, P34 - P12)
    norm_vec = norm_vec / norm(norm_vec)
    j3 = norm_vec
    P0 = P12 + (P34 - P12) / 2
    proj(Pg) = (Pg - P0) - dot(Pg - P0, norm_vec) * norm_vec
    P1 = proj(P1g); P2 = proj(P2g); P3 = proj(P3g); P4 = proj(P4g)
    P12 = (P1 + P2) / 2; P34 = (P3 + P4) / 2
    P23 = (P2 + P3) / 2; P41 = (P4 + P1) / 2
    j1 = (P23 - P41) / norm(P23 - P41)
    j2 = cross(j3, j1)
    T = [j1 j2 j3]
    return T, T' * P1, T' * P2, T' * P3, T' * P4
end

# add_drill.m
function add_drill(k5dpn, nodenr, ifelastic)
    ind = Int[]; i0 = 0
    for i in 1:nodenr
        append!(ind, (1:5) .+ i0); i0 += 6
    end
    k6dpn = zeros(6nodenr, 6nodenr)
    k6dpn[ind, ind] = k5dpn
    if ifelastic
        ind = Int[]; i0 = 0
        for i in 1:nodenr
            append!(ind, [4, 5] .+ i0); i0 += 5
        end
        kd = diag(k5dpn)
        stif = minimum(kd[ind]) / 100
        for i in 1:nodenr
            k6dpn[6i, 6i] = stif
        end
    end
    return k6dpn
end

# rotate3d.m, dpn = 6
function rotate3d(T3, nodenr)
    T = Matrix(1.0I, 6nodenr, 6nodenr)
    i0 = 0
    for i in 1:nodenr
        ind = [1, 2, 3] .+ i0; T[ind, ind] = T3
        ind = [4, 5, 6] .+ i0; T[ind, ind] = T3
        i0 += 6
    end
    return T
end

# stiffmat_e.m lines 141-159: full 24×24 global element stiffness for one q42 element
function element_ke_global(P1g, P2g, P3g, P4g, t, E, ν)
    G = E / (2(1 + ν))
    T, P1e, P2e, P3e, P4e = ct_4node_g2e(P1g, P2g, P3g, P4g)
    ke_uv = ke_uv_4n_condens_from_12to8dof_num(P1e, P2e, P3e, P4e, t, E, E, ν, ν, G, 4)
    ke_wt = ke_wt_4n_condens_from_18to12dof_num(P1e, P2e, P3e, P4e, t, E, E, ν, ν, G, 9)
    induv = Int[]; indwt = Int[]; n0 = 0
    for i in 1:4
        append!(induv, n0 .+ [1, 2]); append!(indwt, n0 .+ [3, 4, 5]); n0 += 5
    end
    ke = zeros(20, 20)
    ke[induv, induv] = ke_uv
    ke[indwt, indwt] = ke_wt
    ke24 = add_drill(ke, 4, true)
    TT = rotate3d(T, 4)
    return TT * ke24 * TT', ke24, T, (P1e, P2e, P3e, P4e)
end

end # module
