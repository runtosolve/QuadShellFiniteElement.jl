# QuadShellFiniteElement.jl

`QuadShellFiniteElement.jl` is a Julia package that implements a **4-node quadrilateral shell finite element** with drilling degrees of freedom (DOF). The element combines membrane, Mindlin plate bending, and transverse shear stiffness contributions into a 24-DOF shell element suitable for linear static and elastic buckling analysis of shell structures. It is the quadrilateral companion of [TriShellFiniteElement.jl](https://github.com/runtosolve/TriShellFiniteElement.jl) and follows the same code structure and conventions.

The element formulation is the "4+2" Mindlin quadrilateral of S. Ádány: the five shell displacement functions (u, v, w, θₓ, θᵧ) are interpolated with the four bilinear corner shape functions plus two quadratic bubble functions N₅ = 1 − ξ², N₆ = 1 − η² whose amplitudes are removed by static condensation. It is documented in

> Moen, C.D. and Ádány, S. (2025). "Thin shell finite element formulations implemented in open-source software." *Proceedings of the Annual Stability Conference, Structural Stability Research Council*, Louisville, Kentucky.

and coded in MATLAB as `ke_uv_4n_condens_from_12to8dof_num.m` and `ke_wt_4n_condens_from_18to12dof_num.m` (element type `q42`). The Julia element reproduces those MATLAB matrices to machine precision (see `test/`).

---

## Key Features

- 4-node quadrilateral shell element with 6 DOF per node (u, v, w, θₓ, θᵧ, θᵤ)
- Composite element formulation combining:
  - In-plane membrane stiffness (4 + 2 shape functions, 12 → 8 DOF by static condensation, 2×2 Gauss)
  - Out-of-plane Mindlin–Reissner plate bending and transverse shear stiffness with 5/6 shear correction factor (4 + 2 shape functions, 18 → 12 DOF by static condensation, 3×3 Gauss)
  - Drilling DOF stabilization (penalty stiffness of 1/100 of the smallest rotational diagonal term)
  - Optional Tessler–Hughes type shear relaxation factor `Cs` to remove residual shear locking in thin plates
- Geometric stiffness matrix for linear buckling analysis (Moen & Ádány 2025, Eq. 11–13)
- Membrane stress recovery in the element local frame and a consistent uniform pressure load vector
- Integration with [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/) for mesh handling, DOF management, quadrature and shape function gradients
- Local-to-global coordinate transformation for arbitrarily oriented and mildly warped quadrilaterals in 3D space (average-plane projection of `ct_4node_g2e.m`)

---

## Element Formulation

The element has **24 DOF total** (4 nodes × 6 DOF/node):

```
Node DOF: [u, v, w, θₓ, θᵧ, θᵤ]
```

- **u, v** — in-plane translations (membrane)
- **w** — out-of-plane translation (bending/shear)
- **θₓ, θᵧ** — bending rotations, with γₓᵤ = ∂w/∂x + θᵧ and γᵧᵤ = ∂w/∂y − θₓ
- **θᵤ** — drilling rotation (in-plane rotation)

The local element stiffness is assembled from:

| Contribution | Uncondensed | Condensed | Notes |
|---|---|---|---|
| Membrane | 12×12 | 8×8 | Plane stress, 6 shape functions × 2 DOF, 2×2 Gauss |
| Bending + shear | 18×18 | 12×12 | Mindlin plate, 6 shape functions × 3 DOF (w, θₓ, θᵧ), 3×3 Gauss, condensed together |
| Drilling | — | 4 diagonal terms | Penalty stiffness |

Element matrices are formed in a local planar frame whose normal is perpendicular to the average plane of the four (possibly warped) nodes, then rotated to global XYZ. In the assembled global matrices the DOFs follow the Ferrite field order (`:u` translations of the four nodes, then `:θ` rotations).

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/runtosolve/QuadShellFiniteElement.jl")
```

---

## Basic Usage

### Element-Level Stiffness Matrices

```julia
using Ferrite, Tensors, QuadShellFiniteElement

E = 200_000.0   # Young's modulus (MPa)
ν = 0.30        # Poisson's ratio
t = 1.0         # Shell thickness (mm)

# Element node coordinates in the local 2D system, counter-clockwise
x = [Tensors.Vec((0.0, 0.0)),
     Tensors.Vec((100.0, 0.0)),
     Tensors.Vec((100.0, 100.0)),
     Tensors.Vec((0.0, 100.0))]

ip4 = IP4()      # 4 bilinear shape functions (geometry, geometric stiffness)
ip6 = IP6()      # 4 bilinear + 2 bubble shape functions (elastic stiffness)
qr_m = QuadratureRule{RefQuadrilateral}(2)   # 2×2 Gauss, membrane
qr_b = QuadratureRule{RefQuadrilateral}(3)   # 3×3 Gauss, bending + shear

# 24×24 local element stiffness, node dofs [u v w θx θy θz]
Ke = QuadShellFiniteElement.local_elastic_stiffness_matrix!(qr_m, qr_b, ip4, ip6, E, ν, t, x)

# Uncondensed building blocks
cv = CellValues(qr_b, ip6, ip4)
reinit!(cv, x)
Dm = QuadShellFiniteElement.calculate_membrane_constitutive_matrix(E, ν, t)
Db = QuadShellFiniteElement.calculate_bending_constitutive_matrix(E, ν, t)
Ds = QuadShellFiniteElement.calculate_shear_constitutive_matrix(E, ν, t)
Ke_m = QuadShellFiniteElement.calculate_element_membrane_stiffness_matrix(Dm, cv)  # 12×12
Ke_b = QuadShellFiniteElement.calculate_element_bending_stiffness_matrix(Db, cv)   # 18×18
Ke_s = QuadShellFiniteElement.calculate_element_shear_stiffness_matrix(Ds, cv)     # 18×18
Ke_wt = QuadShellFiniteElement.static_condensation(Ke_b + Ke_s, 12)                # 12×12
```

### Global Stiffness Assembly

```julia
using Ferrite, QuadShellFiniteElement

E = 200_000.0
ν = 0.30
t = 1.0

# Build a quadrilateral mesh with 3D nodes and the DOF handler
grid = let grid_2D = generate_grid(Quadrilateral, (10, 10), Vec((0.0, 0.0)), Vec((1000.0, 1000.0)))
    Grid(grid_2D.cells, [Node((n.x[1], n.x[2], 0.0)) for n in grid_2D.nodes])
end
ip = Lagrange{RefQuadrilateral, 1}()
dh = DofHandler(grid)
add!(dh, :u, ip^3)   # u, v, w translations
add!(dh, :θ, ip^3)   # θₓ, θᵧ, θᵤ rotations
close!(dh)

ip4 = IP4(); ip6 = IP6()
qr_m = QuadratureRule{RefQuadrilateral}(2)
qr_b = QuadratureRule{RefQuadrilateral}(3)

# Assemble the global elastic stiffness
K = allocate_matrix(dh)
K = QuadShellFiniteElement.assemble_global_Ke!(K, dh, qr_m, qr_b, ip4, ip6, E, ν, t)

# Consistent nodal loads of a uniform pressure p along the element normals
f = zeros(ndofs(dh))
QuadShellFiniteElement.assemble_pressure_load!(f, dh, qr_m, ip4, 0.001)
```

### Buckling Analysis

After solving the linear static problem `K u = f` under the reference load:

```julia
# membrane stresses in each element's local frame at the 2×2 Gauss points
qr_g = QuadratureRule{RefQuadrilateral}(2)
σXX, σYY, τXY = QuadShellFiniteElement.element_membrane_stresses(dh, u, ip4, E, ν, t; qr = qr_g)

# geometric stiffness from the stress resultants σ·t
Kg = allocate_matrix(dh)
Kg = QuadShellFiniteElement.assemble_global_Kg!(Kg, dh, qr_g, ip4, σXX .* t, σYY .* t, τXY .* t)

# Solve the generalized eigenvalue problem K·φ = λ·(-Kg)·φ on the free dofs
# using your preferred eigensolver; λ scales the reference load to the buckling load.
```

---

## Shear Relaxation Factor `Cs`

With `Cs = 0` the element is exactly the MATLAB `q42` element. Its two condensed bubble modes remove the shear locking of the bilinear Mindlin quadrilateral in beam-like bending, but not in general plate bending: for thin plates meshed with elements much larger than the thickness the parasitic shear stiffness still dominates and buckling loads come out too high. To control this the transverse shear stiffness is scaled by `1 / (1 + Cs * alpha)`, where `alpha` is the ratio of the element shear to bending rotational stiffness (a Tessler–Hughes type relaxation), as in `TriShellFiniteElement.jl`.

`Cs` is a keyword argument of `assemble_global_Ke!` and `local_elastic_stiffness_matrix!` with default `QuadShellFiniteElement.DEFAULT_SHEAR_RELAXATION = 0.1`, calibrated on plate buckling benchmarks (`test/runtests.jl`). Simply supported square plate in uniform compression, exact k = 4.0, thickness/width = 1/105:

| Elements across width | `Cs = 0.0` | `Cs = 0.05` | `Cs = 0.1` | `Cs = 0.2` |
|---|---|---|---|---|
| 8  | 5.38 | 4.10 | 3.94 | 3.79 |
| 10 | 4.56 | 4.04 | 3.94 | 3.83 |
| 16 | 4.07 | 3.98 | 3.94 | 3.89 |
| 24 | 3.99 | 3.97 | 3.95 | 3.92 |

With `Cs = 0` the result at a fixed mesh also depends strongly on thickness (10×10 mesh: k = 14.4, 5.70, 4.56, 4.06 for b/t = 460, 184, 105, 46); with `Cs = 0.1` it is thickness independent (k = 3.95, 3.95, 3.94, 3.88). The clamped square plate gives k = 9.96 at 16×16 (exact ≈ 10.07). Pass `Cs = 0.0` to recover the unrelaxed element:

```julia
K = QuadShellFiniteElement.assemble_global_Ke!(K, dh, qr_m, qr_b, ip4, ip6, E, ν, t; Cs = 0.0)
```

---

## Validation against Moen & Ádány (2025)

The SSRC 2025 paper studies a 100 mm × 1000 mm plate (E = 200000 MPa, ν = 0.30), simply supported at the short ends, for a thick (t = 100 mm) and a thin (t = 2 mm) case. `test/runtests.jl` reproduces the examples of Section 5 with this element (`Cs = 0.1`):

| Example | Mesh | Analytical | QuadShellFiniteElement.jl |
|---|---|---|---|
| Out-of-plane bending, midspan line load 1 N/mm, t = 2 | 20 × 2 | 156.25 mm | 156.46 mm |
| Out-of-plane bending, midspan line load 1 N/mm, t = 2 | 92 × 10 | 156.25 mm | 156.12 mm |
| Out-of-plane bending, uniform pressure 0.001 N/mm², t = 2 | 20 × 2 | 97.657 mm | 97.57 mm |
| Out-of-plane bending, midspan line load 1000 N/mm, t = 100 | 92 × 10 | 1.289 mm | 1.290 mm |
| In-plane bending, midspan line load 1000 N/mm, t = 2 | 92 × 10 | 64.45 mm | 64.99 mm |
| Column buckling stress, t = 2 | 92 × 10 | 0.65797 N/mm² | 0.6586 N/mm² |
| Column buckling stress, t = 100 | 30 × 4 | 1603.78 N/mm² | 1602 N/mm² |

The thick cases include the Mindlin shear deformation (Euler–Bernoulli would give 1.25 mm).

---

## Tests

```julia
using Pkg; Pkg.test("QuadShellFiniteElement")
```

`test/runtests.jl` checks the element matrices on a distorted, warped and tilted quadrilateral against two independent references: `test/reference/octave_q42/`, the matrices written by S. Ádány's unmodified MATLAB routines (`ct_4node_g2e.m`, `ke_uv_4n_condens_from_12to8dof_num.m`, `ke_wt_4n_condens_from_18to12dof_num.m`, `add_drill.m`, `rotate3d.m`) run in GNU Octave with `test/reference/make_reference.m`, and `test/adany_matlab_reference.jl`, a line-by-line Julia transcription of the same routines. Agreement is to round-off (relative difference of order 1e-15). The suite also verifies symmetry, six rigid body modes and a constant-strain membrane patch test; runs the SSRC 2025 benchmarks above; and solves simply supported and clamped rectangular plate buckling problems, including thickness independence and invariance to the plate's orientation in 3D. `test/benchmarks.jl` holds the mesh and solver drivers shared by the tests.

---

## Dependencies

| Package | Role |
|---|---|
| [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) | Mesh, DOF handler, cell values, quadrature |
| [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl) | Coordinate vectors and tensor operations |
| LinearAlgebra | Matrix operations (standard library) |

---
