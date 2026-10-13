# --------------------------------------------------------------------------------------------
# Incremental Newton–Raphson solver for the elastic–plastic, geometrically nonlinear shell model.
# Shared verbatim between TriShellFiniteElement.jl and QuadShellFiniteElement.jl.
# --------------------------------------------------------------------------------------------

# K[i, i] += v[i] in place (the diagonal is always part of the Ferrite sparsity pattern)
function add_diagonal!(K::SparseMatrixCSC, v)
    rows = rowvals(K); vals = nonzeros(K)
    for j in 1:size(K, 2), k in nzrange(K, j)
        if rows[k] == j
            vals[k] += v[j]
            break
        end
    end
    return K
end

"""
    solve_nonlinear(model, ch; reaction_dofs = Int[], f_ext = nothing, Δt = 0.01, Δt_min = 1e-5,
                    Δt_max = 0.05, t_end = 1.0, rtol = 1e-6, atol = 1e-10, maxiter = 25,
                    grow_after = 5, max_line_search = 3, save_every = 0, verbose = false,
                    callback = nothing, damping = 0.0, damping_ratio = 2e-4, mass = nothing)

Incremental Newton–Raphson solution of the nonlinear shell model over the pseudo-time
`t ∈ [0, t_end]`, with adaptive step size. `ch` is a Ferrite `ConstraintHandler` whose Dirichlet
conditions are functions of `t` (displacement control); `f_ext(t)` optionally returns an external
load vector (load control, or both). The equilibrium residual is `r = f_int(u) − f_ext(t)`; a step
converges when `‖r_free‖₂ ≤ max(rtol ‖f_int‖₂, atol)`. A Newton correction that increases the
residual norm is halved up to `max_line_search` times (backtracking line search), which keeps the
iteration from flip-flopping when integration points switch between plastic loading and elastic
unloading near a collapse mechanism. A step that does not converge in `maxiter`
iterations is retried with a quarter of the increment, and the increment is grown by 1.5 after
steps converging in at most `grow_after` iterations. Each step starts from a tangent predictor
(the linearized problem at the last converged state for the new prescribed values and loads),
which matters for displacement control: without it the elements next to a loaded edge see the
whole prescribed increment and the first Newton iteration starts from a spurious plastic state.

Artificial damping (the ABAQUS `*STATIC, STABILIZE` idea, Moen 2008 Sec. 7.1.2.2): with
`damping = c > 0` the viscous forces `c M (u − uₙ) / Δt` are added to the residual, `M` being the
lumped mass of the shell with unit density (`mass`, by default [`lumped_mass`](@ref)(model)) and
`Δt` the pseudo-time increment, so that local instabilities (mode jumps, fold formation) that have
no static equilibrium path under displacement control can be passed. `damping = :auto` picks `c`
so that the viscous work of the first increment is `damping_ratio` times its strain energy
(ABAQUS default 2e-4). The histories `W_ext` (external work, from the reactions at the prescribed
dofs and `f_ext`) and `W_visc` (work of the viscous forces) are returned to check that the damping
stays negligible.

Returns a named tuple with the converged step histories: `t`, `P` (sum of the internal forces at
`reaction_dofs`, i.e. the reaction of a loaded edge), `iterations`, `W_ext`, `W_visc`, the damping
factor `c`, the final solution `u`, the solutions `snapshots` saved every `save_every` steps
(`0` = only the last), their times `snapshot_t`, and `converged` (`false` if the increment fell
below `Δt_min`). `callback(t, u, model)` is called after every converged step.
"""
function solve_nonlinear(model, ch::ConstraintHandler; reaction_dofs = Int[], f_ext = nothing,
                         Δt = 0.01, Δt_min = 1e-5, Δt_max = 0.05, t_end = 1.0, rtol = 1e-6, atol = 1e-10,
                         maxiter = 25, grow_after = 5, max_line_search = 3, save_every = 0, verbose = false,
                         callback = nothing, damping = 0.0, damping_ratio = 2e-4, mass = nothing)

    dh = model.dh
    n = ndofs(dh)
    K = allocate_matrix(dh)
    f = zeros(n)
    r = zeros(n)
    u = zeros(n)
    u_try = zeros(n)
    u_ls = zeros(n)
    fext = zeros(n)
    f_prev = zeros(n)
    fext_prev = zeros(n)

    hist_t = Float64[]; hist_P = Float64[]; hist_it = Int[]; hist_Wext = Float64[]; hist_Wvisc = Float64[]
    snapshots = Vector{Vector{Float64}}(); snapshot_t = Float64[]
    push!(hist_t, 0.0); push!(hist_P, 0.0); push!(hist_it, 0); push!(hist_Wext, 0.0); push!(hist_Wvisc, 0.0)
    W_ext = 0.0; W_visc = 0.0
    prescribed = ch.prescribed_dofs

    auto_damping = damping == :auto
    c = auto_damping ? 0.0 : float(damping)
    m = (c > 0 || auto_damping) ? (mass === nothing ? lumped_mass(model) : mass) : zeros(n)
    visc = zeros(n)

    t = 0.0
    step = 0
    converged_all = true
    while t < t_end - 1e-12 * t_end
        Δt = min(Δt, t_end - t)
        t_try = t + Δt
        update!(ch, t_try)
        copyto!(u_try, u)
        apply!(u_try, ch)
        f_ext === nothing ? fill!(fext, 0.0) : copyto!(fext, f_ext(t_try))

        # tangent predictor: linearize at the last converged state for the new prescribed values
        # and external load, so that a prescribed displacement increment is distributed through the
        # mesh before the first residual evaluation
        assemble_internal!(K, f, model, u)
        cm = c / Δt
        r .= K * (u_try .- u) .+ f .- fext .+ cm .* m .* (u_try .- u)
        cm > 0 && add_diagonal!(K, cm .* m)
        apply_zero!(K, r, ch)
        u_try .-= K \ r

        converged = false
        nit = 0
        assemble_internal!(K, f, model, u_try)
        visc .= cm .* m .* (u_try .- u)
        r .= f .- fext .+ visc
        cm > 0 && add_diagonal!(K, cm .* m)
        apply_zero!(K, r, ch)
        res = norm(r)
        for it in 1:maxiter
            nit = it
            ref = max(norm(f), norm(fext))
            verbose && println("  t = $(round(t_try, sigdigits = 6))  it = $it  ‖r‖ = $(res)  ‖f‖ = $(ref)")
            if res <= max(rtol * ref, atol)
                converged = true
                break
            end
            Δu = K \ r
            any(isnan, Δu) && break
            # backtracking line search on the residual norm
            α = 1.0
            res_new = Inf
            for ls in 0:max_line_search
                u_ls .= u_try .- α .* Δu
                assemble_internal!(K, f, model, u_ls)
                visc .= cm .* m .* (u_ls .- u)
                r .= f .- fext .+ visc
                cm > 0 && add_diagonal!(K, cm .* m)
                apply_zero!(K, r, ch)
                res_new = norm(r)
                (res_new <= res || ls == max_line_search) && break
                α /= 2
                verbose && println("    line search: ‖r‖ = $(res_new) > $(res), α → $α")
            end
            copyto!(u_try, u_ls)
            res = res_new
        end

        if converged
            commit!(model)
            Δu_step = u_try .- u
            # work of the reactions at the prescribed dofs (trapezoidal) and of the external load
            W_ext += sum(d -> 0.5 * (f[d] + f_prev[d]) * Δu_step[d], prescribed; init = 0.0) +
                     sum(d -> 0.5 * (fext[d] + fext_prev[d]) * Δu_step[d], 1:n; init = 0.0)
            W_visc += dot(visc, Δu_step)
            if auto_damping && step == 0
                # choose c so that the viscous work of this increment is damping_ratio × strain energy
                W_v1 = sum(m .* Δu_step .^ 2) / Δt           # viscous work for c = 1
                E_s = 0.5 * dot(f, u_try)                     # strain energy (elastic first step)
                c = damping_ratio * E_s / W_v1
                verbose && println("automatic damping factor c = $c")
            end
            copyto!(f_prev, f); copyto!(fext_prev, fext)
            copyto!(u, u_try)
            t = t_try
            step += 1
            push!(hist_t, t); push!(hist_it, nit); push!(hist_Wext, W_ext); push!(hist_Wvisc, W_visc)
            push!(hist_P, isempty(reaction_dofs) ? 0.0 : sum(f[d] for d in reaction_dofs))
            if save_every > 0 && step % save_every == 0
                push!(snapshots, copy(u)); push!(snapshot_t, t)
            end
            callback === nothing || callback(t, u, model)
            verbose && println("step $step converged, t = $(round(t, sigdigits = 6)), P = $(hist_P[end]), iterations = $nit")
            nit <= grow_after && (Δt = min(1.5 * Δt, Δt_max))
        else
            restore!(model)
            Δt /= 4
            verbose && println("step at t = $(round(t_try, sigdigits = 6)) did not converge, cutting the increment to $Δt")
            if Δt < Δt_min
                @warn "solve_nonlinear: increment below Δt_min at t = $t, stopping"
                converged_all = false
                break
            end
        end
    end

    if isempty(snapshots) || snapshot_t[end] != t
        push!(snapshots, copy(u)); push!(snapshot_t, t)
    end

    return (; t = hist_t, P = hist_P, iterations = hist_it, W_ext = hist_Wext, W_visc = hist_Wvisc, c, u,
            snapshots, snapshot_t, converged = converged_all)
end
