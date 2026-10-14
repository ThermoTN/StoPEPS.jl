abstract type UpdateMethod end
"""
    FullUpdate(; max_iterations=4, error_tolerance=1e-8,
        diagonal_shift=1e-10, linear_max_iterations=24, linear_tolerance=1e-9)

Alternating two-site full update with preconditioned conjugate-gradient solves.
`error_tolerance` controls the relative squared environment-weighted fit
residual; `linear_tolerance` controls each linear solve.
"""
Base.@kwdef struct FullUpdate <: UpdateMethod
    max_iterations::Int = 4
    error_tolerance::Float64 = 1e-8
    diagonal_shift::Float64 = 1e-10
    linear_max_iterations::Int = 24
    linear_tolerance::Float64 = 1e-9
end
const RESIDUAL_ROUNDOFF_TOLERANCE = 1e-8

function _reduced_pair(A,B,gate,direction,D;sector_space=nothing)
    pa=direction==:horizontal ? (1,2,6,4,3,5) : (1,2,5,4,3,6)
    pb=direction==:horizontal ? (2,5,6,4,3,1) : (1,5,6,4,3,2)
    Qa,Ca=left_orth(permute(A,(pa[1:4],pa[5:6])))
    Qb,Cb=left_orth(permute(B,(pb[1:4],pb[5:6])))
    a=_out(Ca); b=_perm(Cb,(3,2,1))
    @tensor target[r,p,q,s] := a[r,u,k]*b[k,v,s]*gate[p,q,u,v]
    M=permute(target,((1,2),(3,4)))
    cutoff=eps(Float64)*maximum(dim(space(target,k)) for k in 1:4)*norm(M)
    U,S,V,_=svd_trunc(M;trunc=_truncation(D,sector_space)&trunctol(atol=cutoff))
    dim(domain(U))>0 || error("gate produced a zero target")
    root=copy(S)
    for (_,block) in blocks(root), k in axes(block,1)
        block[k,k]=sqrt(real(block[k,k]))
    end
    return (;Qa=_out(Qa),Qb=_out(Qb),target,
        left=_out(U*root),right=_out(root*V),pa,pb)
end

function _restore_pair(pair,A,B)
    Qa,Qb=pair.Qa,pair.Qb
    @tensor first[u,v,w,a,p,b] := Qa[u,v,w,a,r]*A[r,p,b]
    @tensor second[u,v,w,a,p,b] := Qb[u,v,w,a,r]*B[b,p,r]
    return _perm(first,invperm(pair.pa)),_perm(second,invperm(pair.pb))
end

function _reduced_metric(H::AbstractTensorMap,Qa,Qb)
    @tensor N[rb,sb;rk,sk] := H[a,ab,b,bb,c,cb,d,db,e,eb,f,fb]*
        Qa[a,b,c,u,rk]*conj(Qa[ab,bb,cb,u,rb])*
        Qb[d,e,f,v,sk]*conj(Qb[db,eb,fb,v,sb])
    N
end

"""External double legs (ket,bra), with the two reduced-core legs left open."""
function _reduced_layer(Q)
    @tensor A[w,wb,n,nb,e,eb,rb,rk] := Q[w,n,e,u,rk]*conj(Q[wb,nb,eb,u,rb])
    for k in (5,3,1)
        A=_fuse(A,k,2)
    end
    A
end

function _reduced_metric(env::BondEnvironment,Qa,Qb)
    A=_reduced_layer(Qa); B=_reduced_layer(Qb)
    left,right=env.left,env.right
    if env.direction==:horizontal
        T1,T2=env.top; B1,B2=env.bottom
        @tensor X[b,n,c,w] := left[a,w,b]*T1[a,n,c]
        @tensor Y[b,c,s,rb,rk] := X[b,n,c,w]*A[w,n,s,rb,rk]
        @tensor L[c,f,rb,rk] := Y[b,c,s,rb,rk]*B1[b,s,f]
        @tensor U[c,v,e,g] := right[d,e,g]*T2[c,v,d]
        @tensor V[c,g,t,sb,sk] := U[c,v,e,g]*B[v,e,t,sb,sk]
        @tensor R[c,f,sb,sk] := V[c,g,t,sb,sk]*B2[f,t,g]
        @tensor N[rb,sb;rk,sk] := L[c,f,rb,rk]*R[c,f,sb,sk]
    else
        T=only(env.top); bottom=only(env.bottom)
        @tensor X[w,v,b,n,c] := left[a,w,v,b]*T[a,n,c]
        @tensor Y[v,b,c,e,rb,rk] := X[w,v,b,n,c]*A[w,n,e,rb,rk]
        @tensor L[v,b,f,d,rb,rk] := Y[v,b,c,e,rb,rk]*right[c,e,f,d]
        @tensor R[v,f,sb,sk,b,d] := B[v,f,s,sb,sk]*bottom[b,s,d]
        @tensor N[rb,sb;rk,sk] := L[v,b,f,d,rb,rk]*R[v,f,sb,sk,b,d]
    end
    N
end

function _apply_pair_metric(N,theta)
    @tensor output[r,p,q,s] := N[r,s,t,u]*theta[t,p,q,u]
    output
end
function _pair_tensor(A,B)
    @tensor pair[r,p,q,s] := A[r,p,b]*B[b,q,s]
    pair
end
function _residual_cost(N,A,B,target)
    residual=_pair_tensor(A,B)-target
    return dot(residual,_apply_pair_metric(N,residual))
end

function _repair_trigger(raw,target)
    scale=max(abs(target),floatmin(Float64))
    isfinite(raw) && isfinite(target) || error("nonfinite FU residual/target")
    reason=real(raw)/scale < -RESIDUAL_ROUNDOFF_TOLERANCE ? :negative_residual :
        abs(imag(raw))/scale > RESIDUAL_ROUNDOFF_TOLERANCE ? :complex_residual :
        real(target)<0 ? :negative_target_norm :
        abs(imag(target))/scale > RESIDUAL_ROUNDOFF_TOLERANCE ? :complex_target_norm : nothing
    return reason
end
function _cost_evaluation(raw,target)
    relative=real(raw)/max(abs(target),floatmin(Float64))
    isfinite(relative) || error("nonfinite relative FU residual")
    relative >= -RESIDUAL_ROUNDOFF_TOLERANCE || error("negative FU residual needs PSD repair")
    return (;cost=abs(Float64(relative)),fallback=relative<0,relative_cost=Float64(relative))
end

"""Jacobi-preconditioned CG entirely on TensorMaps and their native blocks."""
function _solve_core(metric,rhs,initial,options)
    diagonal=Dict(c=>max.(real.(diag(b)),0) for (c,b) in blocks(metric))
    scale=maximum((maximum(v;init=0.0) for v in values(diagonal));init=0.0)
    shift=options.diagonal_shift*max(scale,floatmin(Float64))
    floor=max(shift,eps(Float64)*scale,floatmin(Float64))
    function precondition(r)
        z=copy(r)
        for (c,b) in blocks(z)
            b ./= reshape(max.(diagonal[c].+shift,floor),:,1)
        end
        z
    end
    action(v)=metric*v+shift*v
    x=copy(initial); r=rhs-action(x); rhsnorm=max(norm(rhs),floatmin(Float64))
    residual=norm(r)/rhsnorm
    residual <= options.linear_tolerance && return x,residual
    z=precondition(r); p=copy(z); rz=real(dot(r,z))
    for iteration in 1:options.linear_max_iterations
        Ap=action(p); curvature=real(dot(p,Ap))
        curvature > eps(Float64)*max(norm(p)*norm(Ap),floatmin(Float64)) || break
        alpha=rz/curvature
        x=x+alpha*p; r=r-alpha*Ap; residual=norm(r)/rhsnorm
        residual <= options.linear_tolerance && break
        z=precondition(r); next=real(dot(r,z))
        next>=0 || break
        p=z+(next/max(rz,floatmin(Float64)))*p; rz=next
    end
    return x,Float64(residual)
end

function _left_normal_equation(metric,right,target)
    @tensor normal[first,bond;second,other] := metric[first,south,second,north]*
        conj(right[bond,physical,south])*right[other,physical,north]
    @tensor rhs[first,physical,bond] := metric[first,south,second,north]*
        target[second,physical,partner,north]*conj(right[bond,partner,south])
    normal,permute(rhs,((1,3),(2,)))
end

function _right_normal_equation(metric,left,target)
    @tensor normal[bond,south;other,north] := metric[west,south,east,north]*
        conj(left[west,physical,bond])*left[east,physical,other]
    @tensor rhs[bond,physical,south] := metric[west,south,east,north]*
        target[east,partner,physical,north]*conj(left[west,partner,bond])
    normal,permute(rhs,((1,3),(2,)))
end

function _optimize_pair(pair,N,options)
    A=copy(pair.left); B=copy(pair.right); target=pair.target
    target_norm=dot(target,_apply_pair_metric(N,target))
    raw=_residual_cost(N,A,B,target)
    trigger=_repair_trigger(raw,target_norm)
    trigger===nothing || return (;trigger,stage=:initial_local_svd,iteration=0,raw,target_norm)
    initial=_cost_evaluation(raw,target_norm)
    bestA,bestB=copy(A),copy(B); best=initial.cost; residual=0.0
    fallbacks=NamedTuple[]; iterations=0; iteration_limit_reached=false
    stop_reason=options.max_iterations==0 ? :disabled : :iteration_limit
    if initial.fallback
        stop_reason=:residual_fallback
        push!(fallbacks,(;iteration=0,relative_cost=initial.relative_cost,used_initial_local_svd=true))
    elseif options.max_iterations>0 && best<=options.error_tolerance
        # The initial gated SVD already meets the same normalized squared
        # residual threshold used after each ALS iteration.
        stop_reason=:error_tolerance
    else
        for iteration in 1:options.max_iterations
            MA,ya=_left_normal_equation(N,B,target)
            sol,ra=_solve_core(MA,ya,permute(A,((1,3),(2,))),options)
            A=_perm(sol,(1,3,2))
            MB,yb=_right_normal_equation(N,A,target)
            sol,rb=_solve_core(MB,yb,permute(B,((1,3),(2,))),options)
            B=_perm(sol,(1,3,2))
            # Reciprocal rescaling of the two tensors preserves the bond contraction.
            gauge=sqrt(norm(B)/norm(A)); A=A*gauge; B=B/gauge
            raw=_residual_cost(N,A,B,target)
            trigger=_repair_trigger(raw,target_norm)
            trigger===nothing || return (;trigger,stage=:als_iteration,iteration,raw,target_norm)
            evaluation=_cost_evaluation(raw,target_norm)
            residual=max(ra,rb); iterations=iteration
            if evaluation.fallback
                stop_reason=:residual_fallback
                push!(fallbacks,(;iteration,relative_cost=evaluation.relative_cost,used_initial_local_svd=false))
                break # retain the best fit with a nonnegative residual
            end
            if evaluation.cost<=best
                best=evaluation.cost; bestA=copy(A); bestB=copy(B)
            end
            if best<=options.error_tolerance
                stop_reason=:error_tolerance
                break
            end
            iteration_limit_reached=iteration==options.max_iterations
        end
    end
    return (;trigger=nothing,left=bestA,right=bestB,initial_cost=initial.cost,
        final_cost=best,residual,iterations,iteration_limit_reached,fallbacks,stop_reason)
end

function update_bond!(state,i,j,gate,D,method::FullUpdate,hole;direction,boundary_error=0.0,sector_space=nothing)
    pair=_reduced_pair(state[i],state[j],gate,direction,D;sector_space)
    N=_reduced_metric(hole,pair.Qa,pair.Qb)
    fit=_optimize_pair(pair,N,method); repairs=NamedTuple[]
    if fit.trigger!==nothing
        # PSD-project the full two-site environment, then apply QR reduction.
        matrix=permute(bond_hole(hole),((2,4,6,8,10,12),(1,3,5,7,9,11)))
        repaired,diagnostic=_positive_metric(matrix)
        repaired_hole=_perm(repaired,(7,1,8,2,9,3,10,4,11,5,12,6))
        push!(repairs,merge(diagnostic,(;reason=fit.trigger,stage=fit.stage,
            iteration=fit.iteration,raw_cost=fit.raw,target_norm=fit.target_norm)))
        fit=_optimize_pair(pair,_reduced_metric(repaired_hole,pair.Qa,pair.Qb),method)
        fit.trigger===nothing || error("FU residual evaluation failed after PSD environment repair")
    end
    state.sites[i],state.sites[j]=_restore_pair(pair,fit.left,fit.right)
    return (;bond=(Tuple(i),Tuple(j)),direction,beta=state.beta,
        initial_cost=fit.initial_cost,final_cost=fit.final_cost,residual=fit.residual,
        iterations=fit.iterations,iteration_limit_reached=fit.iteration_limit_reached,
        stop_reason=fit.stop_reason,
        max_environment_truncation=boundary_error,fallbacks=fit.fallbacks,repairs)
end

function _sweep!(state,model,delta,D,method,contraction::AbstractRowBoundaryMPS,direction,reverse;
        sector_space=nothing,entropy=nothing,variational=nothing)
    l=state.lattice; layers=_row_layers(state,contraction)
    env=boundary_environments(layers,contraction;side=reverse ? :north : :south,entropy)
    environment_error=env.max_environment_truncation
    if variational!==nothing
        append!(variational,_row_variational_records(reverse ? env.north[end] : env.south[1]))
    end
    ymax=direction==:horizontal ? l.Ly : l.Ly-1
    xmax=direction==:horizontal ? l.Lx-1 : l.Lx
    records=NamedTuple[]
    for y in (reverse ? (ymax:-1:1) : (1:ymax))
        top=env.north[y].tensors; lastrow=y+(direction==:horizontal ? 0 : 1)
        bottom=env.south[lastrow+1].tensors
        row=direction==:horizontal ? collect(layers[:,y]) :
            [SlabColumn(layers[x,y],layers[x,y+1]) for x in 1:l.Lx]
        edges=row_environments(top,row,bottom;side=reverse ? :left : :right)
        rolling=reverse ? edges.right[end] : edges.left[1]
        for x in (reverse ? (xmax:-1:1) : (1:xmax))
            i=CartesianIndex(x,y); j=direction==:horizontal ? CartesianIndex(x+1,y) : CartesianIndex(x,y+1)
            finalx=direction==:horizontal ? x+1 : x
            left=reverse ? edges.left[x] : rolling
            right=reverse ? rolling : edges.right[finalx+1]
            H=bond_environment(state,layers,env,i,j,direction;left,right)
            gate=trotter_gate(model,l,i,j,delta;T=scalartype(state[i]))
            error=max(env.north[y].max_truncation,env.south[lastrow+1].max_truncation)
            push!(records,update_bond!(state,i,j,gate,D,method,H;direction,boundary_error=error,sector_space))
            layers[i]=_row_site(contraction,state[i]);layers[j]=_row_site(contraction,state[j])
            column=reverse ? finalx : x
            if x != (reverse ? 1 : xmax)
                E=direction==:horizontal ? layers[column,y] : SlabColumn(layers[column,y],layers[column,y+1])
                rolling=_safe_scale(reverse ? _right_step(top[column],E,bottom[column],rolling) :
                    _left_step(rolling,top[column],E,bottom[column]))
            end
        end
        if reverse
            env.south[lastrow]=_absorb(env.south[lastrow+1],collect(layers[:,lastrow]),contraction;
                side=:south,entropy,row_cut=lastrow-1)
            environment_error=max(environment_error,env.south[lastrow].max_truncation)
            variational===nothing || push!(variational,last(_row_variational_records(env.south[lastrow])))
        else
            env.north[y+1]=_absorb(env.north[y],collect(layers[:,y]),contraction;entropy,row_cut=y)
            environment_error=max(environment_error,env.north[y+1].max_truncation)
            variational===nothing || push!(variational,last(_row_variational_records(env.north[y+1])))
        end
    end
    return records,environment_error
end

function evolution_step!(state,model,delta,D,method::FullUpdate,contraction::AbstractBoundaryMPS;
        order=2,sector_space=nothing,record_entropy=false)
    records=NamedTuple[]; environment_error=0.0
    variational=_is_variational(typeof(contraction)) ? NamedTuple[] : nothing
    entropy=record_entropy ? BoundaryEntropy() : nothing
    gate_delta=order==2 ? delta/2 : delta
    sweeps=order==2 ? ((:horizontal,false),(:vertical,false),(:vertical,true),(:horizontal,true)) :
        ((:horizontal,false),(:vertical,false))
    if length(state.lattice)==1
        onsite=onsite_hamiltonian(model;T=scalartype(state[1,1]))
        state.sites[1,1]=_act(state[1,1],exp(-delta*onsite),3)
    else
        for (direction,reverse) in sweeps
            isempty(bonds(state.lattice,direction)) && continue
            extra=variational===nothing ? (;) : (;variational)
            r,e=_sweep!(state,model,gate_delta,D,method,contraction,direction,reverse;sector_space,entropy,extra...)
            append!(records,r);environment_error=max(environment_error,e)
        end
    end
    state.beta+=2delta
    normalizations=normalize_state!(state)
    return merge((;beta=state.beta,delta_tau=delta,normalizations,
        bond_updates=length(records),
        iteration_limit_hits=count(r->r.iteration_limit_reached,records),
        max_iteration_limit_error=maximum((r.final_cost for r in records if r.iteration_limit_reached);init=0.0),
        stop_reasons=(;error_tolerance=count(r->r.stop_reason==:error_tolerance,records),
            iteration_limit=count(r->r.stop_reason==:iteration_limit,records),
            residual_fallback=count(r->r.stop_reason==:residual_fallback,records),
            disabled=count(r->r.stop_reason==:disabled,records)),
        linear_unconverged_updates=count(r->r.residual>method.linear_tolerance,records),
        max_state_truncation=maximum((r.final_cost for r in records);init=0.0),
        max_environment_truncation=environment_error,
        max_boundary_entropy=entropy===nothing ? NaN : entropy.maximum,
        max_solver_residual=maximum((r.residual for r in records);init=0.0),
        fallbacks=[merge(f,(;bond=r.bond,direction=r.direction,beta=state.beta)) for r in records for f in r.fallbacks],
        repairs=[merge(f,(;bond=r.bond,direction=r.direction,beta=state.beta)) for r in records for f in r.repairs]),
        _variational_result(contraction,variational))
end
