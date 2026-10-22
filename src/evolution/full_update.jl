abstract type UpdateMethod end
"""
    FullUpdate(; max_iterations=4, error_tolerance=1e-8,
        diagonal_shift=1e-10, linear_max_iterations=24, linear_tolerance=1e-9)

Environment-gauged two-site full update with preconditioned conjugate-gradient
solves, internal QR/LQ center motion and final bond-center SVD balancing.
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
const _FU_ALGORITHM_REVISION = 2

function _reduced_pair(A,B,gate,direction)
    pa=direction==:horizontal ? (1,2,6,4,3,5) : (1,2,5,4,3,6)
    pb=direction==:horizontal ? (2,5,6,4,3,1) : (1,5,6,4,3,2)
    Qa,Ca=left_orth(permute(A,(pa[1:4],pa[5:6])))
    Qb,Cb=left_orth(permute(B,(pb[1:4],pb[5:6])))
    a=_out(Ca); b=_perm(Cb,(3,2,1))
    @tensor target[r,p,q,s] := a[r,u,k]*b[k,v,s]*gate[p,q,u,v]
    return (;Qa=_out(Qa),Qb=_out(Qb),target,pa,pb)
end

function _fu_svd_initial(target,D,sector_space;context=nothing)
    M=permute(target,((1,2),(3,4)))
    cutoff=eps(real(float(one(scalartype(M)))))*maximum(dim(space(target,k)) for k in 1:4)*norm(M)
    U,S,V,_=_svd_trunc(M;trunc=_truncation(D,sector_space)&trunctol(atol=cutoff),
        context=(;operation=:full_update_initial_svd,location=context))
    dim(domain(U))>0 || error("gate produced a zero target")
    root=copy(S)
    for (_,block) in blocks(root), k in axes(block,1)
        block[k,k]=sqrt(real(block[k,k]))
    end
    return _out(U*root),_out(root*V)
end

"""Balance the retained pair QA*C*QB using only its untruncated bond center."""
function _fu_balance_bond(QA,C,QB;context=nothing)
    U,S,V=_svd_compact(C;context=(;operation=:full_update_bond_balance,location=context))
    root=copy(S)
    for (_,block) in blocks(root), k in axes(block,1)
        block[k,k]=sqrt(real(block[k,k]))
    end
    return _out(QA*(U*root)),_out((root*V)*QB)
end

"""Apply the inverse gauges one leg at a time, without forming their tensor product."""
function _fu_transform_metric(N,left_inverse,right_inverse)
    # Keep each intermediate in the matrix partition used by its contraction.
    @tensor first[j,k,l;x] := N[i,j,k,l]*conj(left_inverse[i,x])
    @tensor second[k,l,x;y] := first[j,k,l,x]*conj(right_inverse[j,y])
    @tensor third[l,x,y;z] := second[k,l,x,y]*left_inverse[k,z]
    @tensor metric[x,y;z,w] := third[l,x,y,z]*right_inverse[l,w]
    metric
end

"""Environment gauge from independent QR factorizations of a positive square root.

The spectral floor changes only the auxiliary matrix used to choose coordinates.
The ALS metric is the congruence of the original N, including its full spectrum.
Both QR factorizations retain the complete exterior spaces. On all-outgoing
cores the right gauge acts on the last index; its matrix transpose is the R in
the usual two-site notation L*T*R.
"""
function _fu_environment_gauge(N)
    _finite(N) || error("nonfinite FU norm metric")
    scale=norm(N)
    scale>0 || error("zero FU norm metric")
    normalized=N/scale
    H=(normalized+normalized')/2
    values,vectors=eigh_full(H)
    spectral_scale=maximum((maximum(abs,b;init=0.0) for (_,b) in blocks(values));init=0.0)
    precision=eps(real(float(one(scalartype(N)))))
    relative_floor=max(sqrt(precision),dim(domain(N))*precision)
    # A purely anti-Hermitian N still gets invertible coordinates; the original
    # target/residual checks below decide whether full-hole repair is necessary.
    spectral_scale=iszero(spectral_scale) ? one(spectral_scale) : spectral_scale
    floor=relative_floor*spectral_scale
    root=copy(values); floored_multiplets=0; floored_eigenvalues=0
    minimum_value=Inf
    for (c,b) in blocks(root), k in axes(b,1)
        value=real(b[k,k]); minimum_value=min(minimum_value,value)
        if value<floor
            floored_multiplets+=1; floored_eigenvalues+=dim(c)
        end
        b[k,k]=sqrt(max(value,floor))
    end
    # Z'Z is the positive auxiliary metric. Treat the two external legs
    # independently, so neither gauge preferentially conditions the other.
    Z=root*vectors'
    _,left=left_orth(permute(Z,((1,3),(2,))))
    _,right=left_orth(permute(Z,((1,2),(3,))))
    left_inverse=inv(left);right_inverse=inv(right)
    metric=_fu_transform_metric(N,left_inverse,right_inverse)
    diagnostic=(;relative_floor=Float64(relative_floor),floored_multiplets,floored_eigenvalues,
        minimum_relative_eigenvalue=Float64(minimum_value/spectral_scale),
        hermiticity_defect=Float64(norm(normalized-normalized')))
    return (;left,right,left_inverse,right_inverse,metric,diagnostic)
end

function _initialize_pair(pair,N,D;sector_space=nothing,context=nothing)
    gauge=_fu_environment_gauge(N)
    working_target=_act(_act(pair.target,gauge.left,1),gauge.right,4)
    left,right=_fu_svd_initial(working_target,D,sector_space;context)
    merge(pair,(;left,right,working_target,working_metric=gauge.metric,
        left_inverse=gauge.left_inverse,right_inverse=gauge.right_inverse,
        gauge_diagnostic=gauge.diagnostic))
end

_fu_original_cores(pair,A,B)=(_act(A,pair.left_inverse,1),_act(B,pair.right_inverse,3))

function _fu_gauge_summary(records)
    (;constructions=length(records),
        floored_multiplets=sum((r.floored_multiplets for r in records);init=0),
        floored_eigenvalues=sum((r.floored_eigenvalues for r in records);init=0),
        max_relative_floor=maximum((r.relative_floor for r in records);init=0.0),
        minimum_relative_eigenvalue=isempty(records) ? 0.0 : minimum(r.minimum_relative_eigenvalue for r in records),
        max_hermiticity_defect=maximum((r.hermiticity_defect for r in records);init=0.0))
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

# Keep the two-site repair in matrix partitions until its exterior legs have
# been grouped. Joining all twelve outgoing legs would create a large tensor
# and fusion-tree layout before the PSD projection even starts.
function _fu_repair_join(a,b)
    common=intersect(a.labels,b.labels)
    ia=Tuple(findfirst(==(k),a.labels) for k in common)
    ib=Tuple(findfirst(==(k),b.labels) for k in common)
    oa=Tuple(setdiff(eachindex(a.labels),ia))
    ob=Tuple(setdiff(eachindex(b.labels),ib))
    data=permute(a.data,(oa,ia))*permute(b.data,(ib,ob))
    (;data,labels=vcat(a.labels[collect(oa)],b.labels[collect(ob)]))
end

# Canonical fusion preserves the ordering of TensorKit's reduced matrix blocks.
# Reinterpret that storage without building a dense fusion isomorphism.
_fu_matrix_view(t)=TensorMap(t.data,ProductSpace(fuse(codomain(t)))←ProductSpace(fuse(domain(t))))
_fu_sector_slice(V::TensorKit.GradedSpace,c,n)=
    typeof(V)((isdual(V) ? dual(c) : c)=>n;dual=isdual(V))
_fu_sector_slice(V::ComplexSpace,c,n)=ComplexSpace(n;dual=isdual(V))

# A full hole in two grouped exterior spaces. Never form its twelve single
# legs: even the fusion-tree metadata of that layout can exceed node memory.
# Each three-leg group uses the same canonical matrix basis on H and its Q.
function _group_hole_half(tensors,labels,external,spaces)
    X=_DiagonalTensor(first(tensors),first(labels))
    for n in 1:3
        n==1 || (X=_fu_repair_join(X,_DiagonalTensor(tensors[n],labels[n])))
        axis=findfirst(==(external[n]),X.labels)
        data=_split(X.data,axis,(dual(spaces[n]),spaces[n]))
        names=copy(X.labels);splice!(names,axis:axis,[-201,-202])
        X=_DiagonalTensor(data,names)
        if n>1
            for (old,new,ket) in ((-203,-201,true),(-204,-202,false))
                selected=Tuple(findfirst(==(k),X.labels) for k in (old,new))
                other=Tuple(setdiff(eachindex(X.labels),selected))
                p=ket ? (other,selected) : (selected,other)
                M=permute(X.data,p)
                layout=ket ? codomain(M)←ProductSpace(fuse(domain(M))) :
                    ProductSpace(fuse(codomain(M)))←domain(M)
                names=ket ? vcat(X.labels[collect(other)],new) : vcat(new,X.labels[collect(other)])
                X=(;data=TensorMap(M.data,layout),labels=names)
            end
        end
        X=(;data=X.data,labels=[k==-201 ? -203 : k==-202 ? -204 : k for k in X.labels])
    end
    X
end

function _grouped_bond_hole(env::BondEnvironment)
    if env.direction==:horizontal
        T1,T2=env.top;B1,B2=env.bottom
        lt=(env.left,T1,B1);ll=([1,11,2],[1,12,3],[2,13,4])
        rt=(env.right,T2,B2);rl=([5,15,6],[3,14,5],[4,16,6])
        aorder=(1,2,3);border=(5,4,6)
        lext=(11,12,13);rext=(15,14,16)
    else
        # Exact QR splits only the two four-leg strip edges. The two halves
        # then each have three exterior legs and two connector bonds.
        L,R=left_orth(permute(env.left,((1,2),(3,4))))
        U,V=left_orth(permute(env.right,((1,2),(3,4))))
        lt=(_out(L),only(env.top),_out(U));ll=([1,11,3],[1,12,2],[2,13,4])
        rt=(_out(R),only(env.bottom),_out(V));rl=([3,14,5],[5,16,6],[4,15,6])
        aorder=(1,2,3);border=(4,6,5)
        lext=(11,12,13);rext=(14,16,15)
    end
    A=_group_hole_half(lt,ll,lext,map(i->env.outer_spaces[i],aorder))
    A=_DiagonalTensor(copy(A.data),[k==-203 ? -211 : k==-204 ? -212 : k for k in A.labels])
    B=_group_hole_half(rt,rl,rext,map(i->env.outer_spaces[i],border))
    B=_DiagonalTensor(B.data,[k==-203 ? -213 : k==-204 ? -214 : k for k in B.labels])
    H=_fu_repair_join(A,B)
    indices=map(k->findfirst(==(k),H.labels),(-212,-214,-211,-213))
    matrix=permute(H.data,(indices[1:2],indices[3:4]))
    # Retain only rank-two storage plus its four-leg logical matrix interface.
    (;matrix=_fu_matrix_view(matrix),layout=space(matrix),
        orders=(aorder,Tuple(i-3 for i in border)))
end

function _group_hole_q(Q,order,bra)
    t=bra ? _conj(Q) : Q
    t=_perm(t,(order...,4,5))
    for _ in 1:2
        others=Tuple(3:numind(t))
        M=permute(t,bra ? (others,(1,2)) : ((1,2),others))
        layout=bra ? codomain(M)←ProductSpace(fuse(domain(M))) :
            ProductSpace(fuse(codomain(M)))←domain(M)
        packed=TensorMap(M.data,layout)
        t=bra ? _perm(packed,(numind(packed),Tuple(1:numind(packed)-1)...)) : _out(packed)
    end
    t
end

function _grouped_repair_piece(W,Qs)
    # W'W is the COMPLETE hole's PSD projection in the grouped matrix basis.
    bra=_DiagonalTensor(_out(W'),[11,12,1])
    ket=_DiagonalTensor(_out(W),[1,13,14])
    bra=_fu_repair_join(bra,_DiagonalTensor(Qs[3],[11,2,-1]))
    bra=_fu_repair_join(bra,_DiagonalTensor(Qs[4],[12,3,-2]))
    ket=_fu_repair_join(ket,_DiagonalTensor(Qs[1],[13,2,-3]))
    ket=_fu_repair_join(ket,_DiagonalTensor(Qs[2],[14,3,-4]))
    N=_fu_repair_join(bra,ket)
    p=map(k->findfirst(==(k),N.labels),(-1,-2,-3,-4))
    permute(N.data,(Tuple(p[1:2]),Tuple(p[3:4])))
end

function _positive_reduced_metric(env::BondEnvironment,Qa,Qb)
    grouped=_grouped_bond_hole(env)
    H=grouped.matrix
    _finite(H) || error("nonfinite norm metric")
    scale=norm(H)
    Qs=(_group_hole_q(Qa,grouped.orders[1],false),
        _group_hole_q(Qb,grouped.orders[2],false),
        _group_hole_q(Qa,grouped.orders[1],true),
        _group_hole_q(Qb,grouped.orders[2],true))
    minimum_value=0.0;spectral_scale=0.0;negative_weight=0.0
    negative_multiplets=0;negative_squared=0.0;spectral_squared=0.0;defect_squared=0.0
    core=dual(space(Qa,5))⊗dual(space(Qb,5))
    result=zeros(promote_type(eltype(H),eltype(Qa),eltype(Qb)),core←core)
    for (c,blockH) in blocks(H)
        # One owned dense sector workspace; the complete H+, V and diagonal
        # eigenvalue TensorMaps are never constructed.
        M=Matrix(blockH)
        iszero(scale) || (M ./= scale)
        weight=Float64(dim(c))
        defect_squared+=weight*sum(abs2(M[i,j]-conj(M[j,i])) for i in axes(M,1),j in axes(M,2))
        for j in axes(M,2),i in 1:j
            v=(M[i,j]+conj(M[j,i]))/2
            M[i,j]=v;M[j,i]=conj(v)
        end
        F=eigen!(Hermitian(M))
        values=F.values
        minimum_value=min(minimum_value,minimum(values;init=0.0))
        spectral_scale=max(spectral_scale,maximum(abs,values;init=0.0))
        negative_multiplets+=count(<(0),values)
        negative_weight+=weight*count(<(0),values)
        negative_squared+=weight*sum(abs2,min.(values,0))
        spectral_squared+=weight*sum(abs2,values)
        # Eigenvector rows are batched as complete irreps. This also bounds
        # Q-projection intermediates without changing any spectral cutoff.
        positive=findall(>(0),values)
        for first in 1:32:length(positive)
            selected=view(positive,first:min(first+31,length(positive)))
            V=_fu_sector_slice(codomain(H)[1],c,length(selected))
            layout=ProductSpace(V)←domain(grouped.layout)
            W=zeros(eltype(H),V←domain(H)[1])
            destination=block(W,c)
            for (i,k) in enumerate(selected),j in axes(destination,2)
                destination[i,j]=sqrt(max(values[k],0))*conj(F.vectors[j,k])
            end
            piece=_grouped_repair_piece(TensorMap(W.data,layout),Qs)
            axpy!(one(eltype(piece)),piece,result)
        end
    end
    report=(;minimum_relative_eigenvalue=minimum_value/max(spectral_scale,floatmin(Float64)),
        negative_eigenvalues=negative_weight,negative_multiplets,
        negative_spectral_fraction=sqrt(negative_squared)/max(sqrt(spectral_squared),floatmin(Float64)),
        hermiticity_defect=sqrt(defect_squared)/(iszero(scale) ? floatmin(Float64) : 1.0))
    result,report
end
_positive_reduced_metric(env::FactorizedRowBondEnvironment,Qa,Qb)=
    _positive_reduced_metric(env.environment,Qa,Qb)

function _positive_reduced_metric(hole,Qa,Qb)
    matrix=permute(bond_hole(hole),((2,4,6,8,10,12),(1,3,5,7,9,11)))
    repaired,diagnostic=_positive_metric(matrix)
    repaired_hole=_perm(repaired,(7,1,8,2,9,3,10,4,11,5,12,6))
    _reduced_metric(repaired_hole,Qa,Qb),diagnostic
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

function _left_normal_equation(metric,right,weighted_target)
    @tensor normal[first,bond;second,other] := metric[first,south,second,north]*
        conj(right[bond,physical,south])*right[other,physical,north]
    @tensor rhs[first,physical,bond] := weighted_target[first,physical,partner,south]*
        conj(right[bond,partner,south])
    normal,permute(rhs,((1,3),(2,)))
end

function _right_normal_equation(metric,left,weighted_target)
    @tensor normal[bond,south;other,north] := metric[west,south,east,north]*
        conj(left[west,physical,bond])*left[east,physical,other]
    @tensor rhs[bond,physical,south] := weighted_target[west,partner,physical,south]*
        conj(left[west,partner,bond])
    normal,permute(rhs,((1,3),(2,)))
end

function _optimize_pair(pair,N,options;context=nothing)
    A=copy(pair.left); B=copy(pair.right)
    target=pair.working_target; metric=pair.working_metric
    target_norm=dot(pair.target,_apply_pair_metric(N,pair.target))
    originalA,originalB=_fu_original_cores(pair,A,B)
    raw=_residual_cost(N,originalA,originalB,pair.target)
    trigger=_repair_trigger(raw,target_norm)
    trigger===nothing || return (;trigger,stage=:initial_local_svd,iteration=0,raw,target_norm)
    initial=_cost_evaluation(raw,target_norm)
    bestA,bestB=originalA,originalB; best=initial.cost; residual=0.0
    best_factors=nothing
    fallbacks=NamedTuple[]; iterations=0; iteration_limit_reached=false
    stop_reason=options.max_iterations==0 ? :disabled : :iteration_limit
    if initial.fallback
        stop_reason=:residual_fallback
        push!(fallbacks,(;iteration=0,relative_cost=initial.relative_cost,used_initial_local_svd=true))
    elseif options.max_iterations>0 && best<=options.error_tolerance
        # The initial gated SVD already meets the same normalized squared
        # residual threshold used after each ALS iteration.
        stop_reason=:error_tolerance
    elseif options.max_iterations>0
        # The target and environment stay fixed throughout this ALS solve.
        weighted_target=_apply_pair_metric(metric,target)
        for iteration in 1:options.max_iterations
            MA,ya=_left_normal_equation(metric,B,weighted_target)
            sol,ra=_solve_core(MA,ya,permute(A,((1,3),(2,))),options)
            # Move the center to the right without inverting or truncating R.
            QA,R=left_orth(permute(sol,((1,3),(2,))))
            right_initial=R*permute(B,((1,),(2,3)))
            A=_out(QA)
            MB,yb=_right_normal_equation(metric,A,weighted_target)
            sol,rb=_solve_core(MB,yb,permute(right_initial,((1,3),(2,))),options)
            C,QB=right_orth(permute(sol,((1,),(3,2))))
            A=_out(QA*C); B=_out(QB)
            originalA,originalB=_fu_original_cores(pair,A,B)
            raw=_residual_cost(N,originalA,originalB,pair.target)
            trigger=_repair_trigger(raw,target_norm)
            trigger===nothing || return (;trigger,stage=:als_iteration,iteration,raw,target_norm)
            evaluation=_cost_evaluation(raw,target_norm)
            residual=max(ra,rb); iterations=iteration
            if evaluation.fallback
                stop_reason=:residual_fallback
                push!(fallbacks,(;iteration,relative_cost=evaluation.relative_cost,used_initial_local_svd=false))
                break # retain the last nonnegative best, never the old ungated state
            end
            if evaluation.cost<=best
                best=evaluation.cost; bestA=originalA; bestB=originalB
                # Later solves allocate new factors, so the best factors can
                # be retained directly and reused for the final small SVD.
                best_factors=(QA,C,QB)
            end
            if best<=options.error_tolerance
                stop_reason=:error_tolerance
                break
            end
            iteration_limit_reached=iteration==options.max_iterations
        end
    end
    if best_factors!==nothing
        A,B=_fu_balance_bond(best_factors...;context)
        bestA,bestB=_fu_original_cores(pair,A,B)
    end
    return (;trigger=nothing,left=bestA,right=bestB,initial_cost=initial.cost,
        final_cost=best,residual,iterations,iteration_limit_reached,fallbacks,stop_reason)
end

function update_bond!(state,i,j,gate,D,method::FullUpdate,hole;direction,boundary_error=0.0,sector_space=nothing)
    pair=_reduced_pair(state[i],state[j],gate,direction)
    N=_reduced_metric(hole,pair.Qa,pair.Qb)
    context=(;bond=(Tuple(i),Tuple(j)),direction,beta=state.beta)
    initial=_initialize_pair(pair,N,D;sector_space,context)
    gauges=[initial.gauge_diagnostic]
    fit=_optimize_pair(initial,N,method;context); repairs=NamedTuple[]
    if fit.trigger!==nothing
        # Project the complete hole, not N: PSD projection and QR reduction do
        # not commute. This is the only production FU path materializing H.
        repaired,diagnostic=_positive_reduced_metric(hole,pair.Qa,pair.Qb)
        push!(repairs,merge(diagnostic,(;reason=fit.trigger,stage=fit.stage,
            iteration=fit.iteration,raw_cost=fit.raw,target_norm=fit.target_norm)))
        initial=_initialize_pair(pair,repaired,D;sector_space,context)
        push!(gauges,initial.gauge_diagnostic)
        fit=_optimize_pair(initial,repaired,method;context)
        fit.trigger===nothing || error("PSD repair did not remove the FU residual failure")
    end
    state.sites[i],state.sites[j]=_restore_pair(pair,fit.left,fit.right)
    return (;bond=(Tuple(i),Tuple(j)),direction,beta=state.beta,
        initial_cost=fit.initial_cost,final_cost=fit.final_cost,residual=fit.residual,
        iterations=fit.iterations,iteration_limit_reached=fit.iteration_limit_reached,
        stop_reason=fit.stop_reason,
        max_environment_truncation=boundary_error,fallbacks=fit.fallbacks,repairs,gauges)
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
        fu_gauge=_fu_gauge_summary([g for r in records for g in r.gauges]),
        fallbacks=[merge(f,(;bond=r.bond,direction=r.direction,beta=state.beta)) for r in records for f in r.fallbacks],
        repairs=[merge(f,(;bond=r.bond,direction=r.direction,beta=state.beta)) for r in records for f in r.repairs]),
        _variational_result(contraction,variational))
end
