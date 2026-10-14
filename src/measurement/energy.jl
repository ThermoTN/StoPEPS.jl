function _real_ratio(a,b)
    iszero(b) && error("zero local norm environment")
    z=a/b
    isfinite(z) && abs(imag(z))<=1e-9*max(1.0,abs(z)) || error("nonreal observable: $z")
    Float64(real(z))
end

function _operator_factors(op)
    M=permute(op,((1,3),(2,4)))
    U,S,V,_=svd_trunc(M;trunc=trunctol(atol=64eps(Float64)*norm(M)))
    (_out(U*S),_out(_twistdual(V,(1,))))
end
function _operator_pair(A,B,factors,direction)
    F,G=factors
    if direction==:horizontal
        @tensor C[w,n,p,a,e,r,s] := A[w,n,u,a,e,s]*F[p,u,r]
        @tensor D[w,r,n,q,a,e,s] := B[w,n,v,a,e,s]*G[r,q,v]
        return _fuse(C,5,2),_fuse(D,1,2)
    end
    @tensor C[w,n,p,a,e,s,r] := A[w,n,u,a,e,s]*F[p,u,r]
    @tensor D[w,n,r,q,a,e,s] := B[w,n,v,a,e,s]*G[r,q,v]
    return _fuse(C,6,2),_fuse(D,2,2)
end

function _energy_terms(model::TFIM,::Type{Scalar}) where Scalar
    operators=spin_operators(model,Scalar)
    (;onsite=operators.X,pair=operators.ZZ,onsite_scale=-model.h,pair_scale=-model.J)
end
function _energy_terms(model::Hubbard,::Type{Scalar}) where Scalar
    operators=hubbard_operators(model,Scalar)
    (;onsite=operators.D,pair=operators.hopping,onsite_scale=model.U,pair_scale=-model.t)
end

"""Cached one-site and nearest-neighbor measurement, shared by PEPO and PEPS."""
function measure_energy(state::FiniteState,model::AbstractModel,method::AbstractRowBoundaryMPS;record_entropy=false)
    layers=_row_layers(state,method)
    entropy=record_entropy ? BoundaryEntropy(size(layers)...) : nothing
    env=boundary_environments(layers,method;entropy)
    terms=_energy_terms(model,scalartype(state[1,1]))
    X=map(A->_row_site(method,A;physical=terms.onsite),state.sites)
    factors=_operator_factors(terms.pair)
    Lx,Ly=size(layers); xsum=0.0; zzsum=0.0
    for y in 1:Ly
        row=collect(layers[:,y]); top=env.north[y].tensors; bottom=env.south[y+1].tensors
        e=row_environments(top,row,bottom)
        for x in 1:Lx
            den=_row_scalar(e.left[x],top[x],row[x],bottom[x],e.right[x+1])
            num=_row_scalar(e.left[x],top[x],X[x,y],bottom[x],e.right[x+1])
            xsum+=_real_ratio(num,den)
        end
        for x in 1:Lx-1
            A,B=state[x,y],state[x+1,y]
            C,D=_operator_pair(A,B,factors,:horizontal)
            left=_left_step(e.left[x],top[x],_row_site(method,C,A),bottom[x])
            num=_row_scalar(left,top[x+1],_row_site(method,D,B),bottom[x+1],e.right[x+2])
            left=_left_step(e.left[x],top[x],row[x],bottom[x])
            den=_row_scalar(left,top[x+1],row[x+1],bottom[x+1],e.right[x+2])
            zzsum+=_real_ratio(num,den)
        end
    end
    for y in 1:Ly-1
        row=[SlabColumn(layers[x,y],layers[x,y+1]) for x in 1:Lx]
        top=env.north[y].tensors; bottom=env.south[y+2].tensors
        e=row_environments(top,row,bottom)
        for x in 1:Lx
            A,B=state[x,y],state[x,y+1]
            C,D=_operator_pair(A,B,factors,:vertical)
            impurity=SlabColumn(_row_site(method,C,A),_row_site(method,D,B))
            den=_row_scalar(e.left[x],top[x],row[x],bottom[x],e.right[x+1])
            num=_row_scalar(e.left[x],top[x],impurity,bottom[x],e.right[x+1])
            zzsum+=_real_ratio(num,den)
        end
    end
    final=env.north[end]
    log_norm=log(_positive_real(_close_boundary(final.tensors),"norm"))+final.log_scale
    energy_total=terms.pair_scale*zzsum+terms.onsite_scale*xsum
    return merge((;energy_total,energy_per_site=energy_total/length(state.lattice),
        log_norm=log_norm+2state.log_amplitude,tensor_log_norm=log_norm,
        max_environment_truncation=env.max_environment_truncation,
        Se=entropy===nothing ? nothing : entropy.Se),
        _variational_result(method,vcat(_row_variational_records(final),_row_variational_records(env.south[1]))))
end
