abstract type ContractionMethod end
abstract type AbstractBoundaryMPS <: ContractionMethod end
abstract type AbstractRowBoundaryMPS <: AbstractBoundaryMPS end
"""
    BoundaryMPS(chi, sector_space=nothing)

Row-by-row boundary-MPS contraction with canonical QR/SVD compression.
`chi` sets the retained bond dimension; `sector_space` specifies a symmetry
space budget. `BoundaryMPSRow` names the same method.
"""
struct BoundaryMPS{S} <: AbstractRowBoundaryMPS
    chi::Int
    sector_space::S
end
BoundaryMPS(chi::Int)=BoundaryMPS(chi,nothing)
BoundaryMPS(;chi=32,sector_space=nothing)=BoundaryMPS(chi,sector_space)
const BoundaryMPSRow = BoundaryMPS
_truncation(D,sector_space=nothing)=sector_space===nothing ? truncrank(D) : truncrank(D)&truncspace(sector_space)

"""Pretruncation Schmidt entropy in nats, including sector dimensions."""
function _schmidt_entropy(S)
    scale=norm(S)
    iszero(scale) && return NaN # Entropy is undefined for a zero-norm spectrum.
    entropy=0.0
    for (sector,block) in blocks(S), i in axes(block,1)
        p=abs2(block[i,i]/scale)
        iszero(p) || (entropy-=dim(sector)*p*log(p))
    end
    max(0.0,entropy)
end

"""Reuse the same compact SVD, truncation selection and error as svd_trunc."""
function _boundary_svd(M,trunc,observer,bond)
    observer===nothing && return svd_trunc(M;trunc)
    alg=MAK.select_algorithm(MAK.svd_trunc!,M,nothing;trunc)
    U,S,V=svd_compact(M,alg.alg)
    # truncation_error! overwrites the kept entries of S: observe it first.
    observer(bond,_schmidt_entropy(S))
    kept,indices=MAK.truncate(MAK.svd_trunc!,(U,S,V),alg.trunc)
    err=MAK.truncation_error!(TensorKit.diagview(S),indices)
    return kept...,err
end

mutable struct BoundaryEntropy{P}
    maximum::Float64
    Se::P
end
BoundaryEntropy()=BoundaryEntropy(0.0,nothing)
BoundaryEntropy(Lx,Ly)=BoundaryEntropy(0.0,
    (;north=fill(NaN,Ly-1,Lx-1),south=fill(NaN,Ly-1,Lx-1)))
function _record_entropy!(diagnostic,side,row,bond,value)
    diagnostic.maximum=max(diagnostic.maximum,value)
    if diagnostic.Se!==nothing && 1<=row<=size(diagnostic.Se.north,1)
        getproperty(diagnostic.Se,side)[row,bond]=value
    end
    nothing
end

function _act(A, O, axis)
    labels=collect(-1:-1:-numind(A)); labels[axis]=1
    return ncon([A,O],[labels,[-axis,1]];output=collect(-1:-1:-numind(A)))
end

"""Double-layer fused legs (W,N,E,S), with each pair ordered (ket,bra)."""
function double_layer(ket, bra=ket; physical=nothing, auxiliary=nothing)
    A=physical === nothing ? ket : _act(ket,physical,3)
    A=auxiliary === nothing ? A : _act(A,auxiliary,4)
    @tensor raw[w,wb,n,nb,e,eb,s,sb] := A[w,n,p,a,e,s]*conj(bra[wb,nb,p,a,eb,sb])
    for axis in (7,5,3,1)
        raw=_fuse(raw,axis,2)
    end
    raw
end
double_layers(s::FiniteState) = map(double_layer,s.sites)
_row_site(::BoundaryMPS,A,B=A;kwargs...)=double_layer(A,B;kwargs...)
_row_layers(s,method::AbstractRowBoundaryMPS)=map(A->_row_site(method,A),s.sites)
_row_working_copy(row::AbstractVector{<:AbstractTensorMap})=AbstractTensorMap[row...]

struct BoundaryEnvironment
    tensors::Vector{AbstractTensorMap}
    log_scale::Float64
    max_truncation::Float64
end
_row_variational_records(::BoundaryEnvironment) = NamedTuple[]

function _trivial_boundary(row, side)
    axis=side==:north ? 2 : 4
    V=oneunit(space(first(row),axis))
    BoundaryEnvironment(AbstractTensorMap[_unit_tensor((V',dual(space(A,axis)),V);
        T=scalartype(A)) for A in row],0.0,0.0)
end

"""Opposite-direction QR first, followed by SVD toward the new center."""
function compress_boundary!(B, chi; reverse=false,sector_space=nothing,on_truncate=nothing)
    chi>=1 || throw(ArgumentError("chi must be positive"))
    N=length(B); max_error=0.0
    if !reverse
        for x in N:-1:2
            L,Q=right_orth(permute(B[x],((1,),(2,3))))
            B[x]=_out(Q)
            A=B[x-1]
            @tensor C[l,p,r] := A[l,p,k]*L[k,r]
            B[x-1]=C
        end
        for x in 1:N-1
            M=permute(B[x],((1,2),(3,)))
            U,S,V,err=_boundary_svd(M,_truncation(chi,sector_space),on_truncate,x)
            max_error=max(max_error,iszero(norm(M)) ? 0.0 : (err/norm(M))^2)
            B[x]=_out(U); R=S*V; A=B[x+1]
            @tensor C[l,p,r] := R[l,k]*A[k,p,r]
            B[x+1]=C
        end
    else
        for x in 1:N-1
            Q,R=left_orth(permute(B[x],((1,2),(3,))))
            B[x]=_out(Q); A=B[x+1]
            @tensor C[l,p,r] := R[l,k]*A[k,p,r]
            B[x+1]=C
        end
        for x in N:-1:2
            M=permute(B[x],((1,),(2,3)))
            U,S,V,err=_boundary_svd(M,_truncation(chi,sector_space),on_truncate,x-1)
            max_error=max(max_error,iszero(norm(M)) ? 0.0 : (err/norm(M))^2)
            B[x]=_out(V); L=U*S; A=B[x-1]
            @tensor C[l,p,r] := A[l,p,k]*L[k,r]
            B[x-1]=C
        end
    end
    center=reverse ? 1 : N
    scale=norm(B[center])
    isfinite(scale) || error("nonfinite boundary norm")
    iszero(scale) || (B[center]=B[center]/scale)
    return (;max_truncation=max_error,log_scale=iszero(scale) ? -Inf : log(scale))
end

function _absorb(env::BoundaryEnvironment,row,method::BoundaryMPS;side=:north,entropy=nothing,row_cut=0)
    B=AbstractTensorMap[]
    for (A,E) in zip(env.tensors,row)
        if side==:north
            @tensor C[l,w,p,r,e] := A[l,n,r]*E[w,n,e,p]
        else
            @tensor C[l,w,p,r,e] := A[l,s,r]*E[w,p,e,s]
        end
        push!(B,_fuse(_fuse(C,4,2),1,2))
    end
    observer=entropy===nothing ? nothing : (b,Se)->_record_entropy!(entropy,side,row_cut,b,Se)
    report=compress_boundary!(B,method.chi;reverse=side==:south,sector_space=method.sector_space,
        on_truncate=observer)
    return BoundaryEnvironment(B,env.log_scale+report.log_scale,
        max(env.max_truncation,report.max_truncation))
end

function _close_boundary(B)
    V=_unit_tensor((dual(space(first(B),1)),);T=scalartype(first(B)))
    for A in B
        P=_unit_tensor((dual(space(A,2)),);T=scalartype(A))
        @tensor W[r] := V[l]*A[l,p,r]*P[p]
        V=W
    end
    R=_unit_tensor((dual(space(V,1)),);T=scalartype(V))
    @tensor value = V[r]*R[r]
    value
end

function contract_layers(layers,method::BoundaryMPS)
    env=_trivial_boundary(collect(layers[:,1]),:north)
    for y in axes(layers,2)
        env=_absorb(env,collect(layers[:,y]),method)
    end
    return (;mantissa=_close_boundary(env.tensors),log_scale=env.log_scale,
        max_environment_truncation=env.max_truncation)
end
"""
    lognorm(state, contraction)

Compute the logarithm of the physical squared norm and boundary diagnostics.
`log_norm` includes the state's global logarithmic amplitude;
`tensor_log_norm` describes the stored tensor network.
"""
function lognorm(s::FiniteState,method::BoundaryMPS)
    c=contract_layers(double_layers(s),method)
    value=_positive_real(c.mantissa,"norm")
    return (;log_norm=log(value)+c.log_scale+2s.log_amplitude,
        tensor_log_norm=log(value)+c.log_scale,
        max_environment_truncation=c.max_environment_truncation)
end
function _positive_real(z,name)
    isfinite(z) && real(z)>0 && abs(imag(z)) <= 1e-9*abs(z) ||
        error("$name requires a finite positive real value; received $z")
    Float64(real(z))
end

function boundary_environments(layers,method::AbstractRowBoundaryMPS;side=:both,entropy=nothing)
    side in (:both,:north,:south) || throw(ArgumentError("invalid boundary side"))
    Ly=size(layers,2)
    firstboundary=_trivial_boundary(collect(layers[:,1]),:north)
    north=Vector{typeof(firstboundary)}(undef,Ly+1)
    south=similar(north)
    north[1]=firstboundary
    south[end]=_trivial_boundary(collect(layers[:,end]),:south)
    for y in (side==:south ? (1:0) : (1:Ly))
        north[y+1]=_absorb(north[y],collect(layers[:,y]),method;entropy,row_cut=y)
    end
    for y in (side==:north ? (0:-1:1) : (Ly:-1:1))
        south[y]=_absorb(south[y+1],collect(layers[:,y]),method;side=:south,entropy,row_cut=y-1)
    end
    (;north,south,max_environment_truncation=max(side==:south ? 0.0 : north[end].max_truncation,
        side==:north ? 0.0 : south[1].max_truncation))
end

"""A vertical slab column stored as separate upper and lower layers."""
struct SlabColumn{A,B}
    upper::A
    lower::B
end
function _left_step(L,T,E::SlabColumn,B)
    upper,lower=E.upper,E.lower
    @tensor X[w,v,b,n,c] := L[a,w,v,b]*T[a,n,c]
    @tensor Y[v,b,c,e,k] := X[w,v,b,n,c]*upper[w,n,e,k]
    @tensor Z[b,c,e,f,s] := Y[v,b,c,e,k]*lower[v,k,f,s]
    @tensor out[c,e,f,d] := Z[b,c,e,f,s]*B[b,s,d]
    out
end
function _right_step(T,E::SlabColumn,B,R)
    upper,lower=E.upper,E.lower
    @tensor X[a,n,e,f,d] := T[a,n,c]*R[c,e,f,d]
    @tensor Y[a,w,k,f,d] := X[a,n,e,f,d]*upper[w,n,e,k]
    @tensor Z[a,w,v,s,d] := Y[a,w,k,f,d]*lower[v,k,f,s]
    @tensor out[a,w,v,b] := Z[a,w,v,s,d]*B[b,s,d]
    out
end

function _left_step(L,T,E,B)
    # Keep the boundary in every intermediate, including for two-row slabs.
    @tensor X[w,b,n,c] := L[a,w,b]*T[a,n,c]
    @tensor Y[b,c,e,s] := X[w,b,n,c]*E[w,n,e,s]
    @tensor out[c,e,d] := Y[b,c,e,s]*B[b,s,d]
    out
end
function _right_step(T,E,B,R)
    # Contract with the boundary R before joining the slab layers.
    @tensor X[a,n,e,d] := T[a,n,c]*R[c,e,d]
    @tensor Y[a,w,s,d] := X[a,n,e,d]*E[w,n,e,s]
    @tensor out[a,w,b] := Y[a,w,s,d]*B[b,s,d]
    out
end
function _row_edge(top,row,bottom,side)
    i=side==:left ? 1 : length(row); a=side==:left ? 1 : 3
    if row[i] isa SlabColumn
        E=row[i]
        return _unit_tensor((dual(space(top[i],a)),dual(space(E.upper,a)),
            dual(space(E.lower,a)),dual(space(bottom[i],a)));T=scalartype(E.upper))
    end
    _unit_tensor((dual(space(top[i],a)),dual(space(row[i],a)),dual(space(bottom[i],a)));
        T=scalartype(row[i]))
end
function _safe_scale(t)
    s=norm(t); isfinite(s) || error("nonfinite row environment")
    iszero(s) ? t : t/s
end
function row_environments(top,row,bottom;side=:both)
    side in (:both,:left,:right) || throw(ArgumentError("invalid row side"))
    N=length(row); left=Vector{AbstractTensorMap}(undef,N+1); right=similar(left)
    left[1]=_row_edge(top,row,bottom,:left)
    right[end]=_row_edge(top,row,bottom,:right)
    for x in (side==:right ? (1:0) : (1:N))
        left[x+1]=_safe_scale(_left_step(left[x],top[x],row[x],bottom[x]))
    end
    for x in (side==:left ? (0:-1:1) : (N:-1:1))
        right[x]=_safe_scale(_right_step(top[x],row[x],bottom[x],right[x+1]))
    end
    (;left,right)
end
function _row_scalar(L,T,E,B,R)
    _join_scalars(_left_step(L,T,E,B),R)
end
function _join_scalars(L,R)
    if numind(L)==4
        @tensor value = L[a,b,c,d]*R[a,b,c,d]
    else
        @tensor value = L[a,b,c]*R[a,b,c]
    end
    value
end
function _stack_column(A,B)
    @tensor C[w,v,n,e,f,s] := A[w,n,e,k]*B[v,k,f,s]
    _fuse(_fuse(C,4,2),1,2)
end

"""Factorized two-site environment with full-hole PSD repair."""
struct BondEnvironment{L,R,T,B,S}
    left::L
    right::R
    top::T
    bottom::B
    outer_spaces::S
    direction::Symbol
end

function bond_environment(state,layers,env,i,j,direction;left=nothing,right=nothing)
    x,y=Tuple(i)
    top=env.north[y].tensors
    bottom=env.south[y+(direction==:horizontal ? 1 : 2)].tensors
    lastx=direction==:horizontal ? x+1 : x
    if left===nothing || right===nothing
        row=direction==:horizontal ? collect(layers[:,y]) :
            [SlabColumn(layers[k,y],layers[k,y+1]) for k in axes(layers,1)]
        edges=row_environments(top,row,bottom)
        left=edges.left[x]; right=edges.right[lastx+1]
    end
    if direction==:horizontal
        T=(top[x],top[x+1]); B=(bottom[x],bottom[x+1])
        outer=((i,1),(i,2),(i,6),(j,2),(j,5),(j,6))
    else
        # Split the column's west/east pair, retaining the fused double legs.
        if numind(left)==3
            left=_split(left,2,(dual(space(layers[x,y],1)),dual(space(layers[x,y+1],1))))
        end
        if numind(right)==3
            right=_split(right,2,(dual(space(layers[x,y],3)),dual(space(layers[x,y+1],3))))
        end
        T=(top[x],); B=(bottom[x],)
        outer=((i,1),(i,2),(i,5),(j,1),(j,5),(j,6))
    end
    spaces=Tuple(space(state[site],axis) for (site,axis) in outer)
    _row_bond_environment(layers,BondEnvironment(left,right,T,B,spaces,direction))
end
_row_bond_environment(layers,env)=env

"""Complete six-double-leg hole; its scales cancel in every FU equation."""
function bond_hole(env::BondEnvironment)
    left,right=env.left,env.right
    if env.direction==:horizontal
        T1,T2=env.top; B1,B2=env.bottom
        @tensor X[w,b,n,c] := left[a,w,b]*T1[a,n,c]
        @tensor L[c,w,n,f,s] := X[w,b,n,c]*B1[b,s,f]
        @tensor Y[c,v,e,g] := right[d,e,g]*T2[c,v,d]
        @tensor R[c,v,e,f,t] := Y[c,v,e,g]*B2[f,t,g]
        @tensor H[w,n,s,v,e,t] := L[c,w,n,f,s]*R[c,v,e,f,t]
    else
        T=only(env.top); B=only(env.bottom)
        @tensor L[w,v,b,n,c] := left[a,w,v,b]*T[a,n,c]
        @tensor R[b,s,c,e,f] := B[b,s,d]*right[c,e,f,d]
        @tensor H[w,n,e,v,f,s] := L[w,v,b,n,c]*R[b,s,c,e,f]
    end
    for k in 6:-1:1
        H=_split(H,k,(dual(env.outer_spaces[k]),env.outer_spaces[k]))
    end
    return H
end
bond_hole(H::AbstractTensorMap)=H
bond_hole(state,layers,env,i,j,direction;kwargs...)=
    bond_hole(bond_environment(state,layers,env,i,j,direction;kwargs...))
