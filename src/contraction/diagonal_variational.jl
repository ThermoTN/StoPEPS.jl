"""
    BoundaryMPSDiagonalVariational(chi; sector_space=nothing, max_sweep=4,
        threshold_sweep=1e-8, target_cache_bytes=256*1024^2)

Two-site variational contraction of a diagonal layer, initialized from the
preceding canonical boundary. A sweep is a complete round trip. From the second
sweep, `threshold_sweep` compares squared norms from retained spectra; zero
runs the full sweep budget. The environment diagnostic is the relative squared
layer fit residual. `target_cache_bytes` sets the per-layer target-block cache
budget; zero evaluates each block on demand.
"""
struct BoundaryMPSDiagonalVariational{S} <: AbstractDiagonalBoundaryMPS
    chi::Int
    sector_space::S
    max_sweep::Int
    threshold_sweep::Float64
    target_cache_bytes::Int
    function BoundaryMPSDiagonalVariational(chi::Int,sector_space;
            max_sweep::Int=4,threshold_sweep::Real=1e-8,target_cache_bytes::Int=256*1024^2)
        chi>=1 || throw(ArgumentError("chi must be positive"))
        max_sweep>=1 || throw(ArgumentError("max_sweep must be positive"))
        isfinite(threshold_sweep) && threshold_sweep>=0 ||
            throw(ArgumentError("threshold_sweep must be finite and nonnegative"))
        target_cache_bytes>=0 || throw(ArgumentError("target_cache_bytes must be nonnegative"))
        new{typeof(sector_space)}(chi,sector_space,max_sweep,Float64(threshold_sweep),target_cache_bytes)
    end
end
BoundaryMPSDiagonalVariational(chi::Int;sector_space=nothing,kwargs...)=
    BoundaryMPSDiagonalVariational(chi,sector_space;kwargs...)
BoundaryMPSDiagonalVariational(;chi=32,kwargs...)=BoundaryMPSDiagonalVariational(chi;kwargs...)

struct _VariationalNumericalError <: Exception
    reason::Symbol
end
Base.showerror(io::IO,e::_VariationalNumericalError)=print(io,"variational boundary: ",e.reason)
_variational_failure(e)=e isa _VariationalNumericalError ? e.reason :
    e isa LinearAlgebra.LAPACKException ? :factorization_failure : nothing

_variational_result(::BoundaryMPSDiagonalVariational,records)=_variational_summary(records)
function _variational_summary(records)
    (;boundary_variational=(;layers=copy(records),
        max_sweeps=maximum((r.sweeps for r in records);init=0),
        sweep_limit_hits=count(r->r.stop_reason==:sweep_limit,records),
        fallback_count=count(r->r.fallback!==nothing,records),
        max_relative_change=maximum((r.relative_change for r in records if isfinite(r.relative_change));init=0.0)))
end
_diagonal_entropy(::BoundaryMPSDiagonalVariational,l;profile=false)=
    BoundaryEntropy(0.0,profile ? (;northwest_cuts=NamedTuple[],southeast_cuts=NamedTuple[]) : nothing)

# Integer twice the transverse coordinate of an edge midpoint. This remains
# meaningful when the frontier grows, shrinks, or slides across a rectangle.
function _diagonal_coordinate(l,edge)
    nh=(l.Lx+1)*l.Ly
    if edge<=nh
        y,a=divrem(edge-1,l.Lx+1)
        return 2a+1-2(y+1)
    end
    b,a=divrem(edge-nh-1,l.Lx)
    2(a+1)-2b-1
end

mutable struct _VariationalTargetCache
    budget::Int
    blocks::Dict{Int,AbstractTensorMap}
    lru::Vector{Int} # oldest first; at most one entry per layer vertex
    bytes::Int
    peak_bytes::Int
    hits::Int
    builds::Int
end
_VariationalTargetCache(budget)=_VariationalTargetCache(budget,Dict{Int,AbstractTensorMap}(),Int[],0,0,0,0)
Base.length(cache::_VariationalTargetCache)=length(cache.blocks)
_variational_bytes(B)=dim(B)*sizeof(scalartype(B))
function _variational_cache!(cache,j,B)
    bytes=_variational_bytes(B)
    bytes>cache.budget && return B
    cache.budget==0 && return B
    while cache.bytes+bytes>cache.budget
        oldest=popfirst!(cache.lru)
        cache.bytes-=_variational_bytes(pop!(cache.blocks,oldest))
    end
    cache.blocks[j]=B;push!(cache.lru,j);cache.bytes+=bytes
    cache.peak_bytes=max(cache.peak_bytes,cache.bytes)
    B
end

# Per-layer counters for profiling contraction work.
Base.@kwdef mutable struct _VariationalWork
    message_steps::Int=0
    small_svd::Int=0
    full_svd::Int=0
    local_fallbacks::Dict{Symbol,Int}=Dict{Symbol,Int}()
end

"""Fixed input with a per-layer LRU cache of exact target blocks."""
mutable struct _DiagonalLayerTarget
    input::DiagonalBoundary
    tensors::Vector{AbstractTensorMap}
    lattice::Lattice
    sites::Vector{CartesianIndex{2}}
    reverse::Bool
    inputs::Vector{UnitRange{Int}}
    outputs::Vector{Vector{Int}}
    ranges::Vector{UnitRange{Int}}
    edges::Vector{Int}
    owner::Vector{Int}
    physical::Vector{ElementarySpace}
    identities::Vector{AbstractTensorMap}
    norms::Vector{Float64}
    weight::Float64
    log_scale::Float64
    cache::_VariationalTargetCache
    work::_VariationalWork
end

function _variational_raw_block(target,j)
    target.cache.builds+=1
    A=target.tensors[j]
    _diagonal_raw_block(target.input,A,A,target.lattice,target.sites[j],
        target.inputs[j],target.outputs[j];reverse=target.reverse)
end

function _variational_block(target,j)
    cache=target.cache
    if haskey(cache.blocks,j)
        cache.hits+=1
        deleteat!(cache.lru,something(findfirst(==(j),cache.lru)));push!(cache.lru,j)
        return cache.blocks[j]
    end
    B=_variational_raw_block(target,j)
    n=target.norms[j]
    B=iszero(n) ? B : (B/n)*target.weight
    _variational_cache!(cache,j,B)
end

# Accumulate the chain norm through QR with a logarithmic scale at each step.
# Temporary tensors preserve separate physical legs and the saved canonical form.
function _variational_chain_lognorm(gettensor,n)
    carry=nothing;scale=0.0
    for j in 1:n
        A=gettensor(j);rank=numind(A)
        if carry!==nothing
            R=carry
            A=_out(R*permute(A,((1,),Tuple(2:rank))))
        end
        _,R=left_orth(permute(A,(Tuple(1:rank-1),(rank,))))
        s=norm(R)
        isfinite(s) || throw(_VariationalNumericalError(:nonfinite_norm))
        iszero(s) && return -Inf
        scale+=log(s);carry=R/s
    end
    scale
end

function _variational_identity(V,T)
    v=_fuse(_out(id(T,V)),1,2;dual_output=isdual(V))
    v/norm(v)
end

function _variational_input_identity(target,edge)
    for (j,site) in enumerate(target.sites)
        ports=_diagonal_ports(target.lattice,site)
        k=findfirst(==(edge),ports)
        k===nothing && continue
        return _variational_identity(dual(space(target.tensors[j],_VIRTUAL_AXES[k])),scalartype(target.tensors[j]))
    end
    error("input edge missing from variational layer")
end

function _diagonal_layer_target(input,sites,l,t;reverse=false,target_cache_bytes=256*1024^2,
        work=_VariationalWork())
    target_cache_bytes>=0 || throw(ArgumentError("target_cache_bytes must be nonnegative"))
    layout=_diagonal_layer_layout(input,l,t;reverse)
    order=layout.sites;inputs=layout.inputs;outputs=layout.outputs;ranges=layout.ranges;edges=layout.edges
    tensors=AbstractTensorMap[sites[i] for i in order]
    owner=Int[];physical=ElementarySpace[];identities=AbstractTensorMap[]
    for (j,i) in enumerate(order)
        ports=_diagonal_ports(l,i)
        append!(owner,fill(j,length(outputs[j])))
        for port in outputs[j]
            k=something(findfirst(==(port),ports))
            axis=_VIRTUAL_AXES[k];A=sites[i]
            push!(physical,_double_space(A,A,axis))
            push!(identities,_variational_identity(space(A,axis),scalartype(A)))
        end
    end
    target=_DiagonalLayerTarget(copy(input),tensors,l,order,reverse,inputs,outputs,ranges,edges,owner,
        physical,identities,ones(length(order)),1.0,input.log_scale,
        _VariationalTargetCache(target_cache_bytes),work)
    lastbuilt=0
    function initial_block(j)
        B=_variational_raw_block(target,j);n=norm(B)
        isfinite(n) || error("nonfinite variational target block")
        target.norms[j]=n;lastbuilt=j
        # Use the stored Float64 norm, as on every later cache miss. This also
        # promotes low-precision payloads before cache byte accounting.
        _variational_cache!(target.cache,j,iszero(n) ? B : B/target.norms[j])
    end
    # The getter records each raw norm and feeds the same normalized block to
    # QR. Even after a zero chain norm, validate every remaining raw block.
    ln=isempty(edges) ? 0.0 : _variational_chain_lognorm(initial_block,length(order))
    for j in lastbuilt+1:length(order)
        initial_block(j)
    end
    if any(iszero,target.norms)
        target.log_scale=-Inf
    elseif !isempty(edges)
        target.log_scale+=sum(log,target.norms)+ln
        target.weight=isfinite(ln) ? exp(-ln/length(order)) : 1.0
        for (j,B) in target.cache.blocks
            target.cache.blocks[j]=B*target.weight
        end
    else
        target.log_scale+=sum(log,target.norms)
    end
    target
end

function _variational_product(target)
    B=AbstractTensorMap[]
    for v in target.identities
        V=oneunit(space(v,1));T=scalartype(v)
        tensor=_diagonal_join(_diagonal_unit(V',1;T),_DiagonalTensor(v,[2]))
        tensor=_diagonal_join(tensor,_diagonal_unit(V,3;T))
        push!(B,_diagonal_reorder(tensor,[1,2,3]).data)
    end
    DiagonalBoundary(B,copy(target.edges),1,0.0,0.0,1.0)
end

# A deterministic common-sector map: expansion is isometric, and projection
# requires recanonicalization of the affected region.
function _variational_space_map(P,Q,T)
    P==Q && return id(T,P)
    R=infimum(P,Q)
    isometry(T,Q,R)*adjoint(isometry(T,P,R))
end

function _variational_warm_start(target)
    old=target.input;isempty(old.tensors) && return _variational_product(target)
    C=copy(old);C.log_scale=0.0;C.max_truncation=0.0;empty!(C.variational)
    coordinates=[_diagonal_coordinate(target.lattice,e) for e in target.edges]
    oldcoordinates=[_diagonal_coordinate(target.lattice,e) for e in C.edges]
    # Removing an endpoint is a physical projection. Move the center there
    # before removing it, so the surviving part is already canonical.
    while !isempty(C.tensors) && !(first(oldcoordinates) in coordinates)
        _diagonal_center!(C,1)
        A=first(C.tensors)
        cap=_conj(_variational_input_identity(target,first(C.edges)))
        H=_diagonal_join(_DiagonalTensor(A,[1,2,3]),_DiagonalTensor(cap,[2]))
        H=_diagonal_join(H,_diagonal_unit(dual(space(A,1)),1;T=scalartype(A)))
        popfirst!(C.tensors);popfirst!(C.edges);popfirst!(oldcoordinates)
        isempty(C.tensors) && break
        R=_diagonal_join(H,_DiagonalTensor(C.tensors[1],[3,4,5]))
        V=oneunit(space(A,1))
        R=_diagonal_join(_diagonal_unit(V,1;T=scalartype(A)),R)
        C.tensors[1]=_diagonal_reorder(R,[1,4,5]).data;C.center=1
    end
    while !isempty(C.tensors) && !(last(oldcoordinates) in coordinates)
        _diagonal_center!(C,length(C.tensors))
        A=last(C.tensors)
        cap=_conj(_variational_input_identity(target,last(C.edges)))
        H=_diagonal_join(_DiagonalTensor(A,[1,2,3]),_DiagonalTensor(cap,[2]))
        H=_diagonal_join(H,_diagonal_unit(dual(space(A,3)),3;T=scalartype(A)))
        pop!(C.tensors);pop!(C.edges);pop!(oldcoordinates)
        isempty(C.tensors) && break
        R=_diagonal_join(_DiagonalTensor(C.tensors[end],[4,5,1]),H)
        R=_diagonal_join(R,_diagonal_unit(oneunit(space(A,3)),3;T=scalartype(A)))
        C.tensors[end]=_diagonal_reorder(R,[4,5,3]).data;C.center=length(C.tensors)
    end
    isempty(C.tensors) && return _variational_product(target)
    # Process physical maps at their centers. Isometric embeddings preserve
    # canonical form; projections are followed by center motion.
    for i in eachindex(C.tensors)
        j=something(findfirst(==(oldcoordinates[i]),coordinates))
        A=C.tensors[i];P=space(A,2);Q=target.physical[j]
        P==Q && continue
        if isdual(P)!=isdual(Q)
            # FU may reverse a virtual arrow while retaining its representation.
            # A native unitary flip preserves the initial guess's isometries.
            C.tensors[i]=flip(A,(2,))
            P=space(C.tensors[i],2)
        end
        infimum(P,Q)==P || _diagonal_center!(C,i)
        C.tensors[i]=_act(C.tensors[i],_variational_space_map(P,Q,scalartype(A)),2)
    end
    product=_variational_product(target)
    while first(oldcoordinates)>first(coordinates)
        j=something(findfirst(==(first(oldcoordinates)),coordinates))-1
        pushfirst!(C.tensors,product.tensors[j]);pushfirst!(oldcoordinates,coordinates[j]);C.center+=1
    end
    while last(oldcoordinates)<last(coordinates)
        j=something(findfirst(==(last(oldcoordinates)),coordinates))+1
        push!(C.tensors,product.tensors[j]);push!(oldcoordinates,coordinates[j])
    end
    C.edges=copy(target.edges)
    s=norm(C.tensors[C.center])
    iszero(s) && return product
    isfinite(s) || error("nonfinite variational warm start")
    C.tensors[C.center]=C.tensors[C.center]/s
    C
end

struct _VariationalMessage
    tensor::_DiagonalTensor
    log_scale::Float64
end
function _variational_message(t,log_scale=0.0)
    s=norm(t.data)
    isfinite(s) || throw(_VariationalNumericalError(:nonfinite_overlap))
    _VariationalMessage(iszero(s) ? t : _DiagonalTensor(t.data/s,t.labels),
        iszero(s) ? -Inf : log_scale+log(s))
end
_variational_target_bond(target,j)=_diagonal_edge_count(target.lattice)+j
_variational_candidate_bond(target,i)=_diagonal_edge_count(target.lattice)+length(target.sites)+1+i
function _variational_labelled_block(target,j)
    _DiagonalTensor(_variational_block(target,j),
        [_variational_target_bond(target,j);target.outputs[j];_variational_target_bond(target,j+1)])
end
function _variational_edge(target,C,left)
    j=left ? 1 : length(target.sites);i=left ? 1 : length(C.tensors)
    H=_variational_block(target,j);A=C.tensors[i]
    a=left ? 1 : numind(H);b=left ? 1 : 3
    tb=_variational_target_bond(target,j+(left ? 0 : 1))
    cb=_variational_candidate_bond(target,i+(left ? 0 : 1))
    E=_diagonal_join(_diagonal_unit(dual(space(H,a)),tb;T=scalartype(H)),
        _diagonal_unit(space(A,b),cb;T=scalartype(A)))
    _variational_message(E)
end
function _variational_step(message,target,i,A,left)
    target.work.message_steps+=1
    j=target.owner[i];range=target.ranges[j];E=message.tensor
    if i==(left ? first(range) : last(range))
        E=_diagonal_join(E,_variational_labelled_block(target,j))
    end
    bra=_conj(A)
    E=_diagonal_join(E,_DiagonalTensor(bra,
        [_variational_candidate_bond(target,i),target.edges[i],_variational_candidate_bond(target,i+1)]))
    _variational_message(E,message.log_scale)
end
function _variational_messages(target,C,left)
    n=length(C.tensors);messages=Vector{_VariationalMessage}(undef,n+1)
    messages[left ? 1 : n+1]=_variational_edge(target,C,left)
    for i in (left ? (1:n) : (n:-1:1))
        messages[left ? i+1 : i]=_variational_step(messages[left ? i : i+1],target,i,C.tensors[i],left)
    end
    messages
end
function _variational_pair(target,i,left,right)
    E=left.tensor
    for j in unique(target.owner[i:i+1])
        range=target.ranges[j]
        first(range)>=i && last(range)<=i+1 || continue
        E=_diagonal_join(E,_variational_labelled_block(target,j))
    end
    E=_diagonal_join(E,right.tensor)
    labels=[_variational_candidate_bond(target,i),target.edges[i],target.edges[i+1],
        _variational_candidate_bond(target,i+2)]
    _diagonal_reorder(E,labels).data*exp(left.log_scale+right.log_scale)
end

# Factor the cross-block projection through the input boundary bond,
# which bounds its rank.
function _variational_pair_factors(target,i,left,right)
    jl=target.owner[i];jr=target.owner[i+1]
    jl!=jr || throw(ArgumentError("pair factors require a cross-block cut"))
    L=left.tensor;R=right.tensor
    length(target.ranges[jl])==1 && (L=_diagonal_join(L,_variational_labelled_block(target,jl)))
    length(target.ranges[jr])==1 && (R=_diagonal_join(_variational_labelled_block(target,jr),R))
    bond=_variational_target_bond(target,jr)
    L=_diagonal_reorder(L,[_variational_candidate_bond(target,i),target.edges[i],bond]).data
    R=_diagonal_reorder(R,[bond,target.edges[i+1],_variational_candidate_bond(target,i+2)]).data
    permute(L,((1,2),(3,))),
        permute(R,((1,),(2,3)))
end

function _variational_svd_guard(S,fullspace,trunc,m,n)
    # Pad the diagonal spectrum for native multiplet, rank and sector selection.
    padded=TensorKit.similar_diagonal(S,fullspace)
    fill!(padded,0)
    for (c,b) in blocks(S)
        copyto!(TensorKit.diagview(block(padded,c)),TensorKit.diagview(b))
    end
    fullinds=MAK.findtruncated_svd(TensorKit.diagview(padded),trunc)
    inds=MAK.findtruncated_svd(TensorKit.diagview(S),trunc)
    TensorKit.Factorizations.truncate_space(fullspace,fullinds)==
        TensorKit.Factorizations.truncate_space(space(S,1),inds) ||
        return :null_space
    kept=real(scalartype(S))[];discarded=similar(kept)
    for (c,b) in blocks(padded)
        values=TensorKit.diagview(b)
        selected=falses(length(values));selected[fullinds[c]].=true
        append!(kept,values[selected]);append!(discarded,values[.!selected])
    end
    largest=max(maximum(kept;init=zero(eltype(kept))),maximum(discarded;init=zero(eltype(kept))))
    tol=128eps(real(scalartype(S)))*max(m,n)*largest
    # Use the full SVD at near-zero or degenerate truncation boundaries.
    any(s->s<=tol,kept) && return :null_space
    any(a->any(b->abs(a-b)<=tol,discarded),kept) && return :degenerate_cut
    nothing
end

function _variational_small_svd(L,R,scale,trunc,observer,cut)
    m=dim(codomain(L));n=dim(domain(R));r=dim(domain(L))
    r<min(m,n) || return nothing,:not_smaller
    QL,RL=left_orth(L);LR,QR=right_orth(R)
    core=(RL*LR)*scale
    alg=MAK.select_algorithm(MAK.svd_trunc!,core,nothing;trunc)
    U,S,V=svd_compact(core,alg.alg)
    fullspace=infimum(fuse(codomain(L)),fuse(domain(R)))
    failure=_variational_svd_guard(S,fullspace,alg.trunc,m,n)
    failure===nothing || return nothing,failure
    magnitude=norm(S)
    observer===nothing || observer(cut,_schmidt_entropy(S))
    (U,Skept,V),indices=MAK.truncate(MAK.svd_trunc!,(U,S,V),alg.trunc)
    err=MAK.truncation_error!(TensorKit.diagview(S),indices)
    (QL*U,Skept,V*QR,err,magnitude),nothing
end

function _variational_project_svd(target,i,left,right,method,observer)
    trunc=_truncation(method.chi,method.sector_space)
    if target.owner[i]!=target.owner[i+1]
        L,R=_variational_pair_factors(target,i,left,right)
        result=nothing;failure=nothing
        try
            result,failure=_variational_small_svd(L,R,exp(left.log_scale+right.log_scale),trunc,observer,i)
        catch exception
            exception isa LinearAlgebra.LAPACKException || rethrow()
            failure=:factorization_failure
        end
        if result!==nothing
            target.work.small_svd+=1
            return result
        end
        target.work.local_fallbacks[failure]=get(target.work.local_fallbacks,failure,0)+1
    end
    target.work.full_svd+=1
    M=permute(_variational_pair(target,i,left,right),((1,2),(3,4)))
    U,S,V,err=_boundary_svd(M,trunc,observer,i)
    U,S,V,err,norm(M)
end

function _variational_nonnegative_residual(value,scale=1.0)
    isfinite(value) || throw(_VariationalNumericalError(:nonfinite_residual))
    value>=-512eps(Float64)*max(1.0,scale) ||
        throw(_VariationalNumericalError(:negative_residual))
    max(0.0,value)
end
function _variational_projected_residual(norm2)
    isfinite(norm2) && norm2>=0 || throw(_VariationalNumericalError(:nonfinite_norm))
    _variational_nonnegative_residual(1-norm2,norm2)
end

# Evaluate mixed overlaps for the initial and canonical-SVD candidates.
# Initialization reuses the terminal message from the first half sweep.
function _variational_overlap_residual(target,C;left=true,terminal=nothing)
    if terminal===nothing
        terminal=_variational_edge(target,C,left)
        for i in (left ? eachindex(C.tensors) : reverse(eachindex(C.tensors)))
            terminal=_variational_step(terminal,target,i,C.tensors[i],left)
        end
    end
    edge=_variational_edge(target,C,!left)
    overlap=_diagonal_scalar(_diagonal_join(terminal.tensor,edge.tensor))*
        exp(terminal.log_scale+edge.log_scale)
    norm2=abs2(norm(C.tensors[C.center]))
    isfinite(overlap) && isfinite(norm2) || throw(_VariationalNumericalError(:nonfinite_overlap))
    _variational_nonnegative_residual(1+norm2-2real(overlap),max(norm2,2abs(overlap)))
end

function _variational_half_sweep!(C,target,method,right,entropy,profile,
        fixed=_variational_messages(target,C,!right))
    n=length(C.tensors)
    rolling=_variational_edge(target,C,right)
    nextfixed=Vector{_VariationalMessage}(undef,n+1)
    nextfixed[right ? 1 : n+1]=rolling
    maxdiscard=0.0;lastS=nothing
    for i in (right ? (1:n-1) : (n-1:-1:1))
        left=right ? rolling : fixed[i];r=right ? fixed[i+2] : rolling
        observer=entropy===nothing ? nothing : (cut,value)->begin
            entropy.maximum=max(entropy.maximum,value)
            profile[cut]=value
        end
        U,S,V,err,magnitude=_variational_project_svd(target,i,left,r,method,observer)
        iszero(magnitude) && throw(_VariationalNumericalError(:zero_local_projection))
        lastS=S
        maxdiscard=max(maxdiscard,(err/magnitude)^2)
        C.tensors[i]=_out(right ? U : U*S)
        C.tensors[i+1]=_out(right ? S*V : V)
        C.center=right ? i+1 : i
        k=right ? i : i+1
        rolling=_variational_step(rolling,target,k,C.tensors[k],right)
        nextfixed[right ? k+1 : k]=rolling
    end
    # The outside tensors are isometric: the last retained spectrum gives
    # the candidate's squared norm with native sector-dimension weights.
    norm2=abs2(norm(lastS))
    iszero(norm2) && throw(_VariationalNumericalError(:zero_local_projection))
    maxdiscard,nextfixed,norm2
end

function _variational_fit(target,method,entropy)
    C=_variational_warm_start(target);n=length(C.tensors)
    # Choose the nearer endpoint; the previous center is usually one edge away.
    right=C.center-1<n-C.center || (C.center-1==n-C.center && !target.reverse)
    _diagonal_center!(C,right ? 1 : n)
    fixed=_variational_messages(target,C,!right)
    initial=_variational_overlap_residual(target,C;left=!right,terminal=fixed[right ? 1 : n+1])
    best=copy(C);cost=initial
    bestprofile=fill(NaN,max(0,n-1));relative_change=Inf;discard=0.0;sweeps=0;reason=:sweep_limit;fallback=nothing
    previous_norm2=NaN
    for sweep in 1:method.max_sweep
        profile=fill(NaN,n-1)
        sweeps=sweep
        current=Inf
        try
            localdiscard,fixed,_=_variational_half_sweep!(C,target,method,right,entropy,profile,fixed)
            discard=max(discard,localdiscard)
            localdiscard,fixed,norm2=_variational_half_sweep!(C,target,method,!right,entropy,profile,fixed)
            discard=max(discard,localdiscard)
            current=_variational_projected_residual(norm2)
            current<=cost+512eps(Float64)*max(1.0,cost) || throw(_VariationalNumericalError(:residual_increase))
            relative_change=sweep==1 ? Inf :
                abs(norm2-previous_norm2)/max(norm2,previous_norm2,floatmin(Float64))
            previous_norm2=norm2
        catch exception
            failure=_variational_failure(exception)
            failure===nothing && rethrow()
            fallback=(;reason=failure,used_svd=false);reason=:numerical_fallback;break
        end
        if current<=cost
            best=copy(C);cost=current;bestprofile=copy(profile)
        end
        if sweep>1 && method.threshold_sweep>0 && relative_change<=method.threshold_sweep
            reason=:threshold_sweep;break
        end
    end
    best,(;sweeps,relative_change,stop_reason=reason,initial_residual=initial,
        residual=cost,max_local_discarded_weight=discard,fallback),bestprofile
end

function _variational_recover(target,method,C,report,profile)
    alternate=copy(target.input);alternate.max_truncation=0.0
    # Compare the recovery candidate against the same frozen target.
    sites=Dict(zip(target.sites,target.tensors))
    t=sum(Tuple(first(target.sites)))
    _diagonal_absorb_layer!(alternate,sites,target.lattice,t,
        BoundaryMPSDiagonal(method.chi,method.sector_space);reverse=target.reverse)
    alternate.tensors[alternate.center]*=exp(alternate.log_scale-target.log_scale)
    cost=_variational_overlap_residual(target,alternate)
    isfinite(cost) || throw(_VariationalNumericalError(:failed_svd_fallback))
    use_svd=C===nothing || cost<report.residual
    if use_svd
        C=alternate;profile=fill(NaN,length(C.tensors)-1)
    end
    report=merge(report,(;residual=use_svd ? cost : report.residual,
        max_local_discarded_weight=max(report.max_local_discarded_weight,alternate.max_truncation),
        fallback=merge(report.fallback,(;used_svd=use_svd))))
    C,report,profile
end

function _diagonal_absorb_layer!(env,sites,l,t,method::BoundaryMPSDiagonalVariational;
        reverse=false,entropy=nothing,work=_VariationalWork())
    target=_diagonal_layer_target(env,sites,l,t;reverse,target_cache_bytes=method.target_cache_bytes,work)
    n=length(target.edges);profile=fill(NaN,max(0,n-1))
    report=(;sweeps=0,relative_change=0.0,stop_reason=:direct,initial_residual=0.0,
        residual=0.0,max_local_discarded_weight=0.0,fallback=nothing)
    if target.log_scale==-Inf
        C=n==0 ? DiagonalBoundary() : _variational_product(target)
        n==0 ? (C.scalar=0.0) : (C.tensors[C.center]=zero(C.tensors[C.center]))
        report=merge(report,(;stop_reason=:zero_target,residual=NaN,initial_residual=NaN))
    elseif n<=1
        H=_variational_block(target,1)
        if n==0
            E=_DiagonalTensor(H,[1,2])
            for k in 1:2
                E=_diagonal_join(E,_diagonal_unit(dual(space(H,k)),k;T=scalartype(H)))
            end
            C=DiagonalBoundary();C.scalar=_diagonal_scalar(E)
        else
            C=DiagonalBoundary(AbstractTensorMap[H],copy(target.edges),1,0.0,0.0,1.0)
        end
    elseif length(target.sites)==1
        H=_variational_block(target,1);M=permute(H,((1,2),(3,4)))
        observer=entropy===nothing ? nothing : (k,s)->begin entropy.maximum=max(entropy.maximum,s);profile[k]=s end
        U,S,V,err=_boundary_svd(M,_truncation(method.chi,method.sector_space),observer,1)
        C=DiagonalBoundary(AbstractTensorMap[_out(U),_out(S*V)],copy(target.edges),2,0.0,0.0,1.0)
        report=merge(report,(;residual=abs2(err),max_local_discarded_weight=abs2(err)))
    else
        try
            C,report,profile=_variational_fit(target,method,entropy)
        catch exception
            failure=_variational_failure(exception)
            failure===nothing && rethrow()
            C=nothing
            report=merge(report,(;stop_reason=:numerical_fallback,initial_residual=NaN,
                residual=Inf,fallback=(;reason=failure,used_svd=false)))
        end
        report.fallback===nothing || ((C,report,profile)=_variational_recover(target,method,C,report,profile))
    end
    scale=n==0 ? 1.0 : norm(C.tensors[C.center])
    isfinite(scale) || error("nonfinite variational boundary norm")
    if n>0 && !iszero(scale)
        C.tensors[C.center]=C.tensors[C.center]/scale
    end
    env.tensors=C.tensors;env.edges=C.edges;env.center=C.center;env.scalar=C.scalar
    env.log_scale=iszero(scale) ? -Inf : target.log_scale+log(scale)
    isnan(report.residual) || (env.max_truncation=max(env.max_truncation,report.residual))
    push!(env.variational,merge((;layer=t,reverse),report))
    if entropy!==nothing && entropy.Se!==nothing
        records=getproperty(entropy.Se,reverse ? :southeast_cuts : :northwest_cuts)
        append!(records,[(;layer=t,left_edge=env.edges[i],right_edge=env.edges[i+1],entropy=profile[i]) for i in eachindex(profile)])
    end
    env
end
