abstract type AbstractDiagonalBoundaryMPS <: AbstractBoundaryMPS end

"""Absorb a complete diagonal as ket/bra blocks, then canonicalize and truncate."""
struct BoundaryMPSDiagonal{S} <: AbstractDiagonalBoundaryMPS
    chi::Int
    sector_space::S
end
BoundaryMPSDiagonal(chi::Int)=BoundaryMPSDiagonal(chi,nothing)
BoundaryMPSDiagonal(;chi=32,sector_space=nothing)=BoundaryMPSDiagonal(chi,sector_space)

const _VIRTUAL_AXES=(1,2,5,6)

# Positive labels denote lattice edges or boundary-MPS bonds; negative labels
# identify internal legs of a local ket/bra contraction.
struct _DiagonalTensor
    data::Union{Number,AbstractTensorMap}
    labels::Vector{Int}
    function _DiagonalTensor(t::Union{Number,AbstractTensorMap},labels)
        new(t isa Number ? t : _out(t),Int[labels...])
    end
end
function _diagonal_join(a::_DiagonalTensor,b::_DiagonalTensor)
    a.data isa Number && return _DiagonalTensor(a.data*b.data,b.labels)
    b.data isa Number && return _DiagonalTensor(a.data*b.data,a.labels)
    common=intersect(a.labels,b.labels)
    free=vcat(setdiff(a.labels,common),setdiff(b.labels,common))
    encode(k)=k in common ? findfirst(==(k),common) : -findfirst(==(k),free)
    result=ncon([a.data,b.data],[map(encode,a.labels),map(encode,b.labels)];output=collect(-1:-1:-length(free)))
    _DiagonalTensor(result,free)
end
function _diagonal_reorder(a,labels)
    length(labels)==length(a.labels) || throw(DimensionMismatch("diagonal tensor labels"))
    isempty(labels) && return a
    indices=map(k->something(findfirst(==(k),a.labels)),labels)
    _DiagonalTensor(_perm(a.data,Tuple(indices)),labels)
end
function _diagonal_unit(V,label;T=Float64)
    _DiagonalTensor(_unit_tensor((V,);T),[label])
end
function _diagonal_scalar(a)
    isempty(a.labels) || error("unclosed diagonal network")
    a.data isa Number ? a.data : TensorKit.scalar(a.data)
end
function _diagonal_scale(a)
    value=norm(a.data)
    isfinite(value) || error("nonfinite diagonal environment")
    iszero(value) ? a : _DiagonalTensor(a.data/value,a.labels)
end

"""Virtual fusion space of a double-layer leg."""
function _double_space(A,B,axis)
    V=fuse(space(A,axis)⊗dual(space(B,axis)))
    isdual(V)==isdual(space(A,axis)) ? V : flip(V)
end

"""Contract two fused virtual inputs with ket, then bra; retain other legs.

The input labels may be any two virtual ports. This also serves strip messages
and reduced FU insertions. `physical` leaves the two physical/core legs open.
"""
function _diagonal_apply(message,A,B,ports,inputs;physical=nothing)
    length(inputs)==2 || throw(ArgumentError("a diagonal vertex has two input ports"))
    ket=_fermionic(A) ? _twistdual(A,physical===nothing ? (3,4) : (4,)) : A
    bra=_fermionic(B) ? twist(B,_VIRTUAL_AXES) : B
    rawket=(-1,-3,-9,-10,-5,-7)
    rawbra=(-2,-4,-9,-10,-6,-8)
    if physical!==nothing
        rawket=Base.setindex(rawket,physical[1],3)
        rawbra=Base.setindex(rawbra,physical[2],3)
    end
    for port in inputs
        axis=_VIRTUAL_AXES[something(findfirst(==(port),ports))]
        position=something(findfirst(==(port),message.labels))
        if sectortype(A)===Trivial
            data=_split(message.data,position,(dual(space(A,axis)),space(B,axis)))
            labels=copy(message.labels)
            splice!(labels,position:position,[rawket[axis],rawbra[axis]])
            message=_DiagonalTensor(data,labels)
        else
            F=_fusionmap(promote_type(scalartype(A),scalartype(B)),
                (space(A,axis),dual(space(B,axis))),isdual(space(A,axis)))
            message=_diagonal_join(message,_DiagonalTensor(F,[port,rawket[axis],rawbra[axis]]))
        end
    end
    message=_diagonal_join(message,_DiagonalTensor(ket,rawket))
    message=_diagonal_join(message,_DiagonalTensor(_conj(bra),rawbra))
    for (axis,port) in zip(_VIRTUAL_AXES,ports)
        port in inputs && continue
        pair=[rawket[axis],rawbra[axis]]
        others=setdiff(message.labels,pair)
        message=_diagonal_reorder(message,vcat(others,pair))
        message=_DiagonalTensor(_fuse(message.data,length(others)+1,2),vcat(others,port))
    end
    message
end

# Each external OBC edge also has a unique label; its unit space is contracted
# explicitly. Lattice and configuration storage remain sites[x,y].
function _diagonal_ports(l,site)
    x,y=Tuple(site)
    horizontal(a,b)=a+1+(b-1)*(l.Lx+1)
    vertical(a,b)=(l.Lx+1)*l.Ly+a+b*l.Lx
    (horizontal(x-1,y),vertical(x,y-1),horizontal(x,y),vertical(x,y))
end
_diagonal_edge_count(l)=(l.Lx+1)*l.Ly+(l.Ly+1)*l.Lx
function _diagonal_sites(l,t;reverse=false)
    sites=[CartesianIndex(x,t-x) for x in max(1,t-l.Ly):min(l.Lx,t-1)]
    isodd(t) && reverse!(sites)
    reverse && reverse!(sites)
    sites
end
_diagonal_order(l)=[i for t in 2:l.Lx+l.Ly for i in _diagonal_sites(l,t)]

mutable struct DiagonalBoundary
    tensors::Vector{AbstractTensorMap}
    edges::Vector{Int}
    center::Int
    log_scale::Float64
    max_truncation::Float64
    scalar::Number
    variational::Vector{NamedTuple}
end
DiagonalBoundary(B,edges,center,scale,error,scalar)=
    DiagonalBoundary(B,edges,center,scale,error,scalar,NamedTuple[])
DiagonalBoundary()=DiagonalBoundary(AbstractTensorMap[],Int[],0,0.0,0.0,1.0)
Base.copy(e::DiagonalBoundary)=DiagonalBoundary(copy(e.tensors),copy(e.edges),e.center,
    e.log_scale,e.max_truncation,e.scalar,copy(e.variational))

function _diagonal_center!(env,target)
    B=env.tensors
    while env.center<target
        i=env.center
        Q,R=left_orth(permute(B[i],((1,2),(3,))))
        B[i]=_out(Q);R=_twistdual(R,(1,));A=B[i+1]
        @tensor C[l,p,r] := R[l,k]*A[k,p,r]
        B[i+1]=C;env.center+=1
    end
    while env.center>target
        i=env.center
        L,Q=right_orth(permute(B[i],((1,),(2,3))))
        B[i]=_out(_twistdual(Q,(1,)));A=B[i-1]
        @tensor C[l,p,r] := A[l,p,k]*L[k,r]
        B[i-1]=C;env.center-=1
    end
    nothing
end
function _diagonal_entropy(l;profile=false)
    BoundaryEntropy(0.0,profile ?
        (;northwest=fill(NaN,l.Lx,l.Ly),southeast=fill(NaN,l.Lx,l.Ly)) : nothing)
end
function _diagonal_observer(entropy,reverse,site)
    entropy===nothing && return nothing
    (bond,value)->begin
        entropy.maximum=max(entropy.maximum,value)
        entropy.Se===nothing || (getproperty(entropy.Se,reverse ? :southeast : :northwest)[site]=value)
    end
end

function _diagonal_layer_layout(input,l,t;reverse=false)
    sites=sort(_diagonal_sites(l,t);by=i->i[1]-i[2])
    inputs=UnitRange{Int}[];outputs=Vector{Int}[];ranges=UnitRange{Int}[];edges=Int[]
    for site in sites
        ports=_diagonal_ports(l,site);x,y=Tuple(site)
        inside=(x>1,y>1,x<l.Lx,y<l.Ly)
        ia=reverse ? (4,3) : (1,2);oa=reverse ? (1,2) : (4,3)
        incoming=Int[ports[k] for k in ia if inside[k]]
        p=isempty(incoming) ? 1 : something(findfirst(==(first(incoming)),input.edges))
        indices=p:p+length(incoming)-1
        input.edges[indices]==incoming || error("nonadjacent diagonal frontier inputs")
        out=Int[ports[k] for k in oa if inside[k]]
        push!(inputs,indices);push!(outputs,out);push!(ranges,length(edges)+1:length(edges)+length(out))
        append!(edges,out)
    end
    (;sites,inputs,outputs,ranges,edges)
end

"""Exact vertex block from a fixed input, with separate output legs."""
function _diagonal_raw_block(input,A,B,l,site,indices,outputs;reverse=false)
    ports=_diagonal_ports(l,site);x,y=Tuple(site)
    inaxes=reverse ? (4,3) : (1,2)
    outaxes=reverse ? (1,2) : (4,3)
    inside=(x>1,y>1,x<l.Lx,y<l.Ly)
    inputs=Int[ports[k] for k in inaxes if inside[k]]
    leftlabel=_diagonal_edge_count(l)+1;rightlabel=leftlabel+1
    if isempty(inputs)
        isempty(input.tensors) || error("disconnected diagonal initialization")
        V=oneunit(space(A,1))
        phi=_DiagonalTensor(_unit_tensor((V',V);T=scalartype(A)),[leftlabel,rightlabel])
    else
        phi=_DiagonalTensor(input.tensors[first(indices)],[leftlabel,inputs[1],rightlabel])
        if length(inputs)==2
            phi=_DiagonalTensor(phi.data,[leftlabel,inputs[1],rightlabel+1])
            phi=_diagonal_join(phi,_DiagonalTensor(input.tensors[last(indices)],
                [rightlabel+1,inputs[2],rightlabel]))
        end
    end
    for k in inaxes
        inside[k] && continue
        phi=_diagonal_join(phi,_diagonal_unit(dual(_double_space(A,B,_VIRTUAL_AXES[k])),ports[k];T=scalartype(A)))
    end
    theta=_diagonal_apply(phi,A,B,ports,Int[ports[k] for k in inaxes])
    for k in outaxes
        inside[k] && continue
        theta=_diagonal_join(theta,_diagonal_unit(dual(_double_space(A,B,_VIRTUAL_AXES[k])),ports[k];T=scalartype(A)))
    end
    _diagonal_reorder(theta,[leftlabel;outputs;rightlabel]).data
end

function _diagonal_normalized(A)
    scale=norm(A)
    isfinite(scale) || error("nonfinite diagonal boundary norm")
    iszero(scale) ? (A,-Inf) : (A/scale,log(scale))
end

# Move the center between blocks with separate physical legs.
# External bonds stay at the input chi: block QR costs O(chi^3*D^4).
function _diagonal_block_center_step!(blocks,j,right)
    A=blocks[j];rank=numind(A)
    if right
        Q,R=left_orth(permute(A,(Tuple(1:rank-1),(rank,))))
        blocks[j]=_out(Q)
        R,scale=_diagonal_normalized(R)
        next=blocks[j+1]
        blocks[j+1]=_out(_twistdual(R,(1,))*permute(next,((1,),Tuple(2:numind(next)))))
    else
        L,Q=right_orth(permute(A,((1,),Tuple(2:rank))))
        blocks[j]=_out(_twistdual(Q,(1,)))
        L,scale=_diagonal_normalized(L)
        previous=blocks[j-1];last=numind(previous)
        blocks[j-1]=_out(permute(previous,(Tuple(1:last-1),(last,)))*L)
    end
    scale
end

function _diagonal_compress_blocks!(blocks,layout,method;right,reverse=false,entropy=nothing)
    n=length(blocks);scale=0.0;max_error=0.0
    # Establish an exact canonical environment for the complete absorbed layer.
    for j in (right ? (n:-1:2) : (1:n-1))
        scale+=_diagonal_block_center_step!(blocks,j,!right)
    end
    tensors=Vector{AbstractTensorMap}(undef,length(layout.edges))
    for j in (right ? (1:n) : (n:-1:1))
        cut=layout.ranges[j]
        if length(cut)==2
            # Within-block bonds expand. This SVD is at most
            # (chi*D^2) by (chi*D^2), retaining the O(chi^3*D^6) bound.
            M=permute(blocks[j],((1,2),(3,4)))
            U,S,V,err=_boundary_svd(M,_truncation(method.chi,method.sector_space),
                _diagonal_observer(entropy,reverse,layout.sites[j]),first(cut))
            magnitude=norm(M)
            max_error=max(max_error,iszero(magnitude) ? 0.0 : (err/magnitude)^2)
            if right
                tensors[first(cut)]=_out(U)
                blocks[j]=_out(_twistdual(S*V,(1,)))
            else
                tensors[last(cut)]=_out(_twistdual(V,(1,)))
                blocks[j]=_out(U*S)
            end
        end
        # Between blocks, the input boundary already satisfies the chi/sector
        # budget. Move the center exactly through this unexpanded bond.
        if j==(right ? n : 1)
            blocks[j],increment=_diagonal_normalized(blocks[j])
            scale+=increment
        else
            scale+=_diagonal_block_center_step!(blocks,j,right)
        end
        tensors[right ? last(cut) : first(cut)]=blocks[j]
    end
    (;tensors,center=right ? length(tensors) : 1,log_scale=scale,max_truncation=max_error)
end

function _diagonal_absorb_layer!(env,sites,l,t,method::BoundaryMPSDiagonal;reverse=false,entropy=nothing)
    method.chi>=1 || throw(ArgumentError("chi must be positive"))
    layout=_diagonal_layer_layout(env,l,t;reverse)
    blocks=AbstractTensorMap[];scale=env.log_scale
    for (j,i) in enumerate(layout.sites)
        A=_diagonal_raw_block(env,sites[i],sites[i],l,i,layout.inputs[j],layout.outputs[j];reverse)
        A,increment=_diagonal_normalized(A)
        push!(blocks,A);scale+=increment
    end
    if isempty(layout.edges)
        # The final OBC corner has two unit boundary bonds.
        A=only(blocks);theta=_DiagonalTensor(A,[1,2])
        for k in 1:2
            theta=_diagonal_join(theta,_diagonal_unit(dual(space(A,k)),k;T=scalartype(A)))
        end
        env.tensors=AbstractTensorMap[];env.center=0;env.scalar=_diagonal_scalar(theta)
    else
        right=length(blocks)<2 ? !reverse : xor(iseven(t),reverse)
        result=_diagonal_compress_blocks!(blocks,layout,method;right,reverse,entropy)
        env.tensors=result.tensors;env.center=result.center
        env.max_truncation=max(env.max_truncation,result.max_truncation)
        scale+=result.log_scale
    end
    env.edges=layout.edges;env.log_scale=scale
    env
end
function _diagonal_boundaries(state,method;side=:both,entropy=nothing)
    side in (:both,:northwest,:southeast) || throw(ArgumentError("invalid diagonal boundary side"))
    l=state.lattice;last=l.Lx+l.Ly
    northwest=Vector{DiagonalBoundary}(undef,last+1);southeast=similar(northwest)
    northwest[2]=DiagonalBoundary();southeast[last+1]=DiagonalBoundary()
    if side!=:southeast
        for t in 2:last
            northwest[t+1]=_diagonal_absorb_layer!(copy(northwest[t]),state.sites,l,t,method;entropy)
        end
    end
    if side!=:northwest
        for t in last:-1:2
            southeast[t]=_diagonal_absorb_layer!(copy(southeast[t+1]),state.sites,l,t,method;reverse=true,entropy)
        end
    end
    (;northwest,southeast)
end
function lognorm(state::FiniteState,method::AbstractDiagonalBoundaryMPS)
    env=DiagonalBoundary()
    for t in 2:state.lattice.Lx+state.lattice.Ly
        _diagonal_absorb_layer!(env,state.sites,state.lattice,t,method)
    end
    value=log(_positive_real(env.scalar,"norm"))+env.log_scale
    merge((;log_norm=value+2state.log_amplitude,tensor_log_norm=value,
        max_environment_truncation=env.max_truncation),_variational_result(method,env.variational))
end

_variational_result(::ContractionMethod,records)=(;)
_diagonal_entropy(::AbstractDiagonalBoundaryMPS,l;kwargs...)=_diagonal_entropy(l;kwargs...)

"""One diagonal layer or two adjacent layers between cached boundary MPSs."""
struct DiagonalStrip
    lattice::Lattice
    sites::Vector{CartesianIndex{2}}
    ports::Vector{NTuple{4,Int}}
    attachments::Vector{Vector{_DiagonalTensor}}
    left::_DiagonalTensor
    right::_DiagonalTensor
    max_truncation::Float64
end

function _diagonal_strip(state,firstlayer,lastlayer,northwest,southeast)
    l=state.lattice
    sites=sort!([i for t in firstlayer:lastlayer for i in _diagonal_sites(l,t)];by=i->i[1]-i[2])
    ports=[_diagonal_ports(l,i) for i in sites]
    attachments=[_DiagonalTensor[] for _ in sites]
    left=_DiagonalTensor(1.0,Int[]);right=_DiagonalTensor(1.0,Int[])
    base=_diagonal_edge_count(l)+1
    for boundary in (northwest,southeast)
        if !isempty(boundary.tensors)
            first=boundary.tensors[1];last=boundary.tensors[end]
            left=_diagonal_join(left,_diagonal_unit(dual(space(first,1)),base;T=scalartype(first)))
            right=_diagonal_join(right,_diagonal_unit(dual(space(last,3)),base+length(boundary.tensors);T=scalartype(last)))
            for (k,(A,edge)) in enumerate(zip(boundary.tensors,boundary.edges))
                site=findfirst(p->edge in p,ports)
                site===nothing && error("boundary edge and diagonal strip are disconnected")
                push!(attachments[site],_DiagonalTensor(A,[base+k-1,edge,base+k]))
            end
            base+=length(boundary.tensors)+1
        end
    end
    for (k,site) in enumerate(sites)
        x,y=Tuple(site);A=state[site]
        inside=(x>1,y>1,x<l.Lx,y<l.Ly)
        for (j,axis) in enumerate(_VIRTUAL_AXES)
            inside[j] && continue
            push!(attachments[k],_diagonal_unit(dual(_double_space(A,A,axis)),ports[k][j];T=scalartype(A)))
        end
    end
    DiagonalStrip(l,sites,ports,attachments,left,right,max(northwest.max_truncation,southeast.max_truncation))
end

# Select an attached boundary tensor by connectivity.
# A disconnected attachment is a unit OBC vector.
function _diagonal_attach(message,pending)
    scores=[length(intersect(message.labels,A.labels)) for A in pending]
    index=argmax(scores)
    if scores[index]==0
        index=something(findfirst(A->length(A.labels)==1,pending),0)
        index>0 || error("disconnected diagonal strip attachment")
    end
    attachment=splice!(pending,index)
    _diagonal_join(message,attachment)
end
function _diagonal_step(message,strip,k,A,B=A;physical=nothing)
    pending=copy(strip.attachments[k]);ports=strip.ports[k]
    while count(p->p in message.labels,ports)<2
        isempty(pending) && error("missing diagonal strip inputs")
        message=_diagonal_attach(message,pending)
    end
    inputs=Int[p for p in ports if p in message.labels]
    length(inputs)==2 || error("diagonal strip step has more than two inputs")
    message=_diagonal_apply(message,A,B,ports,inputs;physical)
    while !isempty(pending)
        message=_diagonal_attach(message,pending)
    end
    message
end
function _diagonal_messages(state,strip;side=:both)
    side in (:both,:left,:right) || throw(ArgumentError("invalid strip message side"))
    n=length(strip.sites)
    left=Vector{_DiagonalTensor}(undef,n+1);right=similar(left)
    left[1]=strip.left;right[end]=strip.right
    if side!=:right
        for k in 1:n
            left[k+1]=_diagonal_scale(_diagonal_step(left[k],strip,k,state[strip.sites[k]]))
        end
    end
    if side!=:left
        for k in n:-1:1
            right[k]=_diagonal_scale(_diagonal_step(right[k+1],strip,k,state[strip.sites[k]]))
        end
    end
    (;left,right)
end
_diagonal_value(left,right)=_diagonal_scalar(_diagonal_join(left,right))
function _diagonal_site_value(left,right,strip,k,A,B=A)
    _diagonal_value(_diagonal_step(left,strip,k,A,B),right)
end

"""Construct a two-site environment from two bounded halves."""
function _diagonal_hole(left,right,strip,positions,output)
    pending=reduce(vcat,(strip.attachments[k] for k in positions);init=_DiagonalTensor[])
    while !isempty(pending)
        ls=[length(intersect(left.labels,A.labels)) for A in pending]
        rs=[length(intersect(right.labels,A.labels)) for A in pending]
        li=argmax(ls);ri=argmax(rs)
        useleft=ls[li]>rs[ri] || (ls[li]==rs[ri] && length(left.labels)<=length(right.labels))
        index=useleft ? li : ri
        A=splice!(pending,index)
        if useleft
            left=_diagonal_join(left,A)
        else
            right=_diagonal_join(right,A)
        end
    end
    _diagonal_reorder(_diagonal_join(left,right),output).data
end

function _diagonal_site_hole(left,right,strip,k,A)
    H=_diagonal_hole(left,right,strip,k:k,collect(strip.ports[k]))
    for (i,axis) in reverse(collect(enumerate(_VIRTUAL_AXES)))
        V=space(A,axis);H=_split(H,i,(dual(V),V))
    end
    _linearmap_metric(permute(H,((2,4,6,8),(1,3,5,7))))
end
