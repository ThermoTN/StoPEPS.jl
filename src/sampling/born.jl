@inline function _splitmix64(value::UInt64)
    value+=0x9e3779b97f4a7c15
    value=(value ⊻ (value>>30))*0xbf58476d1ce4e5b9
    value=(value ⊻ (value>>27))*0x94d049bb133111eb
    (value ⊻ (value>>31))
end
const SAMPLING_RNG_SCHEME_VERSION = 2
"""
    sample_rng(seed, index)

Construct the reproducible random stream associated with a seed and sample index.
"""
function sample_rng(seed::Integer,index::Integer)
    seed>=0 && index>=1 || throw(ArgumentError("invalid random stream identity"))
    # Encode seed and sample index as separate mixed words with a version header.
    words=UInt32[0x424f524e,SAMPLING_RNG_SCHEME_VERSION]
    for value in (seed,index)
        mixed=_splitmix64(UInt64(value))
        push!(words,UInt32(mixed & 0xffffffff),UInt32(mixed>>32))
    end
    MersenneTwister(words)
end
function _event_rng(seed,domain::UInt64,event,item)
    mixed=_splitmix64(UInt64(seed) ⊻ domain)
    MersenneTwister(_splitmix64(_splitmix64(mixed ⊻ UInt64(event)) ⊻ UInt64(item)))
end
function project_configuration(checkpoint::FinitePEPO,configuration;basis=:auto)
    size(configuration)==size(checkpoint.sites) || throw(DimensionMismatch("configuration lattice"))
    basis=_resolve_basis(basis,space(checkpoint[1,1],3),checkpoint.lattice)
    sites=map(checkpoint.sites,configuration,basis.sites) do A,label,B
        _project(A,B,_basis_index(B,label),:auxiliary)
    end
    state=FinitePEPS(checkpoint.lattice,sites;beta=checkpoint.beta,log_amplitude=checkpoint.log_amplitude)
    records=normalize_state!(state)
    return state,records
end

"""
    SamplingCache(state, contraction; kind=:auxiliary, basis=:auto)

Boundary environments for Born draws and configuration probabilities.
`kind=:auxiliary` samples the thermal PEPO's auxiliary indices;
`kind=:physical` samples physical indices for a resampling proposal.
"""
struct SamplingCache{M<:AbstractRowBoundaryMPS,B}
    state::FiniteState
    layers::Matrix
    candidates::Matrix
    south::Vector{B}
    method::M
    kind::Symbol
    basis::ResolvedBasis
    target_charge::Union{Nothing,Sector}
    reachable::Vector
end
function SamplingCache(state::FiniteState,method::AbstractRowBoundaryMPS;kind=:auxiliary,basis=:auto)
    kind in (:auxiliary,:physical) || throw(ArgumentError("unknown sampling kind"))
    kind==:physical && !(state isa FinitePEPS) && throw(ArgumentError("proposal requires PEPS"))
    basis=_resolve_basis(basis,space(state[1,1],3),state.lattice)
    layers=_row_layers(state,method)
    candidates=map(state.sites,basis.sites) do A,B
        [_row_site(method,_project(A,B,k,kind)) for k in eachindex(B.labels)]
    end
    Ly=size(layers,2);terminal=_trivial_boundary(collect(layers[:,end]),:south)
    south=Vector{typeof(terminal)}(undef,Ly+1)
    south[end]=terminal
    for y in Ly:-1:1
        south[y]=_absorb(south[y+1],collect(layers[:,y]),method;side=:south,row_cut=y-1)
    end
    charged=kind==:physical && sectortype(basis.space)!==Trivial
    SamplingCache(state,layers,candidates,south,method,kind,basis,
        charged ? _state_charge(state) : nothing,charged ? _charge_suffix(basis) : [])
end

function _site_hole(L,T,B,R,A)
    @tensor X[w,b,n,c] := L[a,w,b]*T[a,n,c]
    @tensor Y[b,s,c,e] := B[b,s,d]*R[c,e,d]
    @tensor H[w,n,e,s] := X[w,b,n,c]*Y[b,s,c,e]
    for (k,axis) in reverse(collect(enumerate((1,2,5,6))))
        V=space(A,axis); H=_split(H,k,(dual(V),V))
    end
    _linearmap_metric(permute(H,((2,4,6,8),(1,3,5,7))))
end

"""Draw and evaluate configurations using row-major finite-chi conditional probabilities."""
function sample_configuration(cache::SamplingCache;rng=nothing,configuration=nothing)
    Lx,Ly=size(cache.layers)
    configuration===nothing && rng===nothing && throw(ArgumentError("an explicit RNG is required"))
    if configuration!==nothing
        size(configuration)==(Lx,Ly) || throw(DimensionMismatch("configuration lattice"))
        foreach((B,label)->_basis_index(B,label),cache.basis.sites,configuration)
    end
    selected=configuration===nothing ? Matrix{Int}(undef,Lx,Ly) : Int.(configuration)
    north=_trivial_boundary(collect(cache.layers[:,1]),:north)
    log_probability=0.0; conditionals=Float64[]; repairs=NamedTuple[]
    max_environment_truncation=maximum(e.max_truncation for e in cache.south)
    result(;log_probability)=merge((;configuration=selected,log_probability,conditionals,max_environment_truncation,repairs),
        _variational_result(cache.method,vcat(_row_variational_records(cache.south[1]),_row_variational_records(north))))
    prefix=one(typeof(first(cache.basis.sites[1,1].charges)))
    if configuration!==nothing && cache.target_charge!==nothing
        charge=foldl(_fuse_charge,(B.charges[_basis_index(B,label)]
            for (B,label) in zip(vec(cache.basis.sites),vec(configuration)));init=prefix)
        charge==cache.target_charge || return result(;log_probability=-Inf)
    end
    for y in 1:Ly
        # A real checkpoint can acquire complex layers when sampled in a
        # complex basis. The mutable row must admit the promoted TensorMaps.
        row=_row_working_copy(collect(cache.layers[:,y]))
        top=north.tensors; bottom=cache.south[y+1].tensors
        e=row_environments(top,row,bottom;side=:right); left=e.left[1]
        for x in 1:Lx
            B=cache.basis.sites[x,y]; candidates=cache.candidates[x,y]
            position=x+(y-1)*Lx
            allowed=[cache.target_charge===nothing || any(r->_fuse_charge(_fuse_charge(prefix,c),r)==cache.target_charge,
                cache.reachable[position+1]) for c in B.charges]
            any(allowed) || error("sampling prefix violates the total symmetry sector")
            if cache.kind==:physical
                A=cache.state[x,y]
                metric=_site_hole(left,top[x],bottom[x],e.right[x+1],A)
                root,diagnostic=_positive_metric(metric;square_root=true)
                push!(repairs,merge(diagnostic,(;site=(x,y))))
                values=map(eachindex(B.labels)) do k
                    allowed[k] || return 0.0
                    projected=_project(A,B,k,:physical)
                    vector=permute(projected,((1,2,5,6),(3,4)))
                    abs2(norm(root*vector))
                end
                total=sum(values)
                if iszero(total)
                    probabilities=Float64.(allowed)./count(allowed)
                    push!(repairs,(;site=(x,y),reason=:zero_proposal_weights,allowed_outcomes=count(allowed)))
                else
                    probabilities=values./total
                end
            else
                raw=[_row_scalar(left,top[x],E,bottom[x],e.right[x+1]) for E in candidates]
                scale=maximum(abs,raw;init=0.0)
                all(z->isfinite(z) && abs(imag(z))<=1e-9*max(scale,floatmin(Float64)) &&
                    real(z)>=-64eps(Float64)*scale,raw) ||
                    error("invalid Born conditional at ($x,$y); increase chi_sampling")
                values=max.(real.(raw),0.0)
                total=sum(values); total>0 || error("Born conditional has zero total mass at ($x,$y)")
                probabilities=values./total
            end
            k=configuration===nothing ? _categorical(log.(probabilities),rng) : _basis_index(B,selected[x,y])
            selected[x,y]=B.labels[k]
            p=probabilities[k]; push!(conditionals,p)
            if iszero(p)
                return result(;log_probability=-Inf)
            end
            log_probability+=log(p); row[x]=candidates[k]
            prefix=_fuse_charge(prefix,B.charges[k])
            if x<Lx
                left=_safe_scale(_left_step(left,top[x],row[x],bottom[x]))
            end
        end
        if cache.kind==:auxiliary || y<Ly
            north=_absorb(north,row,cache.method;row_cut=y)
            max_environment_truncation=max(max_environment_truncation,north.max_truncation)
        end
    end
    return result(;log_probability)
end
