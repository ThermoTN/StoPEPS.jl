struct DiagonalSamplingCache{M<:AbstractDiagonalBoundaryMPS}
    state::FiniteState
    southeast::Vector{DiagonalBoundary}
    method::M
    kind::Symbol
    basis::ResolvedBasis
    target_charge::Union{Nothing,Sector}
    reachable::Vector
    order::Vector{CartesianIndex{2}}
end
function SamplingCache(state::FiniteState,method::AbstractDiagonalBoundaryMPS;kind=:auxiliary,basis=:auto)
    kind in (:auxiliary,:physical) || throw(ArgumentError("unknown sampling kind"))
    kind==:physical && !(state isa FinitePEPS) && throw(ArgumentError("proposal requires PEPS"))
    basis=_resolve_basis(basis,space(state[1,1],3),state.lattice)
    env=_diagonal_boundaries(state,method;side=:southeast)
    charged=kind==:physical && sectortype(basis.space)!==Trivial
    order=_diagonal_order(state.lattice)
    DiagonalSamplingCache(state,env.southeast,method,kind,basis,
        charged ? _state_charge(state) : nothing,charged ? _charge_suffix(basis,order) : [],order)
end

"""Draw and evaluate configurations using diagonal finite-chi conditional probabilities."""
function sample_configuration(cache::DiagonalSamplingCache;rng=nothing,configuration=nothing)
    state=cache.state;l=state.lattice;last=l.Lx+l.Ly
    configuration===nothing && rng===nothing && throw(ArgumentError("an explicit RNG is required"))
    if configuration!==nothing
        size(configuration)==size(state.sites) || throw(DimensionMismatch("configuration lattice"))
        foreach((B,label)->_basis_index(B,label),cache.basis.sites,configuration)
    end
    selected=configuration===nothing ? Matrix{Int}(undef,l.Lx,l.Ly) : Int.(configuration)
    northwest=DiagonalBoundary();projected=Matrix{AbstractTensorMap}(undef,l.Lx,l.Ly)
    log_probability=0.0;conditionals=Float64[];repairs=NamedTuple[]
    max_environment_truncation=cache.southeast[2].max_truncation
    result(;log_probability)=merge((;configuration=selected,log_probability,
        conditionals,max_environment_truncation,repairs),
        _variational_result(cache.method,vcat(cache.southeast[2].variational,northwest.variational)))
    prefix=one(typeof(first(cache.basis.sites[1,1].charges)))
    if configuration!==nothing && cache.target_charge!==nothing
        charge=foldl(_fuse_charge,(B.charges[_basis_index(B,label)]
            for (B,label) in zip(vec(cache.basis.sites),vec(configuration)));init=prefix)
        charge==cache.target_charge || return result(;log_probability=-Inf)
    end
    position=0
    for t in 2:last
        strip=_diagonal_strip(state,t,t,northwest,cache.southeast[t+1])
        rightward=iseven(t)
        edges=_diagonal_messages(state,strip;side=rightward ? :right : :left)
        message=rightward ? strip.left : strip.right
        for k in (rightward ? eachindex(strip.sites) : reverse(eachindex(strip.sites)))
            site=strip.sites[k];position+=1
            A=state[site];B=cache.basis.sites[site]
            left=rightward ? message : edges.left[k]
            right=rightward ? edges.right[k+1] : message
            allowed=[cache.target_charge===nothing || any(r->_fuse_charge(_fuse_charge(prefix,c),r)==cache.target_charge,
                cache.reachable[position+1]) for c in B.charges]
            any(allowed) || error("sampling prefix violates the total symmetry sector")
            candidates=[_project(A,B,j,cache.kind) for j in eachindex(B.labels)]
            if cache.kind==:physical
                metric=_diagonal_site_hole(left,right,strip,k,A)
                root,diagnostic=_positive_metric(metric;square_root=true)
                push!(repairs,merge(diagnostic,(;site=Tuple(site))))
                values=map(eachindex(B.labels)) do j
                    allowed[j] || return 0.0
                    abs2(norm(root*permute(candidates[j],((1,2,5,6),(3,4)))))
                end
                total=sum(values)
                if iszero(total)
                    probabilities=Float64.(allowed)./count(allowed)
                    push!(repairs,(;site=Tuple(site),reason=:zero_proposal_weights,allowed_outcomes=count(allowed)))
                else
                    probabilities=values./total
                end
            else
                raw=[_diagonal_site_value(left,right,strip,k,C) for C in candidates]
                scale=maximum(abs,raw;init=0.0)
                all(z->isfinite(z) && abs(imag(z))<=1e-9*max(scale,floatmin(Float64)) &&
                    real(z)>=-64eps(Float64)*scale,raw) ||
                    error("invalid Born conditional at $(Tuple(site)); increase chi_sampling")
                values=max.(real.(raw),0.0);total=sum(values)
                total>0 || error("Born conditional has zero total mass at $(Tuple(site))")
                probabilities=values./total
            end
            j=configuration===nothing ? _categorical(log.(probabilities),rng) : _basis_index(B,selected[site])
            selected[site]=B.labels[j];p=probabilities[j];push!(conditionals,p)
            iszero(p) && return result(;log_probability=-Inf)
            log_probability+=log(p);projected[site]=candidates[j]
            prefix=_fuse_charge(prefix,B.charges[j])
            message=_diagonal_scale(_diagonal_step(message,strip,k,candidates[j]))
        end
        # Sample the layer with a fixed strip boundary, then compress its frontier.
        if cache.kind==:auxiliary || t<last
            _diagonal_absorb_layer!(northwest,projected,l,t,cache.method)
            max_environment_truncation=max(max_environment_truncation,northwest.max_truncation)
        end
    end
    result(;log_probability)
end
