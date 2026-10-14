function _diagonal_pair_value(state,strip,edges,i,j,A,B)
    a=findfirst(==(i),strip.sites);b=findfirst(==(j),strip.sites)
    abs(a-b)==1 || error("nonadjacent diagonal observable")
    message=edges.left[min(a,b)]
    for k in min(a,b):max(a,b)
        site=strip.sites[k]
        message=_diagonal_step(message,strip,k,site==i ? A : B,state[site])
    end
    _diagonal_value(message,edges.right[max(a,b)+1])
end

"""Cached diagonal strips with factorized ket/bra observable insertions."""
function measure_energy(state::FiniteState,model::AbstractModel,method::AbstractDiagonalBoundaryMPS;record_entropy=false)
    l=state.lattice;last=l.Lx+l.Ly
    entropy=record_entropy ? _diagonal_entropy(method,l;profile=true) : nothing
    env=_diagonal_boundaries(state,method;entropy)
    terms=_energy_terms(model,scalartype(state[1,1]));factors=_operator_factors(terms.pair)
    onsite=0.0;pair=0.0
    for t in 2:last
        strip=_diagonal_strip(state,t,t,env.northwest[t],env.southeast[t+1])
        edges=_diagonal_messages(state,strip)
        for (k,i) in enumerate(strip.sites)
            A=state[i]
            den=_diagonal_site_value(edges.left[k],edges.right[k+1],strip,k,A)
            num=_diagonal_site_value(edges.left[k],edges.right[k+1],strip,k,_act(A,terms.onsite,3),A)
            onsite+=_real_ratio(num,den)
        end
        t==last && continue
        strip=_diagonal_strip(state,t,t+1,env.northwest[t],env.southeast[t+2])
        edges=_diagonal_messages(state,strip)
        for direction in (:horizontal,:vertical), (i,j) in _diagonal_layer_bonds(l,t,direction)
            C,D=_operator_pair(state[i],state[j],factors,direction)
            num=_diagonal_pair_value(state,strip,edges,i,j,C,D)
            den=_diagonal_pair_value(state,strip,edges,i,j,state[i],state[j])
            pair+=_real_ratio(num,den)
        end
    end
    final=env.northwest[last+1]
    log_norm=log(_positive_real(final.scalar,"norm"))+final.log_scale
    energy_total=terms.onsite_scale*onsite+terms.pair_scale*pair
    error=max(final.max_truncation,env.southeast[2].max_truncation)
    merge((;energy_total,energy_per_site=energy_total/length(l),log_norm=log_norm+2state.log_amplitude,
        tensor_log_norm=log_norm,max_environment_truncation=error,Se=entropy===nothing ? nothing : entropy.Se),
        _variational_result(method,vcat(final.variational,env.southeast[2].variational)))
end
