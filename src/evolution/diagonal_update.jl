struct DiagonalBondEnvironment
    strip::DiagonalStrip
    left::_DiagonalTensor
    right::_DiagonalTensor
    positions::UnitRange{Int}
    first::CartesianIndex{2}
    second::CartesianIndex{2}
    direction::Symbol
    outer_spaces::Tuple
end

_diagonal_external_axes(direction)=direction==:horizontal ? ((1,2,6),(2,5,6)) : ((1,2,5),(1,5,6))
function _diagonal_bond_environment(state,strip,left,right,i,j,direction)
    a=findfirst(==(i),strip.sites);b=findfirst(==(j),strip.sites)
    abs(a-b)==1 || error("bond sites must be adjacent in a diagonal strip")
    axes=_diagonal_external_axes(direction)
    spaces=(map(k->space(state[i],k),axes[1])...,map(k->space(state[j],k),axes[2])...)
    DiagonalBondEnvironment(strip,left,right,min(a,b):max(a,b),i,j,direction,spaces)
end
function bond_hole(env::DiagonalBondEnvironment)
    axes=_diagonal_external_axes(env.direction)
    ports=map(site->_diagonal_ports(env.strip.lattice,site),(env.first,env.second))
    output=[ports[k][findfirst(==(axis),_VIRTUAL_AXES)] for k in 1:2 for axis in axes[k]]
    H=_diagonal_hole(env.left,env.right,env.strip,env.positions,output)
    for k in 6:-1:1
        V=env.outer_spaces[k];H=_split(H,k,(dual(V),V))
    end
    H
end

# Absorb each reduced Q as a site with an open core leg and a unit bond leg.
function _diagonal_core_site(Q,direction,first)
    axes=_diagonal_external_axes(direction)[first ? 1 : 2]
    missing=only(setdiff(_VIRTUAL_AXES,axes))
    V=oneunit(space(Q,1));missing in (1,2) && (V=dual(V))
    tensor=_diagonal_join(_DiagonalTensor(Q,[axes...,4,3]),_diagonal_unit(V,missing;T=scalartype(Q)))
    _diagonal_reorder(tensor,1:6).data
end
function _reduced_metric(env::DiagonalBondEnvironment,Qa,Qb)
    A=_diagonal_core_site(Qa,env.direction,true)
    B=_diagonal_core_site(Qb,env.direction,false)
    # Close each Q against its own side before joining the open core pairs.
    function absorb(message,k)
        isfirst=env.strip.sites[k]==env.first
        _diagonal_step(message,env.strip,k,isfirst ? A : B;
            physical=isfirst ? (-11,-12) : (-13,-14))
    end
    left=absorb(env.left,first(env.positions))
    right=absorb(env.right,last(env.positions))
    result=_diagonal_reorder(_diagonal_join(left,right),[-12,-14,-11,-13])
    permute(result.data,((1,2),(3,4)))
end

function _diagonal_layer_bonds(l,t,direction;reverse=false)
    shift=direction==:horizontal ? CartesianIndex(1,0) : CartesianIndex(0,1)
    [(i,i+shift) for i in _diagonal_sites(l,t;reverse)
        if i[1]+shift[1]<=l.Lx && i[2]+shift[2]<=l.Ly]
end

function _sweep!(state,model,delta,D,method,contraction::AbstractDiagonalBoundaryMPS,direction,reverse;
        sector_space=nothing,entropy=nothing,variational=nothing)
    l=state.lattice;last=l.Lx+l.Ly
    env=_diagonal_boundaries(state,contraction;side=reverse ? :northwest : :southeast,entropy)
    fixed=reverse ? env.northwest[3:last+1] : env.southeast[2:last]
    environment_error=maximum(e.max_truncation for e in fixed)
    rolling=DiagonalBoundary();records=NamedTuple[]
    for t in (reverse ? (last-1:-1:2) : (2:last-1))
        nw=reverse ? env.northwest[t] : rolling
        se=reverse ? rolling : env.southeast[t+2]
        strip=_diagonal_strip(state,t,t+1,nw,se)
        order=_diagonal_sites(l,t;reverse)
        rightward=length(order)<2 ? !reverse : first(order)[1]<Base.last(order)[1]
        edges=_diagonal_messages(state,strip;side=rightward ? :right : :left)
        message=rightward ? strip.left : strip.right
        cursor=rightward ? 1 : length(strip.sites)
        for (i,j) in _diagonal_layer_bonds(l,t,direction;reverse)
            positions=sort([findfirst(==(i),strip.sites),findfirst(==(j),strip.sites)])
            a,b=positions
            while rightward ? cursor<a : cursor>b
                message=_diagonal_scale(_diagonal_step(message,strip,cursor,state[strip.sites[cursor]]))
                cursor+=rightward ? 1 : -1
            end
            left=rightward ? message : edges.left[a]
            right=rightward ? edges.right[b+1] : message
            hole=_diagonal_bond_environment(state,strip,left,right,i,j,direction)
            gate=trotter_gate(model,l,i,j,delta;T=scalartype(state[i]))
            push!(records,update_bond!(state,i,j,gate,D,method,hole;direction,
                boundary_error=strip.max_truncation,sector_space))
            for k in (rightward ? (a:b) : (b:-1:a))
                message=_diagonal_scale(_diagonal_step(message,strip,k,state[strip.sites[k]]))
            end
            cursor=rightward ? b+1 : a-1
        end
        _diagonal_absorb_layer!(rolling,state.sites,l,reverse ? t+1 : t,contraction;reverse,entropy)
        environment_error=max(environment_error,rolling.max_truncation)
    end
    if variational!==nothing
        append!(variational,reverse ? env.northwest[end].variational : env.southeast[2].variational)
        append!(variational,rolling.variational)
    end
    records,environment_error
end
