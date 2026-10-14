"""
    BoundaryMPSRowVariational(chi; sector_space=nothing, max_sweep=4, threshold_sweep=1e-8)

Two-site variational contraction of a row with separate ket and bra factors.
A sweep traverses columns 1 to Lx and back. From the second sweep,
`threshold_sweep` compares successive squared norms from retained spectra;
a zero threshold runs the full sweep budget. The environment diagnostic is
the maximum local projected SVD discarded weight.
"""
struct BoundaryMPSRowVariational{S} <: AbstractRowBoundaryMPS
    chi::Int
    sector_space::S
    max_sweep::Int
    threshold_sweep::Float64
    function BoundaryMPSRowVariational(chi::Int,sector_space;
            max_sweep::Int=4,threshold_sweep::Real=1e-8)
        chi>=1 || throw(ArgumentError("chi must be positive"))
        max_sweep>=1 || throw(ArgumentError("max_sweep must be positive"))
        isfinite(threshold_sweep) && threshold_sweep>=0 ||
            throw(ArgumentError("threshold_sweep must be finite and nonnegative"))
        new{typeof(sector_space)}(chi,sector_space,max_sweep,Float64(threshold_sweep))
    end
end
BoundaryMPSRowVariational(chi::Int;sector_space=nothing,kwargs...)=
    BoundaryMPSRowVariational(chi,sector_space;kwargs...)
BoundaryMPSRowVariational(;chi=32,kwargs...)=BoundaryMPSRowVariational(chi;kwargs...)

_is_variational(::Type{<:ContractionMethod})=false
_is_variational(::Type{<:Union{BoundaryMPSDiagonalVariational,BoundaryMPSRowVariational}})=true
_variational_result(::BoundaryMPSRowVariational,records)=_variational_summary(records)

function _row_site(::BoundaryMPSRowVariational,A,B=A;physical=nothing,auxiliary=nothing)
    physical===nothing || (A=_act(A,physical,3))
    auxiliary===nothing || (A=_act(A,auxiliary,4))
    FactorizedRowSite(A,B)
end

mutable struct RowVariationalBoundary
    tensors::Vector{AbstractTensorMap}
    center::Int
    log_scale::Float64
    max_truncation::Float64
    variational::Vector{NamedTuple}
end
Base.copy(B::RowVariationalBoundary)=RowVariationalBoundary(copy(B.tensors),B.center,
    B.log_scale,B.max_truncation,copy(B.variational))
_row_variational_records(B::RowVariationalBoundary)=B.variational
function _trivial_boundary(row::AbstractVector{FactorizedRowSite},side)
    axis=side==:north ? 2 : 4
    V=oneunit(space(first(row),axis))
    B=AbstractTensorMap[_unit_tensor((V',dual(space(A,axis)),V);T=scalartype(A)) for A in row]
    RowVariationalBoundary(B,1,0.0,0.0,NamedTuple[])
end

"""Fixed input and single-layer factors with a logarithmic target scale."""
mutable struct _RowLayerTarget
    input::RowVariationalBoundary
    row::Vector{FactorizedRowSite}
    side::Symbol
    log_scale::Float64
    reference::Float64
    zero::Bool
end
function _row_layer_target(input,row;side=:north)
    side in (:north,:south) || throw(ArgumentError("invalid row absorption side"))
    length(input.tensors)==length(row) || throw(DimensionMismatch("row boundary length"))
    scale=input.log_scale;zero_target=scale==-Inf
    factors=FactorizedRowSite[]
    for E in row
        a=norm(E.ket);b=norm(E.bra)
        isfinite(a) && isfinite(b) || throw(_VariationalNumericalError(:nonfinite_target))
        zero_target|=iszero(a)||iszero(b)
        push!(factors,FactorizedRowSite(iszero(a) ? E.ket : E.ket/a,iszero(b) ? E.bra : E.bra/b))
        scale+=log(a)+log(b)
    end
    _RowLayerTarget(copy(input),factors,side,scale,NaN,zero_target)
end
_row_output_axis(target)=target.side==:north ? 4 : 2
_row_input_label(t,i)=4(length(t.row)+1)+i
_row_output_label(t,i)=3(length(t.row)+1)+i
_row_horizontal_label(t,i)=2(length(t.row)+1)+i
_row_candidate_label(t,i)=length(t.row)+1+i
function _row_target_ports(t,i)
    incoming=_row_input_label(t,i);outgoing=_row_output_label(t,i)
    (_row_horizontal_label(t,i),t.side==:north ? incoming : outgoing,
        _row_horizontal_label(t,i+1),t.side==:north ? outgoing : incoming)
end

function _row_product(target)
    B=AbstractTensorMap[]
    for E in target.row
        v=_variational_identity(space(E.ket,_VIRTUAL_AXES[_row_output_axis(target)]),scalartype(E))
        V=oneunit(space(v,1));T=scalartype(v)
        A=_diagonal_join(_diagonal_unit(V',1;T),_DiagonalTensor(v,[2]))
        A=_diagonal_join(A,_diagonal_unit(V,3;T))
        push!(B,_diagonal_reorder(A,[1,2,3]).data)
    end
    RowVariationalBoundary(B,1,0.0,0.0,NamedTuple[])
end
function _row_warm_start(target)
    C=copy(target.input);C.log_scale=0.0;C.max_truncation=0.0;empty!(C.variational)
    for i in eachindex(C.tensors)
        P=space(C.tensors[i],2);Q=space(target.row[i],_row_output_axis(target))
        P==Q && continue
        if isdual(P)!=isdual(Q)
            C.tensors[i]=flip(C.tensors[i],(2,));P=space(C.tensors[i],2)
        end
        infimum(P,Q)==P || _diagonal_center!(C,i)
        C.tensors[i]=_act(C.tensors[i],_variational_space_map(P,Q,scalartype(C.tensors[i])),2)
    end
    _diagonal_center!(C,1)
    s=norm(C.tensors[1])
    isfinite(s) || throw(_VariationalNumericalError(:nonfinite_initialization))
    iszero(s) && return _row_product(target)
    C.tensors[1]=C.tensors[1]/s
    C
end

function _row_mixed_edge(target,C,left)
    n=length(target.row);i=left ? 1 : n;cut=left ? 1 : n+1;a=left ? 1 : 3
    E=target.row[i];T=scalartype(E)
    X=_diagonal_join(_diagonal_unit(dual(space(target.input.tensors[i],a)),cut;T),
        _diagonal_unit(dual(space(E,a)),_row_horizontal_label(target,cut);T))
    X=_diagonal_join(X,_diagonal_unit(space(C.tensors[i],a),_row_candidate_label(target,cut);T))
    _variational_message(X)
end

# A half projection has (candidate, output, input boundary, horizontal ket/bra)
# legs. Each half consumes its own ket and bra before the halves are joined.
function _row_half_projection(message,target,i,left)
    X=_diagonal_join(message.tensor,
        _DiagonalTensor(target.input.tensors[i],[i,_row_input_label(target,i),i+1]))
    ports=_row_target_ports(target,i)
    X=_row_apply(X,target.row[i],ports,
        (_row_horizontal_label(target,left ? i : i+1),_row_input_label(target,i)))
    _variational_message(X,message.log_scale)
end
function _row_mixed_step(message,target,i,A,left)
    half=_row_half_projection(message,target,i,left)
    bra=_conj(_twistdual(A,(2,)))
    X=_diagonal_join(half.tensor,_DiagonalTensor(bra,
        [_row_candidate_label(target,i),_row_output_label(target,i),_row_candidate_label(target,i+1)]))
    _variational_message(X,half.log_scale)
end
function _row_mixed_messages(target,C,left)
    n=length(C.tensors);messages=Vector{_VariationalMessage}(undef,n+1)
    messages[left ? 1 : n+1]=_row_mixed_edge(target,C,left)
    for i in (left ? (1:n) : (n:-1:1))
        messages[left ? i+1 : i]=_row_mixed_step(messages[left ? i : i+1],target,i,C.tensors[i],left)
    end
    messages
end
function _row_project_pair(target,i,left,right)
    L=_row_half_projection(left,target,i,true)
    R=_row_half_projection(right,target,i+1,false)
    X=_diagonal_reorder(_diagonal_join(L.tensor,R.tensor),
        [_row_candidate_label(target,i),_row_output_label(target,i),
            _row_output_label(target,i+1),_row_candidate_label(target,i+2)])
    theta=_twistdual(X.data,(1,4));s=norm(theta)
    isfinite(s) || throw(_VariationalNumericalError(:nonfinite_projection))
    iszero(s) && throw(_VariationalNumericalError(:zero_local_projection))
    lognorm=L.log_scale+R.log_scale+log(s)
    isfinite(lognorm) || throw(_VariationalNumericalError(:zero_local_projection))
    theta/s,lognorm
end

# Candidate tensors store a normalized canonical shape. Its mathematical
# amplitude stays in C.log_scale throughout the fit.
# Thus local SVDs and comparison remain safe even for exponentially small F.
function _row_initialize_amplitude!(target,C,fixed)
    terminal=fixed[1];edge=_row_mixed_edge(target,C,true)
    z=_diagonal_scalar(_diagonal_join(terminal.tensor,edge.tensor))
    isfinite(z) || throw(_VariationalNumericalError(:nonfinite_overlap))
    if iszero(z) || terminal.log_scale==-Inf
        C.log_scale=-Inf
        return false
    end
    target.reference=log(abs(z))+terminal.log_scale+edge.log_scale
    C.tensors[1]=C.tensors[1]*(z/abs(z))
    C.log_scale=0.0
    true
end

function _row_candidate_allowed(C,method)
    for A in @view C.tensors[1:end-1]
        V=dual(space(A,3)) # The SVD's retained map space.
        isdual(V) && (V=flip(V))
        dim(V)<=method.chi || return false
        method.sector_space===nothing || infimum(V,method.sector_space)==V || return false
    end
    true
end

function _row_half_sweep!(C,target,method,right,entropy,profile,fixed;work=nothing)
    n=length(C.tensors);rolling=_row_mixed_edge(target,C,right)
    bounded=_row_candidate_allowed(C,method)
    nextfixed=Vector{_VariationalMessage}(undef,n+1)
    nextfixed[right ? 1 : n+1]=rolling
    for i in (right ? (1:n-1) : (n-1:-1:1))
        L=right ? rolling : fixed[i];R=right ? fixed[i+2] : rolling
        theta,logtheta=_row_project_pair(target,i,L,R)
        isnan(target.reference) && (target.reference=logtheta)
        M=permute(theta,((1,2),(3,4)))
        work===nothing || push!(work,(;right,cut=i,rows=dim(codomain(M)),columns=dim(domain(M))))
        observed=NaN
        observer=entropy===nothing ? nothing : (cut,value)->begin
            entropy.maximum=max(entropy.maximum,value);observed=value
        end
        U,S,V,err=_boundary_svd(M,_truncation(method.chi,method.sector_space),observer,i)
        s=norm(S)
        isfinite(s) && s>0 || throw(_VariationalNumericalError(:zero_local_projection))
        newscale=logtheta-target.reference+log(s)
        # g(C)=-||C||^2 after projection (also for the amplitude-fitted seed).
        # Compare in logarithms; a material norm decrease increases the cost.
        !bounded || newscale>=C.log_scale-512eps(Float64)*max(1.0,abs(C.log_scale)) ||
            throw(_VariationalNumericalError(:objective_increase))
        kept=S/s
        C.tensors[i]=_out(right ? U : U*kept)
        C.tensors[i+1]=_out(_twistdual(right ? kept*V : V,(1,)))
        C.center=right ? i+1 : i;C.log_scale=newscale
        profile[i]=observed
        C.max_truncation=max(C.max_truncation,abs2(err)) # norm(M)==1
        k=right ? i : i+1
        rolling=_row_mixed_step(rolling,target,k,C.tensors[k],right)
        nextfixed[right ? k+1 : k]=rolling
    end
    nextfixed
end

function _row_relative_change(a,b)
    largest=max(a,b)
    -expm1(-abs(a-b))*exp(largest-max(largest,log(floatmin(Float64))))
end
function _row_variational_fit(target,method,entropy;work=nothing)
    C=_row_warm_start(target);n=length(C.tensors)
    fixed=_row_mixed_messages(target,C,false)
    valid=_row_initialize_amplitude!(target,C,fixed)
    best=valid && _row_candidate_allowed(C,method) ? copy(C) : nothing
    profile=fill(NaN,n-1);bestprofile=copy(profile)
    previous=NaN;relative_change=Inf;sweeps=0;reason=:sweep_limit;fallback=nothing
    maxdiscard=0.0
    for sweep in 1:method.max_sweep
        sweeps=sweep
        try
            for right in (true,false)
                fixed=_row_half_sweep!(C,target,method,right,entropy,profile,fixed;work)
                if _row_candidate_allowed(C,method) && (best===nothing || C.log_scale>=best.log_scale)
                    best=copy(C);bestprofile=copy(profile)
                end
            end
        catch exception
            failure=_variational_failure(exception)
            failure===nothing && rethrow()
            # Recover from the last valid center.
            if isfinite(C.log_scale) && _row_candidate_allowed(C,method) && (best===nothing || C.log_scale>best.log_scale)
                best=copy(C);bestprofile=copy(profile)
            end
            best===nothing && rethrow()
            fallback=(;reason=failure,used_svd=false);reason=:numerical_fallback
            maxdiscard=max(maxdiscard,C.max_truncation)
            break
        end
        maxdiscard=max(maxdiscard,C.max_truncation)
        w=2C.log_scale
        relative_change=sweep==1 ? Inf : _row_relative_change(w,previous)
        previous=w
        if sweep>1 && method.threshold_sweep>0 && relative_change<=method.threshold_sweep
            reason=:threshold_sweep;break
        end
    end
    best===nothing && throw(_VariationalNumericalError(:no_valid_candidate))
    best,(;sweeps,relative_change,stop_reason=reason,max_local_discarded_weight=maxdiscard,fallback),bestprofile
end

function _row_direct(target)
    C=_row_product(target)
    edge=_row_mixed_edge(target,C,true)
    H=_row_half_projection(edge,target,1,true)
    Htensor=_row_cap(_row_cap(H.tensor,2),_row_horizontal_label(target,2))
    # Retain the left candidate unit leg; append the right unit bond.
    V=oneunit(space(C.tensors[1],3))
    Htensor=_diagonal_join(Htensor,_diagonal_unit(V,_row_candidate_label(target,2);T=scalartype(Htensor.data)))
    A=_diagonal_reorder(Htensor,[_row_candidate_label(target,1),_row_output_label(target,1),_row_candidate_label(target,2)]).data
    A=_twistdual(A,(1,));s=norm(A)
    isfinite(s) || throw(_VariationalNumericalError(:nonfinite_projection))
    C.tensors[1]=iszero(s) ? A : A/s
    C.log_scale=iszero(s) ? -Inf : H.log_scale+log(s)
    target.reference=0.0
    C
end

function _absorb(env::RowVariationalBoundary,row,method::BoundaryMPSRowVariational;
        side=:north,entropy=nothing,row_cut=0,work=nothing)
    target=_row_layer_target(env,row;side)
    n=length(row);profile=fill(NaN,max(0,n-1))
    report=(;sweeps=0,relative_change=0.0,stop_reason=:direct,max_local_discarded_weight=0.0,fallback=nothing)
    if target.zero
        C=_row_product(target);C.tensors[1]=zero(C.tensors[1]);C.log_scale=-Inf
        target.reference=0.0;report=merge(report,(;stop_reason=:zero_target))
    elseif n==1
        C=_row_direct(target)
        C.log_scale==-Inf && (report=merge(report,(;stop_reason=:zero_target)))
    else
        C,report,profile=_row_variational_fit(target,method,entropy;work)
    end
    C.log_scale+=target.log_scale+target.reference
    C.max_truncation=max(env.max_truncation,report.max_local_discarded_weight)
    C.variational=vcat(env.variational,[merge((;layer=side==:north ? row_cut : row_cut+1,reverse=side==:south),report)])
    if entropy!==nothing && entropy.Se!==nothing && 1<=row_cut<=size(entropy.Se.north,1)
        getproperty(entropy.Se,side)[row_cut,:]=profile
    end
    C
end

function lognorm(state::FiniteState,method::BoundaryMPSRowVariational)
    layers=_row_layers(state,method)
    B=_trivial_boundary(collect(layers[:,1]),:north)
    for y in axes(layers,2)
        B=_absorb(B,collect(layers[:,y]),method;row_cut=y)
    end
    value=log(_positive_real(_close_boundary(B.tensors),"norm"))+B.log_scale
    merge((;log_norm=value+2state.log_amplitude,tensor_log_norm=value,
        max_environment_truncation=B.max_truncation),_variational_result(method,B.variational))
end
