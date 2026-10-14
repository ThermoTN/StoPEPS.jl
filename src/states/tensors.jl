# Tensor indices are outgoing; contractions pair dual TensorKit spaces.
_out(t::AbstractTensorMap) = permute(t, (Tuple(1:numind(t)), ()))
_perm(t, p) = permute(t, (Tuple(p), ()))
_conj(t) = TensorOperations.tensorcopy(Tuple(1:numind(t)), t,
    Tuple(1:numind(t)), true)
_product(v) = reduce(⊗, v)
_finite(t::AbstractTensorMap) = all(b -> all(isfinite, last(b)), blocks(t))
_fermionic(t)=BraidingStyle(sectortype(t)) isa Fermionic
_twistdual(t,indices)=_fermionic(t) ? twist(t,Tuple(index for index in indices if isdual(space(t,index)))) : t
_linearmap_metric(t)=_twistdual(t,1:numind(t))

function _fusionmap(::Type{Scalar},spaces,dual_output) where Scalar
    group=_product(spaces)
    fused=fuse(group)
    isdual(fused)==dual_output || (fused=flip(fused))
    if dual_output && BraidingStyle(sectortype(fused)) isa Fermionic
        opposite=isomorphism(Scalar,dual(fused),_product(dual.(spaces)))
        inverse=_twistdual(opposite',Tuple(1:length(spaces)))
        return permute(inverse,((length(spaces)+1,),Tuple(1:length(spaces))))
    end
    isomorphism(Scalar,fused,group)
end

function _unit_tensor(spaces; T=Float64)
    all(isunitspace, spaces) || throw(DimensionMismatch("nontrivial OBC leg"))
    return ones(T,_product(spaces)←one(first(spaces)))
end

"""Fuse a contiguous group through a TensorKit isomorphism, preserving other legs."""
function _fuse(t, first::Int, count::Int; dual_output=isdual(space(t, first)))
    count == 1 && return _out(t)
    group = _product(space(t, k) for k in first:first+count-1)
    V = fuse(group)
    isdual(V) == dual_output || (V = flip(V))
    if sectortype(t) === Trivial
        t=_out(t)
        spaces=collect(space(t,k) for k in 1:numind(t))
        splice!(spaces,first:first+count-1,[V])
        return _plain_reindex(t,spaces)
    end
    F = _fusionmap(scalartype(t),Tuple(space(t,k) for k in first:first+count-1),dual_output)
    labels = collect(-1:-1:-numind(t))
    labels[first:first+count-1] .= 1:count
    flabels = [-first; collect(1:count)]
    output = [-k for k in 1:numind(t) if !(first < k < first+count)]
    return ncon([F, t], [flabels, labels]; output)
end

function _split(t, axis::Int, spaces)
    if sectortype(t) === Trivial
        space(t,axis) ≅ _product(spaces) || throw(SpaceMismatch("tensor splitting space mismatch"))
        t=_out(t)
        output=collect(space(t,k) for k in 1:numind(t))
        splice!(output,axis:axis,collect(spaces))
        return _plain_reindex(t,output)
    end
    F = _fusionmap(scalartype(t),Tuple(spaces),isdual(space(t,axis)))
    labels = collect(-1:-1:-numind(t)); labels[axis] = 1
    extra = [-numind(t)-k for k in eachindex(spaces)]
    output = vcat(labels[1:axis-1], extra, labels[axis+1:end])
    inverse=adjoint(F)
    if _fermionic(t)
        twists=Tuple([findall(isdual,collect(spaces));isdual(space(t,axis)) ? [length(spaces)+1] : Int[]])
        inverse=twist(inverse,twists)
    end
    return ncon([t, inverse], [labels, [extra; 1]]; output)
end

# Trivial tensors use array-order fusion. Copy the native block into
# independent storage for the reshaped tensor.
function _plain_reindex(t,spaces)
    sectortype(t) === Trivial || throw(ArgumentError("plain reindex requires Trivial sectors"))
    TensorMap(copy(vec(block(t,Trivial()))),_product(spaces),one(first(spaces)))
end

"""PSD projection by clipping eigenvalues in native sector blocks."""
function _positive_metric(N; square_root=false)
    _finite(N) || error("nonfinite norm metric")
    scale = norm(N)
    scaled = iszero(scale) ? N : N / scale
    H = (scaled + scaled') / 2
    D, V = eigh_full(H)
    vals = Float64[real(v) for (_, b) in blocks(D) for v in diag(b)]
    weights = Float64[dim(c) for (c,b) in blocks(D) for _ in diag(b)]
    minimum_value = minimum(vals; init=0.0)
    spectral_scale = max(maximum(abs, vals; init=0.0), floatmin(Float64))
    clipped = copy(D)
    for (_, b) in blocks(clipped), i in axes(b, 1)
        b[i,i] = max(real(b[i,i]), 0)
    end
    repaired = V * clipped * V'
    repaired = (repaired + repaired') / 2
    report = (; minimum_relative_eigenvalue=minimum_value/spectral_scale,
        negative_eigenvalues=sum(weights[vals .< 0];init=0.0),
        negative_multiplets=count(<(0),vals),
        negative_spectral_fraction=sqrt(sum(weights.*abs2.(min.(vals,0))))/
            max(sqrt(sum(weights.*abs2.(vals))),floatmin(Float64)),
        hermiticity_defect=norm(scaled-scaled')/max(norm(scaled), floatmin(Float64)))
    if square_root
        for (_, b) in blocks(clipped), i in axes(b,1)
            b[i,i]=sqrt(real(b[i,i]))
        end
        return clipped*V', report
    end
    return repaired, report
end
