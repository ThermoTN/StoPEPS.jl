"""Local orthonormal basis; columns of U are physical kets in the model basis."""
struct OrthonormalBasis{M<:AbstractMatrix}
    vectors::M
    labels::Vector{Int}
    function OrthonormalBasis(U::AbstractMatrix;labels=collect(1:size(U,2)))
        d=size(U,1)
        d>0 && size(U,2)==d || throw(DimensionMismatch("sampling basis must be square and complete"))
        labels=Int.(collect(labels))
        length(labels)==d && length(unique(labels))==d || throw(ArgumentError("basis labels must be distinct"))
        V=float.(Matrix(U))
        all(isfinite,V) || throw(ArgumentError("nonfinite sampling basis"))
        tol=128eps(real(float(one(eltype(V)))))*d
        isapprox(V'*V,Matrix{eltype(V)}(I,d,d);atol=tol,rtol=tol) ||
            throw(ArgumentError("sampling basis columns must be orthonormal"))
        new{typeof(V)}(copy(V),labels)
    end
end
"""Abelian charge states."""
struct SymmetryBasis end
XBasis()=OrthonormalBasis([1. 1.;-1. 1.]/sqrt(2);labels=[-1,1])
YBasis()=OrthonormalBasis(ComplexF64[1 1;-im im]/sqrt(2);labels=[-1,1])
ZBasis()=OrthonormalBasis([0. 1.;1. 0.];labels=[-1,1])

struct LocalSamplingBasis
    vectors::Matrix
    labels::Vector{Int}
    charges::Vector
    physical::Vector{AbstractTensorMap}
    auxiliary::Vector{AbstractTensorMap}
end
struct ResolvedBasis
    space::ElementarySpace
    sites::Matrix{LocalSamplingBasis}
end

function _local_basis(spec,P)
    I=sectortype(P)
    if I===Trivial
        spec isa OrthonormalBasis || throw(ArgumentError("nonsymmetric sampling requires an OrthonormalBasis"))
        size(spec.vectors,1)==dim(P) || throw(DimensionMismatch("sampling basis physical dimension"))
        # Copy and validate the basis descriptor for this cache.
        spec=OrthonormalBasis(spec.vectors;labels=spec.labels)
        U=spec.vectors; labels=spec.labels; charges=fill(one(I),dim(P))
    else
        spec isa SymmetryBasis || throw(ArgumentError("symmetric sampling requires SymmetryBasis()"))
        FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa Bosonic ||
            throw(ArgumentError("sampling currently supports bosonic Abelian symmetry bases"))
        charges=[c for c in sectors(P) for _ in 1:dim(P,c)]
        all(c->dim(c)==1,charges) || throw(ArgumentError("sampling requires one-dimensional irreps"))
        U=Matrix{Float64}(LinearAlgebra.I,dim(P),dim(P))
        labels=collect(1:dim(P))
    end
    physical=AbstractTensorMap[]; auxiliary=AbstractTensorMap[]
    for k in eachindex(labels)
        Q=I===Trivial ? oneunit(P) : spacetype(P)(charges[k]=>1)
        O=TensorMap(reshape(conj.(U[:,k]),1,:),Q,P)
        push!(physical,O)
        # X[p,a] contracted with <b*| gives X|b>, whereas physical
        # probabilities use <b|. This distinction is essential for complex U.
        push!(auxiliary,permute(_conj(O),((1,),(2,))))
    end
    LocalSamplingBasis(copy(U),copy(labels),charges,physical,auxiliary)
end
function _resolve_basis(spec,P,l::Lattice)
    if spec isa ResolvedBasis
        spec.space==P && size(spec.sites)==(l.Lx,l.Ly) || throw(DimensionMismatch("basis/state space or lattice mismatch"))
        return spec
    end
    if spec===:auto
        spec=sectortype(P)===Trivial ? (dim(P)==2 ? XBasis() : OrthonormalBasis(Matrix{Float64}(I,dim(P),dim(P)))) : SymmetryBasis()
    end
    if spec isa AbstractMatrix
        size(spec)==(l.Lx,l.Ly) || throw(DimensionMismatch("site-dependent basis lattice"))
        sites=LocalSamplingBasis[_local_basis(spec[x,y],P) for x in 1:l.Lx,y in 1:l.Ly]
    else
        B=_local_basis(spec,P)
        sites=fill(B,l.Lx,l.Ly)
    end
    ResolvedBasis(P,sites)
end
_basis(c)=_resolve_basis(c.sampling_basis,physical_space(c.model),c.lattice)
function _basis_index(B::LocalSamplingBasis,label)
    k=findfirst(==(label),B.labels)
    k===nothing && throw(ArgumentError("configuration contains an unknown basis label"))
    k
end
_project(A,B::LocalSamplingBasis,k,kind)=_act(A,kind==:auxiliary ? B.auxiliary[k] : B.physical[k],kind==:auxiliary ? 4 : 3)
_fuse_charge(a,b)=only(a⊗b)
function _state_charge(state::FinitePEPS)
    I=sectortype(state[1,1])
    foldl(_fuse_charge,(dual(only(sectors(space(state[i],4)))) for i in site_order(state.lattice));init=one(I))
end
function _charge_suffix(basis::ResolvedBasis,order=CartesianIndices(basis.sites))
    sites=[basis.sites[i] for i in order]; N=length(sites); I=typeof(first(first(sites).charges))
    reachable=Vector{Set{I}}(undef,N+1); reachable[end]=Set([one(I)])
    for i in N:-1:1
        reachable[i]=Set(_fuse_charge(c,r) for c in sites[i].charges for r in reachable[i+1])
    end
    reachable
end
function _basis_parameters(basis::ResolvedBasis)
    encode(B)=Dict("vectors_real"=>real.(B.vectors),"vectors_imag"=>imag.(B.vectors),
        "labels"=>B.labels,"charges"=>string.(B.charges))
    Dict("version"=>1,"convention"=>"physical_ket_columns; auxiliary_conjugate_basis",
        "site_order"=>"x_then_y","sites"=>[encode(B) for B in vec(basis.sites)])
end
