"""
    Lattice(Lx, Ly)

Finite square lattice with open boundaries. Sites are stored as `[x,y]`,
with x west-to-east and y north-to-south; traversal is row-major.
"""
struct Lattice
    Lx::Int
    Ly::Int
    function Lattice(Lx::Integer, Ly::Integer)
        Lx > 0 && Ly > 0 || throw(ArgumentError("lattice dimensions must be positive"))
        new(Int(Lx), Int(Ly))
    end
end
Base.length(l::Lattice) = l.Lx*l.Ly
site_order(l::Lattice) = [CartesianIndex(x,y) for y in 1:l.Ly for x in 1:l.Lx]

abstract type StateKind end
struct Purification <: StateKind end
struct Wavefunction <: StateKind end

"""Finite OBC network. Every site has outgoing (W,N,Pout,Paux,E,S) legs.

PEPS retains one irreducible Paux carrier. The state includes exp(log_amplitude).
TensorKit spaces, including bond orientations, are retained in every kernel.
"""
mutable struct FiniteState{K<:StateKind,TM<:AbstractTensorMap}
    lattice::Lattice
    sites::Matrix{TM}
    beta::Float64
    log_amplitude::Float64
end
const FinitePEPO = FiniteState{Purification}
const FinitePEPS = FiniteState{Wavefunction}
function FiniteState{K}(lattice, sites; beta=0.0, log_amplitude=0.0) where {K<:StateKind}
    stored = map(_out, sites)
    # Different local bases can produce real and complex sites in one state.
    # Promote once so later bond updates can safely store either site's result.
    TM = mapreduce(typeof, promote_type, stored)
    stored = map(A -> convert(TM, A), stored)
    state = FiniteState{K,TM}(lattice, stored, Float64(beta), Float64(log_amplitude))
    validate_state(state)
    return state
end
Base.getindex(s::FiniteState, x::Int, y::Int) = s.sites[x,y]
Base.getindex(s::FiniteState, i::CartesianIndex) = s.sites[i]
Base.copy(s::FiniteState{K}) where K = FiniteState{K}(s.lattice, copy.(s.sites);
    beta=s.beta, log_amplitude=s.log_amplitude)

"""
Validate tensor ranks, finite entries, auxiliary carriers and lattice bond spaces.
"""
function validate_state(s::FiniteState)
    l = s.lattice
    size(s.sites) == (l.Lx,l.Ly) || throw(DimensionMismatch("state lattice mismatch"))
    for i in site_order(l)
        x,y = Tuple(i); A = s[i]
        numind(A) == 6 || throw(DimensionMismatch("expected six local legs"))
        _finite(A) || error("nonfinite tensor at $i")
        if s isa FinitePEPS
            auxiliary=space(A,4)
            charges=collect(sectors(auxiliary))
            length(charges)==1 && dim(auxiliary,only(charges))==1 ||
                error("PEPS auxiliary leg must contain one irreducible carrier")
        end
        for (edge,axis) in ((x==1,1),(y==1,2),(x==l.Lx,5),(y==l.Ly,6))
            edge && !isunitspace(space(A,axis)) && error("nontrivial OBC leg at $i")
        end
        x < l.Lx && space(A,5) != dual(space(s[x+1,y],1)) && error("horizontal space mismatch")
        y < l.Ly && space(A,6) != dual(space(s[x,y+1],2)) && error("vertical space mismatch")
    end
    return s
end

"""
    identity_pepo(lattice; T=Float64, physical_space=ComplexSpace(2))

Construct the identity thermal factor at beta=0 on a finite open lattice.
"""
function identity_pepo(l::Lattice; T=Float64,physical_space=ComplexSpace(2))
    P=physical_space;V=oneunit(P)
    boundary=_unit_tensor((V',V',V,V);T)
    identity=id(T,P)
    @tensor A[w,n,p,a,e,s] := boundary[w,n,e,s]*identity[p,a]
    return FinitePEPO(l,[copy(A) for x in 1:l.Lx,y in 1:l.Ly])
end

"""Accumulate and record local tensor normalization factors."""
function normalize_state!(s::FiniteState)
    records = NamedTuple[]
    for i in site_order(s.lattice)
        value = norm(s[i])
        isfinite(value) && value > 0 || error("zero or nonfinite tensor at $i")
        s.sites[i] = s[i]/value
        increment = log(value)
        s.log_amplitude += increment
        push!(records, (;site=Tuple(i), log_normalization=increment))
    end
    return records
end
