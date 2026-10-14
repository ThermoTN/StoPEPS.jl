abstract type AbstractModel end
"""
    TFIM(; J=1.0, h=0.7, physical_space=ComplexSpace(2))

Transverse-field Ising model `H=-J*sum(ZZ)-h*sum(X)` with Pauli eigenvalues ±1.
For the paper's spin-1/2 couplings J and B, use `TFIM(J=J/4,h=B/2)`.
The physical space selects dense or Z₂-symmetric tensors.
"""
struct TFIM{S<:ElementarySpace} <: AbstractModel
    J::Float64
    h::Float64
    physical_space::S
end
function TFIM(;J=1.0,h=0.7,physical_space=ComplexSpace(2))
    P=physical_space
    (P==ComplexSpace(2) || P==Z2Space(0=>1,1=>1)) ||
        throw(ArgumentError("TFIM supports ComplexSpace(2) or Z2Space(0=>1,1=>1) in the X eigenbasis"))
    TFIM(Float64(J),Float64(h),P)
end
TFIM(J::Real,h::Real)=TFIM(;J,h)
physical_space(m::TFIM)=m.physical_space
function spin_operators(model::TFIM, ::Type{T}=Float64) where T
    P=physical_space(model)
    if sectortype(P)===Trivial
        X=TensorMap(T[0 1;1 0],P,P); Z=TensorMap(T[1 0;0 -1],P,P)
        return (;X,Z,I=id(T,P),ZZ=Z⊗Z)
    end
    # Physical X generates parity. Construct the invariant two-site Z product
    # as a TensorMap.
    X=TensorMap(T[1 0;0 -1],P,P)
    z=T[0 1;1 0]
    ZZ=TensorMap(kron(z,z),P⊗P,P⊗P)
    (;X,Z=nothing,I=id(T,P),ZZ)
end
function bonds(l::Lattice, direction::Symbol)
    direction == :horizontal && return [(CartesianIndex(x,y),CartesianIndex(x+1,y))
        for y in 1:l.Ly for x in 1:l.Lx-1]
    direction == :vertical && return [(CartesianIndex(x,y),CartesianIndex(x,y+1))
        for y in 1:l.Ly-1 for x in 1:l.Lx]
    throw(ArgumentError("unknown bond direction"))
end
_degree(l, i) = (i[1]>1)+(i[1]<l.Lx)+(i[2]>1)+(i[2]<l.Ly)
onsite_hamiltonian(model::TFIM;T=Float64)=-model.h*spin_operators(model,T).X
function bond_hamiltonian(m::TFIM, l::Lattice, a, b; T=Float64)
    o=spin_operators(m,T)
    return -m.J*o.ZZ - m.h/_degree(l,a)*(o.X⊗o.I) - m.h/_degree(l,b)*(o.I⊗o.X)
end
trotter_gate(m::AbstractModel,l,a,b,delta;T=Float64) = exp(-delta*bond_hamiltonian(m,l,a,b;T))

"""Dense Hamiltonian for small-system exact-diagonalization comparisons."""
function dense_hamiltonian(l::Lattice, m::TFIM)
    N=length(l); N<=16 || throw(ArgumentError("ED is limited to 16 sites"))
    H=zeros(Float64,2^N,2^N)
    idx(i)=i[1]+(i[2]-1)*l.Lx
    z(b,i)=iszero(b & (1<<(i-1))) ? 1 : -1
    for b in 0:2^N-1
        for (i,j) in [bonds(l,:horizontal);bonds(l,:vertical)]
            H[b+1,b+1] -= m.J*z(b,idx(i))*z(b,idx(j))
        end
        for i in 1:N
            H[(b ⊻ (1<<(i-1)))+1,b+1] -= m.h
        end
    end
    H
end
"""
    exact_thermal_observables(lattice, model, beta)

Small-system exact thermal energy per site, total energy and log partition
function. The Hubbard method also returns particle number and density.
"""
function exact_thermal_observables(l::Lattice,m::TFIM,beta)
    e=eigvals(Hermitian(dense_hamiltonian(l,m))); shift=minimum(e)
    w=exp.(-beta.*(e.-shift)); Z=sum(w); total=dot(w,e)/Z
    return (;energy_total=total,energy_per_site=total/length(l),lnZ=log(Z)-beta*shift)
end
