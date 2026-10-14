const HubbardSector = U1Irrep ⊠ SU2Irrep ⊠ FermionParity
const FERMION_CONVENTION = "TensorKit_U1(n-1)_SU2_FermionParity(n); W,N,Pout,Paux,E,S; ket,bra; v1"

_hubbard_sectors() = (HubbardSector(-1,0,0), HubbardSector(0,1//2,1), HubbardSector(1,0,0))
_hubbard_space() = Vect[HubbardSector]((_hubbard_sectors() .=> 1)...)

"""Square-lattice Hubbard model; evolution uses H-mu*N, energy reports H."""
struct Hubbard{S<:ElementarySpace} <: AbstractModel
    t::Float64
    U::Float64
    mu::Float64
    symmetry::Symbol
    physical_space::S
end
function Hubbard(;t=1.0,U=8.0,mu=U/2,symmetry=:u1su2)
    symmetry === :u1su2 || throw(ArgumentError("use symmetry=:u1su2 for Hubbard"))
    all(isfinite,(t,U,mu)) || throw(ArgumentError("nonfinite Hubbard parameter"))
    Hubbard(Float64(t),Float64(U),Float64(mu),symmetry,_hubbard_space())
end
physical_space(model::Hubbard)=model.physical_space

function hubbard_operators(model::Hubbard,::Type{Scalar}=Float64) where Scalar
    physical=physical_space(model)
    empty,single,double=_hubbard_sectors()
    number=zeros(Scalar,physical←physical)
    doublon=zeros(Scalar,physical←physical)
    block(number,single).=1
    block(number,double).=2
    block(doublon,double).=1
    hopping=zeros(Scalar,physical⊗physical←physical⊗physical)
    entries=((empty,single,single,empty,HubbardSector(-1,1//2,1),1),
        (single,double,double,single,HubbardSector(1,1//2,1),-1),
        (empty,double,single,single,HubbardSector(0,0,0),sqrt(2)),
        (single,single,double,empty,HubbardSector(0,0,0),sqrt(2)))
    for (outfirst,outsecond,infirst,insecond,total,coefficient) in entries
        outgoing=only(fusiontrees((outfirst,outsecond),total))
        incoming=only(fusiontrees((infirst,insecond),total))
        hopping[outgoing,incoming].=coefficient
    end
    (;I=id(Scalar,physical),N=number,D=doublon,hopping=hopping+hopping')
end
function onsite_hamiltonian(model::Hubbard;T=Float64)
    operators=hubbard_operators(model,T)
    model.U*operators.D-model.mu*operators.N
end
function bond_hamiltonian(model::Hubbard,lattice::Lattice,first,second;T=Float64)
    operators=hubbard_operators(model,T)
    onsite=model.U*operators.D-model.mu*operators.N
    -model.t*operators.hopping+(onsite⊗operators.I)/_degree(lattice,first)+
        (operators.I⊗onsite)/_degree(lattice,second)
end

"""Fock-space reference with row-major sites, up then down at each site."""
function dense_hamiltonian(lattice::Lattice,model::Hubbard)
    sites=length(lattice)
    sites<=6 || throw(ArgumentError("Hubbard ED is limited to six sites"))
    dimension=4^sites
    hamiltonian=zeros(Float64,dimension,dimension)
    position(site)=site[1]+(site[2]-1)*lattice.Lx
    for occupation in 0:dimension-1
        for site in 0:sites-1
            ((occupation>>(2site))&3)==3 && (hamiltonian[occupation+1,occupation+1]+=model.U)
        end
        for direction in (:horizontal,:vertical), (first,second) in bonds(lattice,direction),
                spin in 0:1, (destination,source) in ((position(first),position(second)),(position(second),position(first)))
            annihilate=2(source-1)+spin
            create=2(destination-1)+spin
            iszero(occupation&(1<<annihilate)) && continue
            intermediate=occupation⊻(1<<annihilate)
            iszero(intermediate&(1<<create)) || continue
            sign=isodd(count_ones(occupation&((1<<annihilate)-1))+
                count_ones(intermediate&((1<<create)-1))) ? -1 : 1
            hamiltonian[(intermediate⊻(1<<create))+1,occupation+1]-=model.t*sign
        end
    end
    hamiltonian
end
function exact_thermal_observables(lattice::Lattice,model::Hubbard,beta)
    hamiltonian=dense_hamiltonian(lattice,model)
    numbers=count_ones.(0:size(hamiltonian,1)-1)
    generator=hamiltonian-model.mu*Diagonal(numbers)
    eigenvalues,eigenvectors=eigen(Hermitian(generator))
    shift=minimum(eigenvalues)
    weights=exp.(-beta.*(eigenvalues.-shift))
    partition=sum(weights)
    energy_total=dot(weights,real.(diag(eigenvectors'*hamiltonian*eigenvectors)))/partition
    particle_number=dot(weights,real.(diag(eigenvectors'*Diagonal(numbers)*eigenvectors)))/partition
    (;energy_total,energy_per_site=energy_total/length(lattice),lnZ=log(partition)-beta*shift,
        particle_number,density=particle_number/length(lattice))
end
