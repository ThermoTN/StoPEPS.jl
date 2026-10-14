"""A row vertex stored as separate ket and bra tensors."""
struct FactorizedRowSite
    ket::AbstractTensorMap
    bra::AbstractTensorMap
end
FactorizedRowSite(A)=FactorizedRowSite(A,A)
TensorKit.space(E::FactorizedRowSite,k::Int)=_double_space(E.ket,E.bra,_VIRTUAL_AXES[k])
TensorKit.scalartype(E::FactorizedRowSite)=promote_type(scalartype(E.ket),scalartype(E.bra))
_row_working_copy(row::AbstractVector{FactorizedRowSite})=copy(row)

function _row_apply(message,E::FactorizedRowSite,ports,inputs;kwargs...)
    _diagonal_apply(message,E.ket,E.bra,ports,inputs;kwargs...)
end

# Contract each vertex with the incoming boundary, then its ket and bra tensors.
function _left_step(L,T,E::FactorizedRowSite,B)
    X=_diagonal_join(_DiagonalTensor(L,[1,2,3]),_DiagonalTensor(T,[1,4,5]))
    X=_row_apply(X,E,(2,4,6,7),(2,4))
    X=_diagonal_join(X,_DiagonalTensor(B,[3,7,8]))
    _diagonal_reorder(X,[5,6,8]).data
end
function _right_step(T,E::FactorizedRowSite,B,R)
    X=_diagonal_join(_DiagonalTensor(T,[1,4,5]),_DiagonalTensor(R,[5,6,8]))
    X=_row_apply(X,E,(2,4,6,7),(4,6))
    X=_diagonal_join(X,_DiagonalTensor(B,[3,7,8]))
    _diagonal_reorder(X,[1,2,3]).data
end
function _left_step(L,T,E::SlabColumn{FactorizedRowSite,FactorizedRowSite},B)
    X=_diagonal_join(_DiagonalTensor(L,[1,2,3,4]),_DiagonalTensor(T,[1,5,6]))
    X=_row_apply(X,E.upper,(2,5,7,8),(2,5))
    X=_row_apply(X,E.lower,(3,8,9,10),(3,8))
    X=_diagonal_join(X,_DiagonalTensor(B,[4,10,11]))
    _diagonal_reorder(X,[6,7,9,11]).data
end
function _right_step(T,E::SlabColumn{FactorizedRowSite,FactorizedRowSite},B,R)
    X=_diagonal_join(_DiagonalTensor(T,[1,5,6]),_DiagonalTensor(R,[6,7,9,11]))
    X=_row_apply(X,E.upper,(2,5,7,8),(5,7))
    X=_row_apply(X,E.lower,(3,8,9,10),(8,9))
    X=_diagonal_join(X,_DiagonalTensor(B,[4,10,11]))
    _diagonal_reorder(X,[1,2,3,4]).data
end

"""Select sequential ket/bra reduced metrics while sharing the full-hole repair."""
struct FactorizedRowBondEnvironment
    environment::BondEnvironment
end
_row_bond_environment(::AbstractMatrix{FactorizedRowSite},env)=FactorizedRowBondEnvironment(env)
bond_hole(env::FactorizedRowBondEnvironment)=bond_hole(env.environment)

function _row_cap(message,label)
    axis=something(findfirst(==(label),message.labels))
    _diagonal_join(message,_diagonal_unit(dual(space(message.data,axis)),label;T=scalartype(message.data)))
end

function _reduced_metric(wrapped::FactorizedRowBondEnvironment,Qa,Qb)
    env=wrapped.environment
    A=FactorizedRowSite(_diagonal_core_site(Qa,env.direction,true))
    B=FactorizedRowSite(_diagonal_core_site(Qb,env.direction,false))
    if env.direction==:horizontal
        T1,T2=env.top;B1,B2=env.bottom
        L=_diagonal_join(_DiagonalTensor(env.left,[1,2,3]),_DiagonalTensor(T1,[1,4,5]))
        L=_row_apply(L,A,(2,4,6,7),(2,4);physical=(-11,-12))
        L=_row_cap(_diagonal_join(L,_DiagonalTensor(B1,[3,7,8])),6)
        R=_diagonal_join(_DiagonalTensor(env.right,[9,10,11]),_DiagonalTensor(T2,[5,12,9]))
        R=_row_apply(R,B,(13,12,10,14),(12,10);physical=(-13,-14))
        R=_row_cap(_diagonal_join(R,_DiagonalTensor(B2,[8,14,11])),13)
    else
        L=_diagonal_join(_DiagonalTensor(env.left,[1,2,3,4]),_DiagonalTensor(only(env.top),[1,5,6]))
        L=_row_apply(L,A,(2,5,7,8),(2,5);physical=(-11,-12))
        L=_diagonal_join(_row_cap(L,8),_DiagonalTensor(env.right,[6,7,9,10]))
        R=_diagonal_join(_DiagonalTensor(only(env.bottom),[4,12,10]),
            _diagonal_unit(dual(space(B,2)),11;T=scalartype(B)))
        R=_row_apply(R,B,(3,11,9,12),(11,12);physical=(-13,-14))
    end
    result=_diagonal_reorder(_diagonal_join(L,R),[-12,-14,-11,-13])
    permute(result.data,((1,2),(3,4)))
end
