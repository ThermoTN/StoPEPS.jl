"""Retry only LAPACK nonconvergence, using the original input for both attempts."""
function _retry_svd(f,a;context,sector,
        primary=MAK.DivideAndConquer(),fallback=MAK.QRIteration())
    try
        return f(primary)
    catch err
        err isa LinearAlgebra.LAPACKException && err.info>0 || rethrow()
        @warn "SVD gesdd failed to converge; retrying with gesvd" _group=:svd_retry context sector matrix_size=size(a) scalar_type=eltype(a) lapack_info=err.info
        return f(fallback)
    end
end

# TensorKit owns the factor spaces and truncation. Only the dense block's
# decomposition algorithm changes, preserving the native multiplet structure.
struct _SVDWithRetry{C} <: MAK.AbstractAlgorithm
    context::C
end
function MAK.svd_compact!(t::AbstractTensorMap,factors,alg::_SVDWithRetry)
    TensorKit.foreachblock(t,factors...) do sector,(a,outputs...)
        _retry_svd(a;context=alg.context,sector) do backend
            MAK.svd_compact!(copy(a),outputs,backend)
        end
    end
    factors
end

# Diagonal tensors already have an exact decomposition without LAPACK.
function _svd_algorithm(t,context)
    default=MAK.default_algorithm(MAK.svd_compact!,typeof(t))
    default isa MAK.DiagonalAlgorithm ? default : _SVDWithRetry(context)
end
_svd_compact(t;context)=svd_compact(t,_svd_algorithm(t,context))
_svd_trunc(t;trunc,context)=svd_trunc(t;alg=_svd_algorithm(t,context),trunc)
