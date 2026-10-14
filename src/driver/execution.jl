abstract type AbstractExecution end
struct SerialExecution <: AbstractExecution end
_execution(::Nothing)=SerialExecution()
_execution(comm)=error("load MPI before passing a communicator")
_rank(::SerialExecution)=0
_nprocs(::SerialExecution)=1
_phase(f,::SerialExecution)=f()
_root(f,e::AbstractExecution)=_phase(()->_rank(e)==0 ? f() : nothing,e)
_broadcast(f,e::SerialExecution)=f()
_gather(x,::SerialExecution)=[x]
function _collect_records(local_records,e;by=r->r.id)
    parts=_gather(local_records,e)
    _broadcast(e) do
        sort!(reduce(vcat,parts);by)
    end
end
function _wall_time(local_seconds,e)
    parts=_gather(local_seconds,e)
    _broadcast(()->maximum(parts),e)
end
function _check_arguments(c,e)
    signatures=_gather(repr(parameter_dict(c)),e)
    _root(e) do
        all(==(first(signatures)),signatures) || error("MPI ranks supplied different parameters")
    end
end

function _redistribute(particles,selection,e::SerialExecution)
    pool=Dict(p.id=>p for p in particles)
    [_set_selected!(pool[selection.pool[i].id],selection,i) for i in selection.active]
end

# Assign samples by their positions in the distributed work list.
_representative_owner(position,nranks)=mod(position-1,nranks)
