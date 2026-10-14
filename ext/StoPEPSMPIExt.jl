module StoPEPSMPIExt
using StoPEPS, MPI
const SP=StoPEPS

struct MPIExecution <: SP.AbstractExecution
    comm::MPI.Comm
end
function SP._execution(comm::MPI.Comm)
    MPI.Initialized() && !MPI.Finalized() || error("initialize MPI before run_stochastic")
    MPIExecution(comm)
end
SP._rank(e::MPIExecution)=MPI.Comm_rank(e.comm)
SP._nprocs(e::MPIExecution)=MPI.Comm_size(e.comm)
SP._gather(x,e::MPIExecution)=MPI.gather(x,e.comm;root=0)

# All ranks finish the same error collective before the next communication.
function SP._phase(f,e::MPIExecution)
    result,failure=nothing,""
    try
        result=f()
    catch exception
        failure=sprint(showerror,exception,catch_backtrace())
    end
    failures=MPI.gather(failure,e.comm;root=0)
    message=SP._rank(e)==0 ? join(["rank $(i-1): $s" for (i,s) in enumerate(failures) if !isempty(s)],"\n") : nothing
    message=MPI.bcast(message,e.comm;root=0)
    isempty(message) || error(message)
    result
end
SP._broadcast(f,e::MPIExecution)=MPI.bcast(SP._root(f,e),e.comm;root=0)

function SP._redistribute(particles,selection,e::MPIExecution)
    rank,n=SP._rank(e),SP._nprocs(e)
    pool=Dict(p.id=>p for p in particles)
    next=SP.Particle[]
    # Move selected tensors in a common ordered pairwise schedule.
    for (position,i) in enumerate(selection.active)
        record=selection.pool[i]
        source,destination=record.owner,mod(position-1,n)
        if source==destination
            rank==source && push!(next,SP._set_selected!(pool[record.id],selection,i))
        elseif rank==source
            MPI.send(pool[record.id],e.comm;dest=destination,tag=910)
        elseif rank==destination
            p=MPI.recv(e.comm;source,tag=910)
            push!(next,SP._set_selected!(p,selection,i))
        end
    end
    sort!(next;by=p->p.id)
end
end
