"""A Born sample with its physical normalization and estimator weight."""
mutable struct Particle{S<:FinitePEPS}
    id::Int
    generation::Int
    sample_index::Int
    seed::Int
    count::Int
    configuration::Matrix{Int}
    state::S
    log_p0::Float64
    log_q_proposal::Float64
    relative_log_weight::Float64
    weight_log_scale::Float64
    physical_log_norm::Float64
    tensor_log_norm::Float64
    energy_total::Float64
    energy_per_site::Float64
    max_state_truncation::Float64
    max_environment_truncation::Float64
    max_boundary_entropy::Float64
    Se::Union{Nothing,NamedTuple}
    time_seconds::Float64
    measurement_seconds::Float64
    pending_history::Vector{NamedTuple}
end

function _boundary(c,role::Symbol)
    chi=getfield(c,Symbol(:chi_,role)); V=getfield(c,Symbol(role,:_space))
    c.contraction<:BoundaryMPSDiagonalVariational && return c.contraction(chi,V;
        max_sweep=c.max_sweep,threshold_sweep=c.threshold_sweep,target_cache_bytes=c.target_cache_bytes)
    c.contraction<:BoundaryMPSRowVariational && return c.contraction(chi,V;
        max_sweep=c.max_sweep,threshold_sweep=c.threshold_sweep)
    V===nothing ? c.contraction(chi) : c.contraction(chi,V)
end
function _observe(state,c;energy)
    started=_now_seconds()
    m=energy ? measure_energy(state,c.model,_boundary(c,:measurement);record_entropy=c.record_boundary_entropy) :
        merge(lognorm(state,_boundary(c,:measurement)),(energy_total=NaN,energy_per_site=NaN,Se=nothing))
    merge(m,(measurement_seconds=_now_seconds()-started,))
end

function _new_particle(cp,c,configuration;id,generation=0,count=1,log_p0,log_q=log_p0,
        sampling_error=0.0,energy=false,basis=_basis(c))
    state,normalizations=project_configuration(cp.state,configuration;basis)
    m=_observe(state,c;energy)
    history=NamedTuple[(;phase=:projection,beta=state.beta,normalizations)]
    hasproperty(m,:boundary_variational) && push!(history,
        (;phase=:measurement,beta=state.beta,boundary_variational=m.boundary_variational))
    Particle(id,generation,id,c.seed,count,copy(configuration),state,log_p0,log_q,
        m.log_norm-log_q,0.0,m.log_norm,m.tensor_log_norm,m.energy_total,m.energy_per_site,
        0.0,max(sampling_error,m.max_environment_truncation),NaN,m.Se,0.0,m.measurement_seconds,history)
end

"""Actual beta endpoints, including shortened steps at output nodes."""
function _step_schedule(start,target,delta_tau)
    steps=Float64[]; b=Float64(start)
    while !_same_beta(b,target)
        target>b || error("evolution requires increasing beta endpoints")
        b=min(target,b+2delta_tau)
        _same_beta(b,target) && (b=Float64(target))
        push!(steps,b)
    end
    steps
end

function _evolve!(state,c,steps,D;on_step=r->nothing)
    total=0.0; maxstate=0.0; maxenv=0.0; normalization=0.0
    maxentropy=c.record_boundary_entropy && !isempty(steps) ? 0.0 : NaN
    for beta in steps
        started=_now_seconds()
        r=evolution_step!(state,c.model,(beta-state.beta)/2,D,
            c.update,_boundary(c,:environment);order=c.trotter_order,
            record_entropy=c.record_boundary_entropy,
            sector_space=state isa FinitePEPO ? c.bond_space_pur : c.bond_space_peps)
        elapsed=_now_seconds()-started
        state.beta=beta
        normalization+=sum(n.log_normalization for n in r.normalizations)
        total+=elapsed;maxstate=max(maxstate,r.max_state_truncation)
        maxenv=max(maxenv,r.max_environment_truncation)
        maxentropy=max(maxentropy,r.max_boundary_entropy)
        on_step(merge(r,(;beta,time_seconds=elapsed)))
    end
    (;time_seconds=total,max_state_truncation=maxstate,max_environment_truncation=maxenv,
        max_boundary_entropy=maxentropy,normalization)
end
function _advance_particle!(p,c,steps;energy,progress=r->nothing)
    oldnorm=p.tensor_log_norm
    report=_evolve!(p.state,c,steps,c.D_peps;on_step=r->begin
        push!(p.pending_history,r);progress(r)
    end)
    m=_observe(p.state,c;energy)
    hasproperty(m,:boundary_variational) && push!(p.pending_history,
        (;phase=:measurement,beta=p.state.beta,boundary_variational=m.boundary_variational))
    p.relative_log_weight+=2report.normalization+(m.tensor_log_norm-oldnorm)
    p.physical_log_norm=m.log_norm;p.tensor_log_norm=m.tensor_log_norm
    p.energy_total=m.energy_total;p.energy_per_site=m.energy_per_site
    p.max_state_truncation=report.max_state_truncation
    p.max_environment_truncation=max(report.max_environment_truncation,m.max_environment_truncation)
    p.max_boundary_entropy=report.max_boundary_entropy;p.Se=m.Se
    p.time_seconds=report.time_seconds;p.measurement_seconds=m.measurement_seconds
    p
end
function _particle_scalar(p;owner=0)
    record=(;id=p.id,generation=p.generation,sample_index=p.sample_index,seed=p.seed,count=p.count,
        configuration=copy(p.configuration),owner,beta=p.state.beta,
        log_p0=p.log_p0,log_q_proposal=p.log_q_proposal,
        relative_log_weight=p.relative_log_weight,weight_log_scale=p.weight_log_scale,
        log_weight=p.weight_log_scale+p.relative_log_weight,
        physical_log_norm=p.physical_log_norm,log_amplitude=p.state.log_amplitude,
        energy_total=p.energy_total,energy_per_site=p.energy_per_site,
        max_state_truncation=p.max_state_truncation,max_environment_truncation=p.max_environment_truncation,
        max_boundary_entropy=p.max_boundary_entropy,Se=p.Se===nothing ? nothing : map(copy,p.Se),
        time_seconds=p.time_seconds,measurement_seconds=p.measurement_seconds)
    record
end
function _global_particles(particles,e)
    local_records=_phase(()->[_particle_scalar(p;owner=_rank(e)) for p in particles],e)
    _collect_records(local_records,e)
end
function _archive!(particles,out,e;phase=:ordinary)
    local_history=_phase(e) do
        entries=NamedTuple[]
        for p in particles
            isempty(p.pending_history) && continue
            push!(entries,(;id=p.id,generation=p.generation,phase,history=copy(p.pending_history)))
            empty!(p.pending_history)
        end
        entries
    end
    history=_collect_records(local_history,e)
    _root(e) do
        append!(out.data["diagnostics"]["history"],history)
        reports=[r for entry in history for r in entry.history if hasproperty(r,:bond_updates)]
        _log_fu_warnings(out,reports;phase=phase==:ordinary ? :stochastic : phase)
        pop!(out.data["diagnostics"],"in_progress_history",nothing)
    end
end
function _log_particles!(particles,out,e;phase=:ordinary)
    _phase(e) do
        for p in particles
            _sample_log(out,_particle_scalar(p;owner=_rank(e));phase)
        end
    end
end
function _set_selected!(p,selection,i)
    p.count=selection.multiplicity[i]
    p.relative_log_weight=selection.output_relative_log_weight[i]
    p.weight_log_scale=selection.common_log_weight
    p
end
