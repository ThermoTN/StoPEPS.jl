"""Prepare X(beta0/2) in one process and save the reusable TensorKit checkpoint."""
function run_purification(c::RunConfig)
    validate_config(c)
    out=_open_output(c,:purification)
    try
        state=identity_pepo(c.lattice;physical_space=physical_space(c.model))
        initial=normalize_state!(state)
        history=out.data["diagnostics"]["history"]
        push!(history,(;phase=:initialization,beta=0.0,normalizations=initial))
        out.main===nothing || _section(out.main,"finite-PEPO",PUR_COLUMNS)
        for beta in filter(b->b<=c.beta0,temperature_nodes(c))
            steps=_step_schedule(state.beta,beta,c.delta_tau_pur)
            report=_evolve!(state,c,steps,c.D_pur;on_step=r->begin
                push!(history,r)
                _log_fu_warnings(out,[r];phase=:purification)
                periodic_save!(out;progress=Dict("phase"=>"purification","beta"=>r.beta))
            end)
            m=_observe(state,c;energy=_measure_at(c,beta))
            hasproperty(m,:boundary_variational) && push!(history,
                (;phase=:measurement,beta,boundary_variational=m.boundary_variational))
            row=(;beta,source=:purification,energy_total=m.energy_total,energy_per_site=m.energy_per_site,
                energy_per_site_error=_measure_at(c,beta) ? 0.0 : NaN,
                lnZ=m.log_norm,lnZ_error=0.0,eta=NaN,eta_error=NaN,
                max_state_truncation=report.max_state_truncation,
                max_environment_truncation=max(report.max_environment_truncation,m.max_environment_truncation),
                max_boundary_entropy=report.max_boundary_entropy,Se=m.Se,
                time_seconds=report.time_seconds,measurement_seconds=m.measurement_seconds)
            push!(out.data["observables"]["purification"],row)
            push!(out.data["observables"]["curve"],row)
            _log_purification(out,row)
            periodic_save!(out;progress=Dict("phase"=>"purification","beta"=>beta))
        end
        cp=ThermalCheckpoint(copy(state),parameter_dict(c),copy(out.data["observables"]["purification"]),
            copy(history),out.run_id)
        c.output_directory===nothing || save_checkpoint(joinpath(c.output_directory,"results","checkpoint.jld2"),cp;
            overwrite=c.overwrite_output)
        _finish_output!(out)
        cp
    catch exception
        _finish_output!(out;error=sprint(showerror,exception))
        rethrow()
    end
end

function _initial_particles(cp,c,cache,e,out)
    local_draws=_phase(e) do
        [merge(sample_configuration(cache;rng=sample_rng(c.seed,i)),(;index=i))
            for i in 1:c.Ns if mod(i-1,_nprocs(e))==_rank(e)]
    end
    draws=_collect_records(local_draws,e;by=r->r.index)
    grouping=_group_configurations(getfield.(draws,:configuration))
    particles=_phase(e) do
        Particle[_new_particle(cp,c,draws[i].configuration;id=i,count=grouping.multiplicity[u],
            log_p0=draws[i].log_probability,sampling_error=draws[i].max_environment_truncation,
            energy=_measure_at(c,c.beta0),basis=cache.basis) for (u,i) in enumerate(grouping.first_draw)
            if _representative_owner(u,_nprocs(e))==_rank(e)]
    end
    _root(e) do
        out.data["diagnostics"]["born_draws"]=draws
        out.data["diagnostics"]["born_grouping"]=grouping
        out.main===nothing || println(out.main,"======================== Born sampling ================================")
        _log_draws(out,draws)
        out.main===nothing || _section(out.main,"stochastic PEPS",STO_COLUMNS)
    end
    particles
end

"""
    run_stochastic(config; checkpoint=nothing, checkpoint_path=nothing, comm=nothing)

Draw Born samples from the beta0 checkpoint and evolve the weighted ensemble.
Load MPI and pass a communicator for collective execution; rank zero returns
the numerical result dictionary. The output directory stores measurements in
`result.jld2` and the prepared thermal tensors in `checkpoint.jld2`.
"""
function run_stochastic(c::RunConfig;checkpoint=nothing,checkpoint_path=nothing,comm=nothing)
    e=_execution(comm)
    _phase(()->validate_config(c),e);_check_arguments(c,e)
    cp=_broadcast(e) do
        if checkpoint!==nothing
            checkpoint
        else
            path=checkpoint_path===nothing ? (c.output_directory===nothing ?
                error("supply a checkpoint or checkpoint_path") :
                joinpath(c.output_directory,"results","checkpoint.jld2")) : checkpoint_path
            load_checkpoint(path,c)
        end
    end
    _phase(e) do
        p=parameter_dict(c)
        _same_model_parameters(cp.preparation["model"],p["model"]) && cp.preparation["lattice"]==p["lattice"] &&
            _same_beta(cp.state.beta,c.beta0) || error("checkpoint model, lattice or beta0 mismatch")
    end
    out=_phase(()->_open_output(c,:stochastic;rank=_rank(e),nranks=_nprocs(e),checkpoint=cp),e)
    try
        cache=_phase(()->SamplingCache(cp.state,_boundary(c,:sampling);kind=:auxiliary,basis=_basis(c)),e)
        particles=_initial_particles(cp,c,cache,e,out)
        replay=NamedTuple[];event=0;next_sample_id=c.Ns+1
        nodes=filter(b->b>=c.beta0,temperature_nodes(c))
        for (node,beta) in enumerate(nodes)
            steps=node==1 ? Float64[] : _step_schedule(nodes[node-1],beta,c.delta_tau_peps)
            push!(replay,(;beta,steps=copy(steps)))
            if node>1
                _phase(e) do
                    for p in particles
                        _advance_particle!(p,c,steps;energy=_measure_at(c,beta),
                            progress=r->_particle_progress!(out,p,r;phase=:stochastic,event))
                    end
                end
            end
            elapsed=_wall_time(sum((p.time_seconds for p in particles);init=0.0),e)
            _archive!(particles,out,e);_log_particles!(particles,out,e)
            records=_global_particles(particles,e)
            stats=merge(_statistics(records,c.Ns,length(c.lattice)),(;source=:sampling,time_seconds=elapsed))
            trigger=c.resampling.enabled && stats.eta<c.resampling.eta_th
            _root(()->_log_ensemble(out,stats;phase=trigger ? :pre_resampling : :ordinary),e)
            if trigger
                event+=1
                particles,records,stats,next_sample_id=_resample!(particles,records,cp,c,cache,replay,event,out,e;
                    next_sample_id)
            end
            _root(e) do
                push!(out.data["observables"]["curve"],stats)
                append!(out.data["observables"]["samples"],records)
                periodic_save!(out;progress=Dict("phase"=>"stochastic","beta"=>beta,"event"=>event))
            end
        end
        _phase(()->_finish_output!(out;complete=true),e)
        _rank(e)==0 ? out.data : nothing
    catch exception
        _phase(()->_finish_output!(out;error=sprint(showerror,exception)),e)
        rethrow()
    end
end
