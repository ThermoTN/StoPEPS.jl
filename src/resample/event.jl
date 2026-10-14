function _resampling_plan(records,scores,c)
    relative,_=_aligned_weights(records)
    mass=relative.-maximum(relative).+log.(getfield.(records,:count))
    loo=_loo_log_probabilities(mass,scores)
    p0=getfield.(records,:log_p0)
    # The second-moment objective combines estimator weights and physical norms.
    physical_baseline=getfield.(records,:physical_log_norm).-p0
    risk=mass.+(physical_baseline.-maximum(physical_baseline))
    selection=_select_lambda(risk,loo.-p0,c.resampling)
    stats=_statistics(records,c.Ns,length(c.lattice))
    M=clamp(c.Ns-floor(Int,stats.eta_p*c.Ns),0,c.Ns-1)
    (;selection,M,loo_log_probability=loo,loo_log_ratio=loo.-p0,
        risk_log_amplitude_squared=risk,parent_log_mass=mass,pre_statistics=stats)
end

function _resample_fresh_count(plan,options)
    needs_growth=options.max_resample_sweep>1 && plan.selection.lambda<options.lambda_threshold
    needs_growth ? max(1,plan.M) : plan.M
end
function _resample_stop(lambda,sweep,options)
    lambda>=options.lambda_threshold && return :lambda_threshold
    sweep>=options.max_resample_sweep && return :sweep_limit
    :continue
end
_resample_slot(c,sweep,d)=Base.checked_add(Base.checked_mul(sweep-1,c.Ns),d)

function _evaluate_resampling!(memo,caches,particles,records,c,out,e;event,after_sweep)
    started=_now_seconds()
    _phase(e) do
        for p in particles
            haskey(caches,p.id) && continue
            caches[p.id]=SamplingCache(p.state,_boundary(c,:proposal);kind=:physical,basis=caches[0].basis)
        end
    end
    reference=_score_configurations!(memo,caches,getfield.(records,:id),getfield.(records,:configuration),c,out,e;
        event,sweep=after_sweep,stage="leave_one_out")
    plan=_broadcast(()->_resampling_plan(records,_score_matrix(memo,reference),c),e)
    seconds=_wall_time(_now_seconds()-started,e)
    _root(e) do
        out.main===nothing || @printf(out.main,
            " >> Leave-one-out after sweep %d: lambda = %.10e, time cost %.6f(s)\n",
            after_sweep,plan.selection.lambda,seconds)
    end
    (;after_sweep,plan,score_reference=reference,loo_seconds=seconds)
end

function _pool_selection(old,fresh,c,event)
    pool=vcat(old,fresh)
    relative,scale=_aligned_weights(pool)
    pivot=maximum(relative)
    selection=_sir_select_until_unique(relative.-pivot,getfield.(pool,:count),c.Ns;seed=c.seed,event)
    merge(selection,(;pool,common_log_weight=scale+pivot+selection.common_log_weight,
        pool_statistics=_statistics(pool,c.Ns,length(c.lattice))))
end

function _replay_fresh!(fresh,replay,c,event,out,e;sweep=1)
    records=NamedTuple[]
    _root(e) do
        out.main===nothing || _section(out.main,"Replay fresh samples from beta0 (sweep $sweep)",REPLAY_COLUMNS)
    end
    for (node,segment) in enumerate(replay)
        # Re-evolve through the saved step endpoints. Energy measurements supply
        # the norm at measurement nodes.
        if node>1
            _phase(e) do
                for p in fresh
                    _advance_particle!(p,c,segment.steps;
                        energy=node==length(replay) && _measure_at(c,segment.beta),
                        progress=r->_particle_progress!(out,p,r;phase=:replay,event))
                end
            end
        end
        _archive!(fresh,out,e;phase=:replay);_log_particles!(fresh,out,e;phase=:replay)
        records=_global_particles(fresh,e)
        seconds=_wall_time(sum((p.time_seconds for p in fresh);init=0.0),e)
        _root(e) do
            if !isempty(records)
                stats=merge(_statistics(records,sum(r.count for r in records),length(c.lattice)),
                    (;event,sweep,time_seconds=seconds,source=:fresh_replay))
                push!(out.data["observables"]["replay"],(;event,sweep,statistics=stats,samples=records))
                out.main===nothing || _row(out.main,REPLAY_COLUMNS,(stats.beta,stats.lnZ,stats.lnZ_error,
                    stats.eta,stats.eta_error,stats.max_boundary_entropy,seconds))
            end
            periodic_save!(out;progress=Dict("phase"=>"replay","event"=>event,"sweep"=>sweep,"beta"=>segment.beta))
        end
    end
    records
end

function _draw_resampling_batch!(memo,caches,records,plan,M,c,out,e;event,sweep)
    members=_broadcast(e) do
        [begin
            rng=_event_rng(c.seed,0x4d49585455524543,event,_resample_slot(c,sweep,d))
            rand(rng)<plan.selection.lambda ? _categorical(plan.parent_log_mass,rng) : 0
        end for d in 1:M]
    end
    local_draws=_phase(e) do
        draws=NamedTuple[]
        for d in 1:M
            member=members[d]
            owner=member==0 ? mod(d-1,_nprocs(e)) : records[member].owner
            owner==_rank(e) || continue
            slot=_resample_slot(c,sweep,d)
            rng=_event_rng(c.seed,0x46524553485f424f,event,slot)
            member_id=member==0 ? 0 : records[member].id
            draw=sample_configuration(caches[member_id];rng)
            push!(draws,merge((;index=d,slot,member,member_id,configuration=draw.configuration,
                selected_log_probability=draw.log_probability,
                max_environment_truncation=draw.max_environment_truncation,repairs=draw.repairs),
                hasproperty(draw,:boundary_variational) ? (;boundary_variational=draw.boundary_variational) : (;)))
        end
        draws
    end
    draws=_collect_records(local_draws,e;by=r->r.index)
    _phase(()->_seed_draw_scores!(memo,draws,e),e)
    grouping=_group_configurations(getfield.(draws,:configuration))
    configurations=[draws[d].configuration for d in grouping.first_draw]
    baseline_reference=_score_configurations!(memo,caches,[0],configurations,c,out,e;
        event,sweep,stage="baseline_scores")
    # Score empirical components when the proposal includes them.
    score_reference=plan.selection.lambda==0 ? nothing :
        _score_configurations!(memo,caches,getfield.(records,:id),configurations,c,out,e;
            event,sweep,stage="full_mixture_scores")
    proposals=_broadcast(e) do
        mass_normalizer=_rs_logsumexp(plan.parent_log_mass)
        [begin
            p0=_cached_score(memo,0,column)
            scored=score_reference===nothing ? NamedTuple[] :
                [_cached_score(memo,id,column) for id in score_reference.member_ids]
            pr=score_reference===nothing ? nothing :
                _rs_logsumexp(plan.parent_log_mass.+getfield.(scored,:log_probability))-mass_normalizer
            q=_mixture_log_probability(p0.log_probability,pr,plan.selection.lambda)
            err=max(p0.max_environment_truncation,
                maximum((s.max_environment_truncation for s in scored);init=0.0))
            merge((;log_p0=p0.log_probability,log_pr=pr,log_q=q,max_environment_truncation=err),
                hasproperty(p0,:boundary_variational) ? (;boundary_variational=p0.boundary_variational) : (;))
        end for column in baseline_reference.configuration_indices]
    end
    draws=[merge(d,(;log_probability=proposals[grouping.draw_to_group[i]].log_q,
        max_environment_truncation=max(d.max_environment_truncation,
            proposals[grouping.draw_to_group[i]].max_environment_truncation))) for (i,d) in enumerate(draws)]
    _root(e) do
        _log_draws(out,draws)
        out.data["diagnostics"]["pending_event"]["current_sweep"]=(;sweep,draws,grouping,proposals)
    end
    (;members,draws,grouping,configurations,proposals,baseline_reference,score_reference)
end

function _resample!(particles,records,cp,c,baseline_cache,replay,event,out,e;next_sample_id)
    started=_now_seconds();beta=last(replay).beta
    interval_seconds=_wall_time(sum((p.time_seconds for p in particles);init=0.0),e)
    memo=ResamplingScores()
    caches=Dict{Int,Any}(0=>baseline_cache)
    evaluations=NamedTuple[];sweeps=NamedTuple[]
    _root(e) do
        out.data["diagnostics"]["pending_event"]=Dict("event"=>event,"beta"=>beta,"pre"=>records,
            "lambda_evaluations"=>evaluations,"sweeps"=>sweeps,
            "score_table"=>(;configurations=memo.configurations,entries=memo.entries))
        out.main===nothing || println(out.main,"======================== Resampling ================================")
    end
    evaluation=_evaluate_resampling!(memo,caches,particles,records,c,out,e;event,after_sweep=0)
    push!(evaluations,evaluation)
    plan=evaluation.plan
    M=_resample_fresh_count(plan,c.resampling)
    pool_particles=copy(particles);pool_records=records
    all_fresh=NamedTuple[]
    stop_reason=:continue
    proposal_seconds=0.0;replay_seconds=0.0
    for sweep in 1:c.resampling.max_resample_sweep
        phase_started=_now_seconds()
        _root(e) do
            out.main===nothing || println(out.main," >> Sweep ",sweep,"/",c.resampling.max_resample_sweep,
                ": resampling ",M," samples; lambda = ",plan.selection.lambda)
        end
        batch=_draw_resampling_batch!(memo,caches,pool_records,plan,M,c,out,e;event,sweep)
        proposal_time=_wall_time(_now_seconds()-phase_started,e)
        proposal_seconds+=proposal_time
        stop_reason=M==0 ? :no_fresh_draws : _resample_stop(plan.selection.lambda,sweep,c.resampling)
        # Release proposal environments before re-evolving the final batch.
        stop_reason==:continue || empty!(caches)
        # Reserve Ns sample indices for each fresh batch.
        first_id=next_sample_id
        next_sample_id=Base.checked_add(next_sample_id,c.Ns)
        phase_started=_now_seconds()
        fresh=_phase(e) do
            Particle[_new_particle(cp,c,batch.configurations[u];id=first_id+d-1,generation=event,
                count=batch.grouping.multiplicity[u],log_p0=batch.proposals[u].log_p0,
                log_q=batch.proposals[u].log_q,sampling_error=batch.proposals[u].max_environment_truncation,
                energy=length(replay)==1 && _measure_at(c,beta),basis=baseline_cache.basis)
                for (u,d) in enumerate(batch.grouping.first_draw) if _representative_owner(u,_nprocs(e))==_rank(e)]
        end
        fresh_records=M==0 ? NamedTuple[] : _replay_fresh!(fresh,replay,c,event,out,e;sweep)
        replay_time=_wall_time(_now_seconds()-phase_started,e)
        replay_seconds+=replay_time
        append!(pool_particles,fresh);append!(all_fresh,fresh_records)
        pool_records=vcat(records,all_fresh)
        push!(sweeps,(;sweep,evaluation=length(evaluations),lambda=plan.selection.lambda,M,
            members=batch.members,draws=batch.draws,grouping=batch.grouping,proposals=batch.proposals,
            baseline_reference=batch.baseline_reference,score_reference=batch.score_reference,
            fresh_ids=getfield.(fresh_records,:id),pool_representatives=length(pool_records),
            pool_virtual_samples=sum(r.count for r in pool_records),
            timing=(;proposal_seconds=proposal_time,replay_seconds=replay_time)))
        stop_reason==:continue || break
        evaluation=_evaluate_resampling!(memo,caches,pool_particles,pool_records,c,out,e;event,after_sweep=sweep)
        push!(evaluations,evaluation)
        plan=evaluation.plan
        if plan.selection.lambda>=c.resampling.lambda_threshold
            stop_reason=:lambda_threshold
            break
        end
    end
    empty!(caches)
    selection=_broadcast(()->_pool_selection(records,all_fresh,c,event),e)
    next=_redistribute(pool_particles,selection,e)
    post=_global_particles(next,e)
    stats=merge(_statistics(post,c.Ns,length(c.lattice)),(;source=:sampling,time_seconds=interval_seconds))
    _log_particles!(next,out,e;phase=:post_resampling)
    _root(e) do
        loo_seconds=sum(x.loo_seconds for x in evaluations)
        score_table=(;configurations=memo.configurations,
            entries=sort!(memo.entries;by=r->(r.member_id,r.configuration_index)))
        audit=(;event,beta,M,lambda_evaluations=evaluations,sweeps,stop_reason,score_table,
            pre=records,fresh=all_fresh,selection,post,post_statistics=stats,accepted=true,
            proposal=:finite_chi_psd_sequential_basis,
            timing=(;loo_seconds,proposal_seconds,replay_seconds,total_seconds=_now_seconds()-started))
        push!(out.data["diagnostics"]["events"],audit)
        delete!(out.data["diagnostics"],"pending_event")
        out.main===nothing || println(out.main," >> Resampling stopped: ",stop_reason,
            "; completed sweeps = ",length(sweeps))
        out.main===nothing || _section(out.main,"stochastic PEPS",STO_COLUMNS)
        _log_ensemble(out,stats;phase=:post_resampling)
    end
    next,post,stats,next_sample_id
end
