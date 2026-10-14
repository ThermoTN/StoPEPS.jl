"""
Thermal PEPO at beta0, preparation parameters, observables and normalization records.
"""
struct ThermalCheckpoint{S<:FinitePEPO}
    state::S
    preparation::Dict{String,Any}
    purification::Vector{NamedTuple}
    history::Vector{NamedTuple}
    run_id::String
end
const CHECKPOINT_FORMAT="stopeps-tensorkit-thermal-1.0.0"

"""Save a beta0 checkpoint; replacing an existing path requires `overwrite=true`."""
function save_checkpoint(path,cp::ThermalCheckpoint;overwrite::Bool=false)
    Results.save_result(path;method="stoPEPS_checkpoint",model=cp.preparation["model"],
        parameters=merge(cp.preparation,Dict("checkpoint_format"=>CHECKPOINT_FORMAT)),
        observables=Dict("sites"=>cp.state.sites,"beta0"=>cp.state.beta,
            "log_amplitude"=>cp.state.log_amplitude,"purification"=>cp.purification),
        diagnostics=Dict("history"=>cp.history),
        metadata=Dict("run_id"=>cp.run_id,"complete"=>true),overwrite)
end
"""
    load_checkpoint(path, config)

Load the beta0 thermal PEPO, checking its model, lattice and basis spaces
against the run configuration.
"""
function load_checkpoint(path,c::RunConfig)
    d=Results.load_result(path)
    d["schema_version"] in ("2.0.0","2.1.0","2.2.0","2.3.0","2.4.0","2.5.0","2.6.0",Results.RESULT_SCHEMA_VERSION) &&
        d["parameters"]["checkpoint_format"]==CHECKPOINT_FORMAT || error("unsupported checkpoint format")
    p=d["parameters"]; o=d["observables"]; current=parameter_dict(c)
    _same_model_parameters(p["model"],current["model"]) && p["lattice"]==current["lattice"] &&
        _same_beta(o["beta0"],c.beta0) || error("checkpoint model, lattice or beta0 mismatch")
    state=FinitePEPO(c.lattice,o["sites"];beta=o["beta0"],log_amplitude=o["log_amplitude"])
    all(A->space(A,3)==physical_space(c.model),state.sites) || error("checkpoint physical space mismatch")
    ThermalCheckpoint(state,Dict{String,Any}(p),NamedTuple[o["purification"]...],
        NamedTuple[d["diagnostics"]["history"]...],String(d["metadata"]["run_id"]))
end

mutable struct RunOutput
    config::RunConfig
    rank::Int
    run_id::String
    data::Dict{String,Any}
    main::Union{Nothing,IOStream}
    sample_ids::Set{Int}
    last_save::Float64
end
_now_seconds()=time_ns()/1e9
function _new_record(c,run_id,stage,nranks)
    Dict{String,Any}("parameters"=>parameter_dict(c),"model"=>parameter_dict(c)["model"],
        "observables"=>Dict{String,Any}("curve"=>NamedTuple[],"purification"=>NamedTuple[],
            "samples"=>NamedTuple[],"replay"=>NamedTuple[]),
        "diagnostics"=>Dict{String,Any}("events"=>NamedTuple[],"history"=>NamedTuple[]),
        "metadata"=>Dict{String,Any}("run_id"=>run_id,"stage"=>String(stage),"complete"=>false,
            "mpi_size"=>nranks,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
            "restart_supported"=>false,"progress"=>Dict{String,Any}(),
            "uncertainty"=>"conditional_particle_delta_method; shared_history_scale_fixed"))
end
function _section(io,title,columns)
    println(io,"======================== ",title," ================================")
    _table_header(io,columns)
end
_column_width(name)=max(length(name),name=="probability" ? 24 : 20)
function _table_header(io,columns)
    header=join((lpad(name,_column_width(name)) for name in columns)," | ")
    println(io,header); println(io,repeat('-',length(header))); flush(io)
end
function _parameters(io,dict,prefix="")
    for key in sort!(collect(keys(dict)))
        value=dict[key]; name=isempty(prefix) ? key : prefix*"."*key
        if value isa AbstractDict
            _parameters(io,value,name)
        else
            println(io,'"',name,"\": ",repr(value))
        end
    end
end
const PUR_COLUMNS=("beta","energy_total","energy_per_site","lnZ","max_PEPO_truncation","max_environment_truncation","E.E.","time(s)")
const STO_COLUMNS=("beta","energy_total","energy_per_site","energy_per_site_error","lnZ","lnZ_error","eta","eta_error","max_PEPS_truncation","max_environment_truncation","E.E.","time(s)")
const SAMPLE_COLUMNS=("beta","energy_total","energy_per_site","log_weight","max_PEPS_truncation","max_environment_truncation","count","E.E.","time(s)")
const REPLAY_COLUMNS=("beta","lnZ","lnZ_error","eta","eta_error","E.E.","time(s)")
const SAMPLE_REPLAY_COLUMNS=("beta","log_weight","max_PEPS_truncation","max_environment_truncation","count","E.E.","time(s)")
const DRAW_COLUMNS=("index","max_environment_truncation","probability")
function _row(io,columns,values)
    cells=(v isa AbstractString ? v : v isa Integer ? string(v) : @sprintf("%.10e",Float64(v)) for v in values)
    println(io,join((lpad(value,_column_width(name)) for (name,value) in zip(columns,cells))," | "))
    flush(io)
end

"""Full-update solver summary for each beta."""
function _log_fu_warnings(out,reports;phase)
    out.main===nothing && return
    for beta in sort!(unique([r.beta for r in reports]))
        steps=filter(r->r.beta==beta,reports)
        capped=sum(r.iteration_limit_hits for r in steps)
        linear=sum(r.linear_unconverged_updates for r in steps)
        capped==0 && linear==0 && continue
        updates=sum(r.bond_updates for r in steps)
        capped_error=capped==0 ? NaN : maximum(r.max_iteration_limit_error for r in steps if r.iteration_limit_hits>0)
        residual=maximum(r.max_solver_residual for r in steps)
        u=out.config.update
        println(out.main,"# WARNING FullUpdate phase=",phase," beta=",@sprintf("%.10e",beta),
            " evolution_steps=",length(steps)," bond_updates=",updates,
            " max_iterations_reached=",capped," linear_unconverged=",linear,
            " max_iterations_reached_error=",@sprintf("%.10e",capped_error),
            " max_solver_residual=",@sprintf("%.10e",residual),
            " max_iterations=",u.max_iterations," error_tolerance=",u.error_tolerance,
            " linear_tolerance=",u.linear_tolerance)
    end
    flush(out.main)
end
function _open_output(c,stage;rank=0,nranks=1,checkpoint=nothing)
    id=checkpoint===nothing ? string(now(UTC),"-",time_ns()) : checkpoint.run_id
    data=_new_record(c,id,stage,nranks); main=nothing
    if checkpoint!==nothing
        append!(data["observables"]["purification"],checkpoint.purification)
        append!(data["observables"]["curve"],[r for r in checkpoint.purification if r.beta<c.beta0 && !_same_beta(r.beta,c.beta0)])
        append!(data["diagnostics"]["history"],checkpoint.history)
        data["metadata"]["checkpoint_preparation"]=checkpoint.preparation
    end
    if c.output_directory!==nothing && rank==0
        directory=abspath(c.output_directory); result=joinpath(directory,"results","result.jld2")
        logpath=joinpath(directory,"log","run.log"); append_log=false
        checkpoint_path=joinpath(directory,"results","checkpoint.jld2")
        if stage==:purification && !c.overwrite_output && (ispath(checkpoint_path) || islink(checkpoint_path))
            error("checkpoint already exists; choose another directory or overwrite_output=true")
        end
        if isfile(result)
            old=Results.load_result(result)
            append_log=stage==:stochastic && get(old["metadata"],"run_id",nothing)==id &&
                get(old["metadata"],"stage",nothing)=="purification" &&
                get(old["metadata"],"stage_complete",false)
            append_log || c.overwrite_output || error("output exists; choose another directory or overwrite_output=true")
        elseif isfile(logpath) && !c.overwrite_output
            error("existing run.log requires completed preparation; select a fresh output directory")
        end
        mkpath(dirname(logpath));mkpath(dirname(result));mkpath(joinpath(directory,"log","samples"))
        main=open(logpath,append_log ? "a" : "w")
        println(main,"============================== Parameters ==================================")
        _parameters(main,parameter_dict(c)); println(main,"\"stage\": ",stage)
        println(main,"\"run_id\": ",id); println(main,"\"mpi_size\": ",nranks)
        println(main,"\"julia_threads\": ",Threads.nthreads());println(main,"\"blas_threads\": ",BLAS.get_num_threads())
        println(main,repeat('=',76)); flush(main)
    end
    out=RunOutput(c,rank,id,data,main,Set{Int}(),_now_seconds())
    periodic_save!(out;force=true)
    out
end

function periodic_save!(out::RunOutput;force=false,progress=nothing)
    out.rank==0 || return
    progress===nothing || (out.data["metadata"]["progress"]=progress)
    now=_now_seconds()
    out.config.output_directory===nothing && return
    force || now-out.last_save>=out.config.save_interval_seconds || return
    path=joinpath(out.config.output_directory,"results","result.jld2")
    d=out.data
    Results.save_result(path;method="stoPEPS",model=d["model"],parameters=d["parameters"],
        observables=d["observables"],diagnostics=d["diagnostics"],metadata=d["metadata"])
    out.last_save=now
end
function _particle_progress!(out,p,r;phase,event=0)
    out.rank==0 && out.config.output_directory!==nothing || return
    _now_seconds()-out.last_save>=out.config.save_interval_seconds || return
    # Save local full-update progress alongside the completed ensemble records.
    out.data["diagnostics"]["in_progress_history"]=(;id=p.id,generation=p.generation,
        phase,history=copy(p.pending_history))
    periodic_save!(out;progress=Dict("phase"=>String(phase),"event"=>event,
        "beta"=>r.beta,"sample_id"=>p.id,"ensemble_complete"=>false))
end
function _log_purification(out,r)
    out.main===nothing && return
    _row(out.main,PUR_COLUMNS,(r.beta,r.energy_total,r.energy_per_site,r.lnZ,r.max_state_truncation,r.max_environment_truncation,r.max_boundary_entropy,r.time_seconds))
end
function _log_ensemble(out,r;phase=:ordinary)
    out.main===nothing && return
    phase==:pre_resampling && println(out.main,"# before resampling")
    phase==:post_resampling && println(out.main,"# after resampling; errors condition on inherited common scale")
    _row(out.main,STO_COLUMNS,(r.beta,r.energy_total,r.energy_per_site,r.energy_per_site_error,r.lnZ,r.lnZ_error,
        r.eta,r.eta_error,r.max_state_truncation,r.max_environment_truncation,r.max_boundary_entropy,r.time_seconds))
end
function _sample_log(out,r;phase=:ordinary)
    out.config.output_directory===nothing && return
    directory=joinpath(out.config.output_directory,"log","samples");mkpath(directory)
    path=joinpath(directory,@sprintf("rank_%03d_sample_%03d.log",out.rank,r.id))
    first=!(r.id in out.sample_ids)
    open(path,first ? "w" : "a") do io
        if first
            println(io,"# run_id=",out.run_id," seed=",r.seed," sample_index=",r.sample_index,
                " generation=",r.generation," configuration=",repr(r.configuration))
            _section(io,"finite-PEPS",phase==:replay ? SAMPLE_REPLAY_COLUMNS : SAMPLE_COLUMNS)
        end
        println(io,"# phase=",phase)
        if phase==:replay
            _row(io,SAMPLE_REPLAY_COLUMNS,(r.beta,r.log_weight,r.max_state_truncation,r.max_environment_truncation,r.count,r.max_boundary_entropy,r.time_seconds))
        else
            first || phase!=:post_resampling || _section(io,"finite-PEPS",SAMPLE_COLUMNS)
            _row(io,SAMPLE_COLUMNS,(r.beta,r.energy_total,r.energy_per_site,r.log_weight,r.max_state_truncation,
                r.max_environment_truncation,r.count,r.max_boundary_entropy,r.time_seconds))
        end
    end
    push!(out.sample_ids,r.id)
end
function _probability_string(logp)
    logp==-Inf && return "0.0000000000e+00"
    exponent=floor(Int,logp/log(10.0))
    mantissa=exp(logp-exponent*log(10.0))
    @sprintf("%.10fe%+d",mantissa,exponent)
end
function _log_draws(out,draws)
    out.main===nothing && return
    _table_header(out.main,DRAW_COLUMNS)
    for r in draws
        _row(out.main,DRAW_COLUMNS,(r.index,r.max_environment_truncation,_probability_string(r.log_probability)))
    end
    flush(out.main)
end
function _finish_output!(out;complete=false,error=nothing)
    if out.rank==0
        out.data["metadata"]["complete"]=complete
        out.data["metadata"]["stage_complete"]=error===nothing
        error===nothing || (out.data["metadata"]["error"]=error)
        periodic_save!(out;force=true)
    end
    out.main===nothing || close(out.main)
end
