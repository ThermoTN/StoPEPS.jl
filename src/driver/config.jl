"""
    ResamplingOptions(; enabled=true, eta_th=0.7, lambda_max=0.999,
        lambda_tolerance=1e-8, lambda_threshold=0.2, max_resample_sweep=3,
        score_batch_size=32)

Controls for annealed Born resampling. `eta_th` triggers resampling;
`lambda_max` and `lambda_tolerance` set the mixture optimization interval and
accuracy. `lambda_threshold` and `max_resample_sweep` control fresh batches.
"""
Base.@kwdef struct ResamplingOptions
    enabled::Bool = true
    eta_th::Float64 = 0.7
    lambda_max::Float64 = 0.999
    lambda_tolerance::Float64 = 1e-8
    lambda_threshold::Float64 = 0.2
    max_resample_sweep::Int = 3
    score_batch_size::Int = 32
end

"""
    RunConfig(; kwargs...)

Configuration for thermal PEPO preparation and stochastic PEPS evolution.
Fields specify the lattice and model, beta grids, tensor and boundary bond
dimensions, full-update controls, sampling basis, ensemble size and output.
`beta0` sets the Born sampling inverse temperature; factor time is beta/2.
"""
Base.@kwdef struct RunConfig
    lattice::Lattice = Lattice(2,2)
    model::AbstractModel = TFIM()
    beta0::Float64 = 0.1
    beta_log::Vector{Float64} = collect(0.0:0.05:0.3)
    beta_measure::Vector{Float64} = [0.0,0.1,0.2,0.3]
    delta_tau_pur::Float64 = 0.025
    delta_tau_peps::Float64 = 0.025
    D_pur::Int = 4
    D_peps::Int = 4
    bond_space_pur::Union{Nothing,ElementarySpace} = nothing
    bond_space_peps::Union{Nothing,ElementarySpace} = nothing
    chi_environment::Int = 32
    chi_sampling::Int = chi_environment
    chi_measurement::Int = chi_environment
    chi_proposal::Int = chi_sampling
    environment_space::Union{Nothing,ElementarySpace} = nothing
    sampling_space::Union{Nothing,ElementarySpace} = environment_space
    measurement_space::Union{Nothing,ElementarySpace} = environment_space
    proposal_space::Union{Nothing,ElementarySpace} = sampling_space
    sampling_basis::Union{Symbol,OrthonormalBasis,SymmetryBasis,AbstractMatrix} = :auto
    trotter_order::Int = 2
    update::UpdateMethod = FullUpdate()
    contraction::Type{<:ContractionMethod} = BoundaryMPS
    max_sweep::Int = 4
    threshold_sweep::Float64 = 1e-8
    target_cache_bytes::Int = 256*1024^2
    record_boundary_entropy::Bool = true
    Ns::Int = 16
    seed::Int = 20260924
    resampling::ResamplingOptions = ResamplingOptions()
    save_interval_seconds::Float64 = 3600.0
    output_directory::Union{Nothing,String} = nothing
    overwrite_output::Bool = false
end
_same_beta(a,b)=abs(a-b)<=64eps(Float64)*max(1.0,abs(a),abs(b))
_measure_at(c,beta)=any(b->_same_beta(b,beta),c.beta_measure)
"""
Return the sorted output beta grid from beta_log, beta_measure, 0 and beta0.
"""
function temperature_nodes(c::RunConfig)
    nodes=sort([0.0;c.beta0;c.beta_log;c.beta_measure]); out=Float64[]
    for b in nodes
        b=_same_beta(b,c.beta0) ? c.beta0 : b
        isempty(out) || !_same_beta(out[end],b) || continue
        push!(out,b)
    end
    out
end
"""
Validate numerical controls, model spaces and sampling basis; return the configuration.
"""
function validate_config(c::RunConfig)
    if c.contraction<:BoundaryMPSDiagonalVariational
        BoundaryMPSDiagonalVariational(c.chi_environment,c.environment_space;
            max_sweep=c.max_sweep,threshold_sweep=c.threshold_sweep,target_cache_bytes=c.target_cache_bytes)
    elseif c.contraction<:BoundaryMPSRowVariational
        BoundaryMPSRowVariational(c.chi_environment,c.environment_space;
            max_sweep=c.max_sweep,threshold_sweep=c.threshold_sweep)
    end
    all(b->isfinite(b)&&b>=0,[c.beta0;c.beta_log;c.beta_measure]) || error("invalid beta grid")
    all(x->isfinite(x)&&x>0,(c.delta_tau_pur,c.delta_tau_peps)) || error("delta_tau must be positive")
    minimum((c.D_pur,c.D_peps,c.chi_environment,c.chi_sampling,c.chi_measurement,c.chi_proposal,c.Ns))>=1 ||
        error("dimensions and sample count must be positive")
    c.seed>=0 && c.trotter_order in (1,2) || error("invalid seed or Trotter order")
    isfinite(c.save_interval_seconds) && c.save_interval_seconds>=0 || error("invalid save interval")
    r=c.resampling
    0<=r.eta_th<=1 && 0<=r.lambda_max<1 && r.lambda_tolerance>0 &&
        isfinite(r.lambda_threshold) && 0<=r.lambda_threshold<1 &&
        r.max_resample_sweep>=1 && r.score_batch_size>0 || error("invalid resampling options")
    r.enabled && c.Ns<2 && error("resampling needs Ns >= 2")
    if c.update isa FullUpdate
        u=c.update
        u.max_iterations>=0 && u.linear_max_iterations>0 &&
            all(x->isfinite(x)&&x>=0,(u.error_tolerance,u.diagonal_shift,u.linear_tolerance)) ||
            error("invalid full-update options")
    end
    c.model isa TFIM && !all(isfinite,(c.model.J,c.model.h)) && error("nonfinite model parameter")
    P=physical_space(c.model)
    for V in (c.bond_space_pur,c.bond_space_peps,c.environment_space,c.sampling_space,c.measurement_space,c.proposal_space)
        V===nothing && continue
        sectortype(V)===sectortype(P) && !isdual(V) && dim(V)>0 ||
            throw(ArgumentError("sector budgets must be nondual, nonempty spaces with the model symmetry"))
    end
    _basis(c)
    c
end

"""Encode effective controls as numerical values and ordinary containers."""
function parameter_dict(c::RunConfig)
    encode(x::Number)=x
    encode(x::AbstractString)=x
    encode(x::Symbol)=String(x)
    encode(x::Nothing)=nothing
    encode(x::Type)=string(nameof(x))
    encode(x::AbstractArray)=encode.(x)
    encode(x::ElementarySpace)=_space_parameters(x)
    encode(x)=merge(Dict("type"=>string(nameof(typeof(x)))),
        Dict(string(k)=>encode(getfield(x,k)) for k in fieldnames(typeof(x))))
    data=Dict(string(k)=>encode(getfield(c,k)) for k in fieldnames(RunConfig))
    data["sampling_basis"]=_basis_parameters(_basis(c))
    diagonal=c.contraction<:AbstractDiagonalBoundaryMPS
    variational=c.contraction<:BoundaryMPSDiagonalVariational
    row_variational=c.contraction<:BoundaryMPSRowVariational
    data["contraction"]=row_variational ? "BoundaryMPS-Row-Variational" :
        variational ? "BoundaryMPS-Diagonal-Variational" : diagonal ? "BoundaryMPS-Diagonal" :
        c.contraction<:BoundaryMPS ? "BoundaryMPS-Row" : data["contraction"]
    data["sampling_site_order"]=[Tuple(i) for i in (diagonal ? _diagonal_order(c.lattice) : site_order(c.lattice))]
    data["contraction_order"]=diagonal ? "x+y_ascending; even_diagonal_x_ascending; odd_diagonal_x_descending" : "row_major"
    data["state_truncation_metric"]="relative_squared_FU_residual"
    if c.update isa FullUpdate
        data["full_update_algorithm"]="environment_gauge_SVD_ALS"
        data["full_update_algorithm_revision"]=_FU_ALGORITHM_REVISION
        data["full_update_gauge_floor"]="max(sqrt(eps(real_scalar_type)),metric_dimension*eps(real_scalar_type))*spectral_scale; auxiliary_metric_only"
        data["full_update_cost_evaluation"]="original_coordinates_and_full_metric; full_hole_PSD_repair_on_residual_failure"
        data["full_update_linear_system"]="environment_and_internal_gauge_coordinates; relative_diagonal_shift_and_CG_residual"
        data["full_update_internal_gauge"]="QR_after_left_solve; LQ_after_right_solve; best_center_SVD_without_truncation"
    end
    data["environment_truncation_metric"]="canonical_SVD_relative_discarded_weight"
    data["boundary_entropy_convention"]="pretruncation_Schmidt_entropy; nats; sector_dimension_weighted"
    data["boundary_entropy_log_scope"]="maximum_over_FU_boundary_cuts_in_output_interval"
    data["boundary_entropy_profile_scope"]=diagonal ?
        "energy_measurement_boundaries; Se.northwest[x,y] and Se.southeast[x,y]; local_two_output_split; NaN_for_undefined_split_entropy" :
        "energy_measurement_boundaries; Se.north[y,b] and Se.south[y,b]; cut_after_row_y_and_column_b"
    if diagonal && !variational
        data["diagonal_compression"]="whole_layer_ket_bra_blocks; untruncated_block_QR_LQ; opposite_direction_SVD"
        data["boundary_entropy_profile_scope"]*="; after_whole_layer_absorption"
    end
    if variational
        data["environment_truncation_metric"]="relative_squared_diagonal_layer_fit_residual"
        data["boundary_entropy_convention"]="pretruncation_variational_two_site_projected_entropy; nats; sector_dimension_weighted"
        data["boundary_entropy_profile_scope"]="accepted_sweep_last_visit; Se.northwest_cuts and Se.southeast_cuts; layer,left_edge,right_edge"
        data["variational_sweep_convention"]="round_trip_from_nearest_endpoint; relative_squared_norm_change_from_last_retained_spectrum; compare_from_second_sweep; zero_threshold_disables_early_stop"
        data["variational_residual_evaluation"]="one_minus_retained_norm_squared_after_projection; overlap_for_unprojected_candidates; roundoff_negative_clipped"
        data["variational_initialization"]="previous_canonical_frontier; geometric_edge_matching; deterministic_space_maps"
        data["variational_recovery"]="best_candidate_or_whole_layer_QR_SVD_on_numerical_failure"
    elseif row_variational
        delete!(data,"target_cache_bytes") # Target-block caching applies to diagonal variational contraction.
        data["environment_truncation_metric"]="variational_two_site_projected_relative_discarded_weight"
        data["boundary_entropy_convention"]="pretruncation_variational_two_site_projected_entropy; nats; sector_dimension_weighted"
        data["boundary_entropy_profile_scope"]="accepted_candidate_last_visit; Se.north[y,b] and Se.south[y,b]; NaN_for_undefined_accepted_spectrum"
        data["variational_sweep_convention"]="column_1_to_Lx_and_back_for_both_sides; relative_squared_norm_change_from_last_retained_spectrum; compare_from_second_sweep; zero_threshold_disables_early_stop"
        data["variational_residual_evaluation"]="local_projected_SVD_diagnostic; projected_objective_minus_retained_norm_squared"
        data["variational_initialization"]="previous_canonical_row; deterministic_space_maps; mixed_overlap_amplitude_fit"
        data["variational_recovery"]="best_valid_candidate"
    end
    data["uncertainty"]="conditional_particle_delta_method; shared_history_scale_fixed"
    data["rng_algorithm"]="Random.MersenneTwister"
    data["sampling_rng_scheme_version"]=SAMPLING_RNG_SCHEME_VERSION
    data["resampling_rng_scheme_version"]=1
    data["sir_scheme_version"]=2
    data["sir_draw_mode"]="with_replacement_next_unique_drop_trigger_capped"
    data["resampling_sweep_convention"]="first_batch_included; fixed_initial_M; cumulative_pool_LOO; one_final_SIR"
    data["resampling_rng_convention"]="event_and_slot; slot=(sweep-1)*Ns+draw; Ns_IDs_reserved_per_sweep"
    data["factor_time_convention"]="tau=beta/2"
    data["temperature_nodes"]=temperature_nodes(c)
    data
end

_space_parameters(P)=Dict("sector_type"=>sprint(show,sectortype(P);context=:module=>TensorKit),"dual"=>isdual(P),
    "sectors"=>[Dict("sector"=>string(c),"multiplicity"=>dim(P,c)) for c in sectors(P)])
function _same_model_parameters(a,b)
    # Canonical dense spin-1/2 space for TFIM parameters.
    function canonical(data)
        normalized=get(data,"type",nothing)=="TFIM" && !haskey(data,"physical_space") ?
            merge(data,Dict("physical_space"=>_space_parameters(ComplexSpace(2)))) : data
        haskey(normalized,"physical_space") || return normalized
        physical=normalized["physical_space"]
        name=replace(physical["sector_type"],"TensorKitSectors."=>"","TensorKit."=>"")
        merge(normalized,Dict("physical_space"=>merge(physical,Dict("sector_type"=>name))))
    end
    canonical(a)==canonical(b)
end
