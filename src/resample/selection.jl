function _rs_logsumexp(x)
    isempty(x) && return -Inf
    pivot = maximum(x)
    pivot == -Inf && return -Inf
    isfinite(pivot) || throw(ArgumentError("nonfinite positive log mass"))
    return pivot + log(sum(v -> exp(v - pivot), x))
end

function _mixture_log_probability(log_p0, log_pr, lambda)
    0 <= lambda < 1 || throw(ArgumentError("invalid resampling mixture coefficient"))
    lambda == 0 && return Float64(log_p0)
    return _rs_logsumexp((log1p(-lambda) + log_p0, log(lambda) + log_pr))
end


"""Leave-one-sample-out proposal probabilities with multiplicities."""
function _loo_log_probabilities(log_mass, scores)
    K = length(log_mass)
    size(scores) == (K, K) || throw(DimensionMismatch("LOO scores must be K by K"))
    return [begin
        others = [k for k in 1:K if k != j]
        _rs_logsumexp([log_mass[k] + scores[k, j] for k in others]) -
            (isempty(others) ? 0.0 : _rs_logsumexp(log_mass[others]))
    end for j in 1:K]
end

"""Minimize the estimated second moment over the resampling mixture coefficient."""
function _select_lambda(log_amplitude_squared, log_ratio, config::ResamplingOptions)
    length(log_amplitude_squared) == length(log_ratio) > 0 ||
        throw(DimensionMismatch("invalid LOO arrays"))
    all(isfinite, log_amplitude_squared) &&
        all(v -> isfinite(v) || v == -Inf, log_ratio) ||
        throw(ArgumentError("invalid LOO risk or probability ratio"))
    offset = maximum(log_amplitude_squared)
    amplitude = log_amplitude_squared .- offset
    risk(lambda) = _rs_logsumexp([amplitude[j] -
        _mixture_log_probability(0.0, log_ratio[j], lambda) for j in eachindex(amplitude)])
    if length(amplitude) == 1 || config.lambda_max == 0
        return (; lambda=0.0, log_risk_offset=offset, log_risk_zero=risk(0.0),
            log_risk_selected=risk(0.0), log_risk_upper=risk(config.lambda_max),
            fallback=length(amplitude) == 1 ? :single_lineage : :zero_interval)
    end
    left, right = 0.0, config.lambda_max
    golden = (sqrt(5.0) - 1) / 2
    x1, x2 = right - golden * (right - left), left + golden * (right - left)
    f1, f2 = risk(x1), risk(x2)
    while right - left > config.lambda_tolerance
        if f1 <= f2
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = risk(x1)
        else
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = risk(x2)
        end
    end
    candidates = sort([0.0, (left + right) / 2, config.lambda_max])
    risks = risk.(candidates)
    # Prefer the smaller coefficient when objective values agree within roundoff.
    best = minimum(risks)
    tie_tolerance = 64eps(max(1.0,maximum(abs,risks)))
    index = findfirst(r -> r <= best + tie_tolerance,risks)
    return (; lambda=candidates[index], log_risk_offset=offset, log_risk_zero=first(risks),
        log_risk_selected=risks[index], log_risk_upper=last(risks), fallback=:none)
end

function _categorical(log_mass, rng)
    z = _rs_logsumexp(log_mass)
    isfinite(z) || throw(ArgumentError("proposal requires positive total probability"))
    u, cumulative = rand(rng), 0.0
    last_positive = 0
    for j in eachindex(log_mass)
        log_mass[j] == -Inf && continue
        last_positive = j
        cumulative += exp(log_mass[j] - z)
        u < cumulative && return j
    end
    return last_positive
end


"""Select samples from the combined ensemble using importance weights."""
function _sir_select_until_unique(log_weight, input_multiplicity, Ns; seed, event,
        rng_for_draw=slot -> _event_rng(seed, 0x5349525f534c4f54, event, slot))
    Ns isa Integer && Ns >= 1 || throw(ArgumentError("invalid representative target"))
    length(log_weight) == length(input_multiplicity) > 0 ||
        throw(DimensionMismatch("empty or inconsistent SIR pool"))
    all(isfinite, log_weight) || throw(ArgumentError("pool weights must be finite"))
    all(m -> m isa Integer && m > 0, input_multiplicity) ||
        throw(ArgumentError("pool multiplicities must be positive integers"))
    K_pool, Nv_pool = length(log_weight), sum(input_multiplicity)
    ids = Int[]
    if K_pool <= Ns
        return (; selected_candidate=ids, multiplicity=copy(input_multiplicity),
            active=collect(eachindex(log_weight)), common_log_weight=0.0,
            output_relative_log_weight=copy(log_weight), draw_count=0, discarded_candidate=0,
            stop_reason=:pool_within_target, K_pool, Nv_pool, Nv_post=Nv_pool)
    end
    mass = log_weight .+ log.(input_multiplicity)
    multiplicity = zeros(Int,K_pool)
    distinct = 0; draw_count = 0; discarded_candidate = 0
    for slot in 1:3Ns
        id = _categorical(mass,rng_for_draw(slot))
        draw_count = slot
        # Check before retaining the hit, including on the final allowed draw.
        if distinct == Ns && iszero(multiplicity[id])
            discarded_candidate = id
            break
        end
        push!(ids,id)
        distinct += iszero(multiplicity[id])
        multiplicity[id] += 1
    end
    return (; selected_candidate=ids, multiplicity, active=findall(>(0),multiplicity),
        common_log_weight=_rs_logsumexp(mass)-log(Nv_pool),
        output_relative_log_weight=zeros(K_pool), draw_count, discarded_candidate,
        stop_reason=discarded_candidate == 0 ? :draw_limit : :next_distinct,
        K_pool, Nv_pool, Nv_post=length(ids))
end

function _group_configurations(configurations)
    first_draw = Int[]
    multiplicity = Int[]
    draw_to_group = Int[]
    lookup = Dict{Tuple,Int}()
    for (d, x) in enumerate(configurations)
        key = Tuple(vec(x))
        j = get(lookup, key, 0)
        if j == 0
            push!(first_draw, d)
            push!(multiplicity, 0)
            j = length(first_draw)
            lookup[key] = j
        end
        multiplicity[j] += 1
        push!(draw_to_group, j)
    end
    return (; first_draw, multiplicity, draw_to_group)
end
