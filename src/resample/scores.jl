"""Configuration probabilities cached for one resampling event.

Sample IDs and configuration columns identify the entries. Member zero denotes
the beta0 Born distribution. Rank zero collects numerical scores.
"""
struct ResamplingScores
    configurations::Vector{Matrix{Int}}
    columns::Dict{Tuple,Int}
    local_known::Set{Tuple{Int,Int}}
    entries::Vector{NamedTuple}
    index::Dict{Tuple{Int,Int},Int}
end
ResamplingScores()=ResamplingScores(Matrix{Int}[],Dict{Tuple,Int}(),
    Set{Tuple{Int,Int}}(),NamedTuple[],Dict{Tuple{Int,Int},Int}())

function _score_columns!(memo,configurations)
    [get!(memo.columns,Tuple(vec(x))) do
        push!(memo.configurations,copy(x))
        length(memo.configurations)
    end for x in configurations]
end
function _score_entry(member_id,column,result;source)
    merge((;member_id,configuration_index=column,log_probability=result.log_probability,
        max_environment_truncation=result.max_environment_truncation,repairs=result.repairs,source),
        hasproperty(result,:boundary_variational) ? (;boundary_variational=result.boundary_variational) : (;))
end
function _record_score!(memo,entry)
    key=(entry.member_id,entry.configuration_index)
    haskey(memo.index,key) && return
    push!(memo.entries,entry)
    memo.index[key]=length(memo.entries)
end
_cached_score(memo,id,column)=memo.entries[memo.index[(id,column)]]
_score_matrix(memo,reference)=[_cached_score(memo,id,column).log_probability
    for id in reference.member_ids,column in reference.configuration_indices]

"""Evaluate uncached probabilities with baseline columns distributed across ranks."""
function _score_configurations!(memo,caches,member_ids,configurations,c,out,e;event,sweep,stage)
    columns=_score_columns!(memo,configurations)
    unique_columns=unique(columns)
    for first_index in 1:c.resampling.score_batch_size:length(unique_columns)
        indices=first_index:min(length(unique_columns),first_index+c.resampling.score_batch_size-1)
        local_scores=_phase(e) do
            entries=NamedTuple[]
            for id in member_ids, column in unique_columns[indices]
                if id==0
                    mod(column-1,_nprocs(e))==_rank(e) || continue
                else
                    haskey(caches,id) || continue
                end
                key=(id,column)
                key in memo.local_known && continue
                r=sample_configuration(caches[id];configuration=memo.configurations[column])
                push!(memo.local_known,key)
                push!(entries,_score_entry(id,column,r;source=:score))
            end
            entries
        end
        parts=_gather(local_scores,e)
        _root(e) do
            entries=sort!(reduce(vcat,parts);by=r->(r.member_id,r.configuration_index))
            foreach(r->_record_score!(memo,r),entries)
            # Validate the distributed probability cache before mixture evaluation.
            all(haskey(memo.index,(id,column)) for id in member_ids for column in unique_columns[indices]) ||
                error("missing proposal scores")
            periodic_save!(out;progress=Dict("phase"=>stage,"event"=>event,
                "sweep"=>sweep,"scored"=>last(indices)))
        end
    end
    (;member_ids=copy(member_ids),configuration_indices=columns)
end

"""A draw already evaluated its generating component, including diagnostics."""
function _seed_draw_scores!(memo,draws,e)
    columns=_score_columns!(memo,getfield.(draws,:configuration))
    for (d,column) in zip(draws,columns)
        id=d.member_id
        # Every rank learns baseline hits: its later scoring owner may differ
        # from the rank that drew it. Physical owners stay fixed in an event.
        push!(memo.local_known,(id,column))
        if _rank(e)==0
            result=merge(d,(;log_probability=d.selected_log_probability))
            _record_score!(memo,_score_entry(id,column,result;source=:draw))
        end
    end
    columns
end
