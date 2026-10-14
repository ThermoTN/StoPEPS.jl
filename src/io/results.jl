module Results

using Dates
using JLD2

export RESULT_SCHEMA_VERSION, load_result, save_result

const RESULT_SCHEMA_VERSION = "2.7.0"

function _string_dict(values::AbstractDict)
    return Dict(string(key) => value for (key, value) in pairs(values))
end

# Atomically replace the destination using a filesystem rename.
function _atomic_replace(source::AbstractString, destination::AbstractString)
    err = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), source, destination)
    err < 0 && Base.uv_error("rename($(repr(source)), $(repr(destination)))", err)
    nothing
end

"""
    save_result(path; method, model, parameters, observables, diagnostics, metadata, overwrite=true)

Atomically save one versioned calculation record in JLD2 format. `observables`
may be a result struct, named tuple, array, or dictionary.
`overwrite=false` creates a new destination using an atomic hardlink.
"""
function save_result(
    path::AbstractString;
    method::AbstractString,
    model,
    parameters::AbstractDict=Dict(),
    observables,
    diagnostics=Dict(),
    metadata::AbstractDict=Dict(),
    overwrite::Bool=true,
)
    endswith(lowercase(path), ".jld2") ||
        throw(ArgumentError("result path must use the .jld2 extension"))
    destination = abspath(path)
    !overwrite && (ispath(destination) || islink(destination)) &&
        throw(ArgumentError("output exists: $destination; use overwrite=true to replace it"))
    directory = dirname(destination)
    mkpath(directory)
    temporary_path, temporary_io = mktemp(directory; cleanup=false)
    close(temporary_io)
    try
        jldopen(temporary_path, "w") do file
            file["schema_version"] = RESULT_SCHEMA_VERSION
            file["created_at"] = string(now(UTC))
            file["method"] = String(method)
            file["model"] = model
            file["parameters"] = _string_dict(parameters)
            file["observables"] = observables
            file["diagnostics"] = diagnostics
            file["metadata"] = merge(
                Dict("julia_version" => string(VERSION)),
                _string_dict(metadata),
            )
        end
        if overwrite
            _atomic_replace(temporary_path, destination)
        else
            # Publish the completed file through a same-directory hardlink.
            hardlink(temporary_path, destination)
            rm(temporary_path)
        end
    catch
        rm(temporary_path; force=true)
        rethrow()
    end
    return destination
end

"""Load a package result as a string-keyed dictionary."""
function load_result(path::AbstractString)
    return jldopen(path, "r") do file
        Dict(String(key) => file[String(key)] for key in keys(file))
    end
end

end
