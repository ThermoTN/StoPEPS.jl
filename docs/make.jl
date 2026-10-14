using Documenter, StoPEPS

makedocs(;
    modules=[StoPEPS, StoPEPS.Results],
    sitename="StoPEPS.jl",
    authors="Jianxin-Gao, Chuanshu-Xu",
    format=Documenter.HTML(edit_link="main"),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "User guide" => [
            "Getting started" => "guide/quickstart.md",
            "Applications" => "guide/applications.md",
            "Parameters" => "guide/parameters.md",
            "Reading results" => "guide/results.md",
            "Convergence and uncertainty" => "guide/validation.md",
            "Troubleshooting" => "guide/troubleshooting.md",
        ],
        "Method and conventions" => [
            "Paper and notation" => "design/paper.md",
            "Physical conventions" => "design/physical_conventions.md",
            "Tensor conventions" => "design/tensor_conventions.md",
            "Born sampling" => "design/born_sampling.md",
            "Thermal evolution" => "design/finite_temperature.md",
            "Resampling" => "design/resampling.md",
        ],
        "Library and development" => [
            "Library tutorials" => "guide/library.md",
            "Execution and output" => "design/runtime.md",
            "API reference" => "api.md",
        ],
    ],
)
