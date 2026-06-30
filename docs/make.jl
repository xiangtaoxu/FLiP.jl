using Documenter
using FLiP

makedocs(;
    modules=[FLiP],
    authors="Xiangtao Xu <xiangtaoxu@gmail.com>",
    repo="https://github.com/xiangtaoxu/FLiP.jl/blob/{commit}{path}#{line}",
    sitename="FLiP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xiangtaoxu.github.io/FLiP.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo="github.com/xiangtaoxu/FLiP.jl",
    devbranch="master",
)
