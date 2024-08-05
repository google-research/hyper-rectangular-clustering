cc_library(
    name = "pricer_mip",
    srcs = ["pricer_mip.cc"],
    hdrs = ["pricer_mip.h"],
    deps = [
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:str_format",
        "//util/operations_research/math_opt/cpp:math_opt",
        "//util/operations_research/math_opt/solvers:gscip_solver",
    ],
)
