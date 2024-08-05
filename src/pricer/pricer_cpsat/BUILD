cc_library(
    name = "pricer_cpsat",
    srcs = ["pricer_cpsat.cc"],
    hdrs = ["pricer_cpsat.h"],
    deps = [
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer",
        "//third_party/absl/algorithm:container",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:str_format",
        "//util/operations_research/sat:cp_model",
        "//util/operations_research/sat:cp_model_cc_proto",
        "//util/operations_research/sat:cp_model_solver",
        "//util/operations_research/sat:sat_parameters_cc_proto",
        "//util/operations_research/util:sorted_interval_list",
    ],
)
