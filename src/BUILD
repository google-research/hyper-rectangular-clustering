cc_library(
    name = "problem_model",
    srcs = [
        "branching_handler.cc",
        "pricer_scip.cc",
        "problem_data.cc",
        "problem_model.cc",
        "same_or_diff_constraint_handler.cc",
    ],
    hdrs = [
        "branching_handler.h",
        "pricer_scip.h",
        "problem_data.h",
        "problem_model.h",
        "same_or_diff_constraint_handler.h",
    ],
    deps = [
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_cpsat",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_maxclosure",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_mip",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/scip:libscip_with_glop",
    ],
)

cc_test(
    name = "problem_model_test",
    srcs = ["problem_model_test.cc"],
    deps = [
        ":problem_model",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status:statusor",
        "//third_party/scip:libscip_with_glop",
    ],
)

cc_binary(
    name = "problem_main",
    srcs = ["problem_main.cc"],
    deps = [
        ":problem_model",
        "//base",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/flags:usage",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
    ],
)
