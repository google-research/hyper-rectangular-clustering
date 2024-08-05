cc_library(
    name = "pricer",
    srcs = ["pricer.cc"],
    hdrs = ["pricer.h"],
    deps = [
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
    ],
)

cc_test(
    name = "pricer_test",
    srcs = ["pricer_test.cc"],
    deps = [
        ":pricer",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_cpsat",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_maxclosure",
        "//experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_mip",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
    ],
)
