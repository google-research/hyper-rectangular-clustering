// Copyright 2010-2024 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_MODEL_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_MODEL_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/status/statusor.h"
#include "scip/type_cons.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#include "scip/type_var.h"
#include "src/problem_data.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

using ClusterLimits = std::vector<std::pair<int64_t, int64_t>>;
enum class PricerOptions;
enum class BranchingOptions { kRyanFoster, kNone };

bool AbslParseFlag(absl::string_view text, BranchingOptions* branching,
                   std::string* error);
std::string AbslUnparseFlag(BranchingOptions branching);

// Solution structure, including the total objective, the coverage of points
// and each cluster with an associated fractional value for relaxed solution.
// (should be 1.0 for a valid solution).
struct ProblemSolution {
  double half_perimeter;
  std::vector<double> point_coverage;
  std::vector<std::pair<ClusterLimits, double>> cluster_limits;
};

// Solver parameters {
struct SolveParameters {
  PricerOptions pricer;
  BranchingOptions branching = BranchingOptions::kNone;
  int display_verblevel = 3;
  double limits_time = 300.0;
};

class ClusteringMaxHyperRectangular {
 public:
  ClusteringMaxHyperRectangular(
      const std::vector<std::vector<int64_t>>& coords, int max_num_clusters,
      int max_num_outliers, bool relax_coverage_vars = true,
      bool relax_priced_vars = true)
      : coords_(coords),
        max_num_clusters_(max_num_clusters),
        max_num_outliers_(max_num_outliers),
        problem_data_(coords, relax_coverage_vars, relax_priced_vars) {
    FormulateProblemWithTrivialColumns();
  }

  // Solve the problem with current data using different pricers and branching.
  absl::StatusOr<ProblemSolution> Solve(SolveParameters solve_parameters);

 private:
  SCIP_RETCODE FormulateProblemWithTrivialColumns();
  SCIP_RETCODE SetMaxClusteringPricer(PricerOptions pricer_option);
  SCIP_RETCODE SolveProblem();
  ProblemSolution GetSolution();
  const std::vector<std::vector<int64_t>>& coords_;
  const int max_num_clusters_;
  const int max_num_outliers_;
  SCIP* scip_;
  ProbDataMaxHyperRectangular problem_data_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_MODEL_H_
