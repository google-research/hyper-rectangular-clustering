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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_H_

#include <cstdint>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#else
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/statusor.h"
#endif

using ClusterLimits = std::vector<std::pair<int64_t, int64_t>>;

namespace operations_research {
namespace hyperrectangular_clustering {
// Given a set of points with coordinates and a weight, it find the hypercube
// that maximize the sum of weights of points inside the hypercube minus the
// perimeter of the hypercube.

// Struct with a solution of the problem.
struct MaxHyperRectangularSolution {
  ClusterLimits limits_solution;
  std::vector<bool> included_points;
  double objective_value;
};

class MaxHyperRectangular {
 public:
  explicit MaxHyperRectangular(const std::vector<std::vector<int64_t>>& coords)
      : coords_(coords) {}
  virtual ~MaxHyperRectangular() = default;

  // Find the max hyper rectangular.
  virtual absl::StatusOr<MaxHyperRectangularSolution> SolveMaxHyperRectangular(
      const std::vector<double>& weights,
      const std::vector<int>& forbidden_points,
      const std::vector<std::pair<int, int>>& points_same_cluster,
      const std::vector<std::pair<int, int>>& points_diff_cluster,
      bool ignore_perimeter_in_objective) = 0;

  MaxHyperRectangularSolution GenSolutionFromPointCoverage(
      const std::vector<double>& weights, const std::vector<bool>& is_covered,
      const std::vector<int>& forbidden_points,
      bool ignore_perimeter_in_objective,
      bool include_covered_with_zero_weight = false) const;

 private:
  const std::vector<std::vector<int64_t>>& coords_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_H_
