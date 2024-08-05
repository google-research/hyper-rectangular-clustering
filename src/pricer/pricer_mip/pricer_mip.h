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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MIP_PRICER_MIP_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MIP_PRICER_MIP_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ortools/math_opt/cpp/math_opt.h"
#include "pricer/pricer.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/statusor.h"
#include "util/operations_research/math_opt/cpp/math_opt.h"
#endif

using ClusterLimits = std::vector<std::pair<int64_t, int64_t>>;

namespace operations_research {
namespace hyperrectangular_clustering {

constexpr math_opt::SolverType mip_solver = math_opt::SolverType::kGscip;

// Given a set of points with coordinates and a weight, it find the hypercube
// that maximize the sum of weights of points inside the hypercube minus the
// perimeter of the hypercube.

// Struct with variables of the problem
struct MaxHyperRectangularMIPVars {
  std::vector<math_opt::Variable> point_included_var;
  std::vector<math_opt::Variable> lower_bound_vars;
  std::vector<math_opt::Variable> upper_bound_vars;
};

class MaxHyperRectangularMIP : public MaxHyperRectangular {
 public:
  explicit MaxHyperRectangularMIP(
      const std::vector<std::vector<int64_t>>& coords);

  // Find the max hyper rectangular.
  absl::StatusOr<MaxHyperRectangularSolution> SolveMaxHyperRectangular(
      const std::vector<double>& weights,
      const std::vector<int>& forbidden_points,
      const std::vector<std::pair<int, int>>& points_same_cluster,
      const std::vector<std::pair<int, int>>& points_diff_cluster,
      bool ignore_perimeter_in_objective) override;

 private:
  const std::vector<std::vector<int64_t>>& coords_;
  const std::vector<std::pair<int64_t, int64_t>> minmax_per_dim_;
  const std::unique_ptr<math_opt::Model> mip_model_;
  const MaxHyperRectangularMIPVars vars_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MIP_PRICER_MIP_H_
