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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_CPSAT_PRICER_CPSAT_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_CPSAT_PRICER_CPSAT_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "ortools/sat/cp_model.h"
#include "pricer/pricer.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/statusor.h"
#include "util/operations_research/sat/cp_model.h"
#endif

using ClusterLimits = std::vector<std::pair<int64_t, int64_t>>;

namespace operations_research {
namespace hyperrectangular_clustering {
// Given a set of points with coordinates and a weight, it find the hypercube
// that maximize the sum of weights of points inside the hypercube minus the
// perimeter of the hypercube.

// Struct with variables of the problem
struct MaxHyperRectangularVars {
  std::vector<sat::BoolVar> point_included_var;
  std::vector<std::vector<sat::BoolVar>> point_coord_geq_lb_var;
  std::vector<std::vector<sat::BoolVar>> point_coord_leq_ub_var;
  std::vector<sat::IntVar> lower_bound_vars;
  std::vector<sat::IntVar> upper_bound_vars;
  std::vector<sat::IntVar> length_vars;
  std::vector<sat::IntervalVar> interval_vars;
};

class MaxHyperRectangularCpSat : public MaxHyperRectangular {
 public:
  explicit MaxHyperRectangularCpSat(
      const std::vector<std::vector<int64_t>>& coords,
      bool enforce_enclosed_points = false);

  // Find the max hyper rectangular.
  absl::StatusOr<MaxHyperRectangularSolution> SolveMaxHyperRectangular(
      const std::vector<double>& weights,
      const std::vector<int>& forbidden_points,
      const std::vector<std::pair<int, int>>& points_same_cluster,
      const std::vector<std::pair<int, int>>& points_diff_cluster,
      bool ignore_perimeter_in_objective) override;

 private:
  const std::vector<std::vector<int64_t>>& coords_;
  const std::unique_ptr<sat::CpModelBuilder> cp_model_;
  const MaxHyperRectangularVars vars_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_CPSAT_PRICER_CPSAT_H_
