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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "pricer/pricer.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "third_party/absl/log/check.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
MaxHyperRectangularSolution MaxHyperRectangular::GenSolutionFromPointCoverage(
    const std::vector<double>& weights,
    const std::vector<bool>& is_covered,
    const std::vector<int>& forbidden_points,
    bool ignore_perimeter_in_objective,
    bool include_covered_with_zero_weight) const {
  // Check if forbidden points are excluded.
  for (int id : forbidden_points)
    CHECK(!is_covered[id]);
  MaxHyperRectangularSolution solution;
  solution.included_points = is_covered;
  solution.objective_value = 0.0;
  size_t dimension = coords_[0].size();
  solution.limits_solution.resize(dimension);
  for (int dim = 0 ; dim < dimension ; ++dim) {
    solution.limits_solution[dim].first = std::numeric_limits<int64_t>::max();
    solution.limits_solution[dim].second = std::numeric_limits<int64_t>::min();
  }
  for (int id = 0 ; id < coords_.size() ; ++id) {
    if (!is_covered[id])
      continue;
    solution.objective_value += weights[id];
    for (int dim = 0; dim < dimension; ++dim) {
      if (coords_[id][dim] < solution.limits_solution[dim].first) {
        solution.limits_solution[dim].first = coords_[id][dim];
      }
      if (coords_[id][dim] > solution.limits_solution[dim].second) {
        solution.limits_solution[dim].second = coords_[id][dim];
      }
    }
  }

  // If no point included, return 0,0
  if (solution.limits_solution[0].first > solution.limits_solution[0].second) {
    for (int dim = 0 ; dim < dimension ; ++dim) {
      solution.limits_solution[dim].first = 0;
      solution.limits_solution[dim].second = 0;
    }
  }
  if (!ignore_perimeter_in_objective) {
    for (int dim = 0; dim < dimension; ++dim)
      solution.objective_value -= static_cast<double>(solution
          .limits_solution[dim].second - solution.limits_solution[dim].first);
  }
  if (include_covered_with_zero_weight) {
    // Force to include points with weight 0 and covered by the solution.
    for (int id = 0; id < coords_.size(); ++id) {
      if (is_covered[id]) continue;
      if (weights[id] != 0) continue;
      // check if inside the rectangle.
      bool is_inside_rectangle = true;
      for (int dim = 0; dim < dimension; ++dim) {
        if ((coords_[id][dim] > solution.limits_solution[dim].second) ||
            (coords_[id][dim] < solution.limits_solution[dim].first))
          is_inside_rectangle = false;
      }
      // Point is inside rectangle, weight 0 and not covered. We include it.
      if (is_inside_rectangle)
        solution.included_points[id] = true;
    }
    // If forbidden, we delete it.
    for (int id : forbidden_points)
      if (solution.included_points[id])
        solution.included_points[id] = false;
  }

  return solution;
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
