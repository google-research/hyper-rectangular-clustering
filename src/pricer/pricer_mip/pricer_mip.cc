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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "ortools/math_opt/cpp/math_opt.h"
#include "pricer/pricer_mip/pricer_mip.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_mip/pricer_mip.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_format.h"
#include "util/operations_research/math_opt/cpp/math_opt.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {

// Return the coordinates of each dimension sorted.
std::vector<std::pair<int64_t, int64_t>> GetMinMaxPerDimension(
    const std::vector<std::vector<int64_t>>& coords) {
  size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  size_t dimension = coords[0].size();
  std::vector<std::pair<int64_t, int64_t>> minmax;
  minmax.reserve(dimension);
  for (size_t dim = 0; dim < dimension; ++dim) {
    const auto [min, max] = std::minmax_element(
        coords.begin(), coords.end(),
        [dim](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
          return a[dim] < b[dim];
        });
    minmax.push_back({min->at(dim), max->at(dim)});
  }
  return minmax;
}

MaxHyperRectangularMIPVars CreateModelVars(
    const std::vector<std::vector<int64_t>>& coords,
    const std::vector<std::pair<int64_t, int64_t>>& minmax_per_dim,
    const std::unique_ptr<math_opt::Model>& mip_model) {
  size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  size_t dimension = coords[0].size();
  MaxHyperRectangularMIPVars vars;
  // Boolean variable indicating if a point is included in the hyperrectangle.
  vars.point_included_var.reserve(num_points);
  for (size_t id = 0; id < num_points; ++id) {
    vars.point_included_var.push_back(mip_model->AddBinaryVariable());
  }
  // LB and UB variable with the coordinates of the hyperrectangle.
  vars.lower_bound_vars.reserve(dimension);
  vars.upper_bound_vars.reserve(dimension);
  for (size_t dim = 0; dim < dimension; ++dim) {
    vars.lower_bound_vars.push_back(mip_model->AddContinuousVariable(
        static_cast<double>(minmax_per_dim[dim].first),
        static_cast<double>(minmax_per_dim[dim].second)));
    vars.upper_bound_vars.push_back(mip_model->AddContinuousVariable(
        static_cast<double>(minmax_per_dim[dim].first),
        static_cast<double>(minmax_per_dim[dim].second)));
  }
  return vars;
}

void AddModelConstraints(
    const std::vector<std::vector<int64_t>>& coords,
    const std::vector<std::pair<int64_t, int64_t>>& minmax_per_dim,
    const MaxHyperRectangularMIPVars& vars,
    const std::unique_ptr<math_opt::Model>& mip_model) {
  const size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  const size_t dimension = coords[0].size();
  for (int dim = 0; dim < dimension; ++dim) {
    mip_model->AddLinearConstraint(vars.lower_bound_vars[dim] <=
                                   vars.upper_bound_vars[dim]);
  }
  for (int point = 0; point < num_points; ++point) {
    for (int dim = 0; dim < dimension; ++dim) {
      mip_model->AddLinearConstraint(
          vars.lower_bound_vars[dim] +
              (static_cast<double>(minmax_per_dim[dim].second -
                                   coords[point][dim])) *
                  vars.point_included_var[point] <=
          static_cast<double>(minmax_per_dim[dim].second));
      mip_model->AddLinearConstraint(
          vars.upper_bound_vars[dim] +
              (static_cast<double>(minmax_per_dim[dim].first -
                                   coords[point][dim])) *
                  vars.point_included_var[point] >=
          static_cast<double>(minmax_per_dim[dim].first));
    }
  }
}

}  // namespace

MaxHyperRectangularMIP::MaxHyperRectangularMIP(
    const std::vector<std::vector<int64_t>>& coords)
    : MaxHyperRectangular(coords),
      coords_(coords),
      minmax_per_dim_(GetMinMaxPerDimension(coords_)),
      mip_model_(std::make_unique<math_opt::Model>()),
      vars_(CreateModelVars(coords_, minmax_per_dim_, mip_model_)) {
  AddModelConstraints(coords_, minmax_per_dim_, vars_, mip_model_);
}

absl::StatusOr<MaxHyperRectangularSolution>
MaxHyperRectangularMIP::SolveMaxHyperRectangular(
    const std::vector<double>& weights,
    const std::vector<int>& forbidden_points,
    const std::vector<std::pair<int, int>>& points_same_cluster,
    const std::vector<std::pair<int, int>>& points_diff_cluster,
    bool ignore_perimeter_in_objective) {
  if (weights.size() != coords_.size())
    return absl::InvalidArgumentError(absl::StrFormat(
        "weights and coords must have the same size. (%d vs %d)",
        weights.size(), coords_.size()));
  // Set objective of the problem.
  math_opt::LinearExpression objective;
  size_t dimension = coords_[0].size();
  for (int id = 0; id < weights.size(); ++id) {
    objective += weights[id] * vars_.point_included_var[id];
  }
  if (!ignore_perimeter_in_objective) {
    for (int dim = 0; dim < dimension; ++dim) {
      objective -= (vars_.upper_bound_vars[dim] - vars_.lower_bound_vars[dim]);
    }
  }
  mip_model_->Maximize(objective);
  // Set forbidden points.
  for (int point : forbidden_points)
    mip_model_->set_upper_bound(vars_.point_included_var[point], 0.0);
  // Include locally added constraints for pair of points.
  std::vector<math_opt::LinearConstraint> locally_added_constraints;
  locally_added_constraints.reserve(points_same_cluster.size() +
                                    points_diff_cluster.size());
  for (auto [point1, point2] : points_same_cluster) {
    locally_added_constraints.push_back(mip_model_->AddLinearConstraint(
        vars_.point_included_var[point1] == vars_.point_included_var[point2]));
  }
  for (auto [point1, point2] : points_diff_cluster) {
    locally_added_constraints.push_back(mip_model_->AddLinearConstraint(
        vars_.point_included_var[point1] + vars_.point_included_var[point2] <=
        1.0));
  }

  // Solve problem using mip solver.
  const absl::StatusOr<math_opt::SolveResult> response =
      math_opt::Solve(*mip_model_, mip_solver);
  // Clear locally added constraints.
  for (int point : forbidden_points)
    mip_model_->set_upper_bound(vars_.point_included_var[point], 1.0);
  for (math_opt::LinearConstraint constraint : locally_added_constraints)
    mip_model_->DeleteLinearConstraint(constraint);
  if (!response.ok()) return response.status();
  if (!response->termination.IsOptimalOrFeasible()) {
    return absl::NotFoundError("Infeasible problem");
  }
  if (!response->termination.IsOptimal()) {
    return absl::InternalError("Not optimal solution found.");
  }
  std::vector<bool> is_included(coords_.size());
  for (int id = 0; id < coords_.size(); ++id) {
    is_included[id] =
        (response->variable_values().at(vars_.point_included_var[id]) > 0.5);
  }
  const bool include_covered_with_zero_weight =
      (forbidden_points.empty() && points_same_cluster.empty() &&
       points_diff_cluster.empty())
          ? true
          : false;
  return GenSolutionFromPointCoverage(weights, is_included, forbidden_points,
                                      ignore_perimeter_in_objective,
                                      include_covered_with_zero_weight);
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
