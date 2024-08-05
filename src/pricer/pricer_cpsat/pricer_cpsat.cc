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
#include <memory>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"
#include "pricer/pricer_cpsat/pricer_cpsat.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_cpsat/pricer_cpsat.h"
#include "third_party/absl/algorithm/container.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_format.h"
#include "util/operations_research/sat/cp_model.h"
#include "util/operations_research/sat/cp_model_solver.h"
#include "util/operations_research/util/sorted_interval_list.h"

#endif

namespace operations_research {
namespace hyperrectangular_clustering {

using ::operations_research::sat::BoolVar;
using ::operations_research::sat::Constraint;
using ::operations_research::sat::CpModelBuilder;
using ::operations_research::sat::CpSolverResponse;
using ::operations_research::sat::CpSolverStatus;
using ::operations_research::sat::DoubleLinearExpr;
using ::operations_research::sat::IntervalVar;

namespace {

// Return the coordinates of each dimension sorted.
std::vector<std::vector<int64_t>> GetSortedCoordsPerDimension(
    const std::vector<std::vector<int64_t>>& coords) {
  size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  size_t dimension = coords[0].size();
  std::vector<std::vector<int64_t>> coords_sorted(dimension);
  for (size_t dim = 0; dim < dimension; ++dim)
    coords_sorted[dim].reserve(num_points);
  for (const std::vector<int64_t>& coords_point : coords) {
    for (size_t dim = 0; dim < dimension; ++dim)
      coords_sorted[dim].push_back(coords_point[dim]);
  }
  for (size_t dim = 0; dim < dimension; ++dim) absl::c_sort(coords_sorted[dim]);
  return coords_sorted;
}

MaxHyperRectangularVars CreateModelVars(
    const std::vector<std::vector<int64_t>>& coords,
    const std::unique_ptr<sat::CpModelBuilder>& cp_model,
    bool enforce_enclosed_points) {
  size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  size_t dimension = coords[0].size();
  MaxHyperRectangularVars vars;
  // Boolean variable indicating if a point is included in the hyperrectangle.
  vars.point_included_var.reserve(num_points);
  for (size_t id = 0; id < num_points; ++id) {
    vars.point_included_var.push_back(cp_model->NewBoolVar());
  }
  // Integer variable with the coordinates of the hyperrectangle.
  std::vector<std::vector<int64_t>> coords_sorted =
      GetSortedCoordsPerDimension(coords);
  vars.lower_bound_vars.reserve(dimension);
  vars.upper_bound_vars.reserve(dimension);
  vars.length_vars.reserve(dimension);
  for (size_t dim = 0; dim < dimension; ++dim) {
    Domain valid_coords_domain = Domain::FromValues(coords_sorted[dim]);
    vars.lower_bound_vars.push_back(cp_model->NewIntVar(valid_coords_domain));
    vars.upper_bound_vars.push_back(cp_model->NewIntVar(valid_coords_domain));
    vars.length_vars.push_back(cp_model->NewIntVar(
        Domain{0, coords_sorted[dim].back() - coords_sorted[dim].front()}));
  }
  // Interval variables for bounds.
  std::vector<IntervalVar> interval_vars;
  vars.interval_vars.reserve(dimension);
  for (size_t dim = 0; dim < dimension; ++dim) {
    vars.interval_vars.push_back(cp_model->NewIntervalVar(
        vars.lower_bound_vars[dim], vars.length_vars[dim],
        vars.upper_bound_vars[dim]));
  }
  if (enforce_enclosed_points) {
    // Boolean variable indicating if the coords of a point is greater (lower)
    // than or equal to the lower (upper) bound of the hyperrectangle for each
    // dimension.
    std::vector<std::vector<BoolVar>> point_coord_geq_lb_var;
    std::vector<std::vector<BoolVar>> point_coord_leq_ub_var;
    vars.point_coord_geq_lb_var.resize(num_points);
    vars.point_coord_leq_ub_var.resize(num_points);
    for (size_t id = 0; id < num_points; ++id) {
      for (size_t dim = 0; dim < dimension; ++dim) {
        vars.point_coord_geq_lb_var[id].push_back(cp_model->NewBoolVar());
        vars.point_coord_leq_ub_var[id].push_back(cp_model->NewBoolVar());
      }
    }
  }
  return vars;
}

void AddModelConstraints(const std::vector<std::vector<int64_t>>& coords,
                         const MaxHyperRectangularVars& vars,
                         const std::unique_ptr<sat::CpModelBuilder>& cp_model,
                         bool enforce_enclosed_points) {
  size_t num_points = coords.size();
  CHECK_GT(num_points, 0);
  size_t dimension = coords[0].size();
  // If point is included, then lower and upper bound are valid.
  for (size_t dim = 0; dim < dimension; ++dim) {
    for (size_t id = 0; id < num_points; ++id) {
      cp_model->AddLessOrEqual(vars.lower_bound_vars[dim], coords[id][dim])
          .OnlyEnforceIf(vars.point_included_var[id]);
      cp_model->AddGreaterOrEqual(vars.upper_bound_vars[dim], coords[id][dim])
          .OnlyEnforceIf(vars.point_included_var[id]);
    }
  }
  if (enforce_enclosed_points) {
    // Force if point coordinate is not greater(lower) than lower (upper) bound.
    for (size_t dim = 0; dim < dimension; ++dim) {
      for (size_t id = 0; id < num_points; ++id) {
        cp_model->AddGreaterThan(vars.lower_bound_vars[dim], coords[id][dim])
            .OnlyEnforceIf(vars.point_coord_geq_lb_var[id][dim].Not());
        cp_model->AddLessThan(vars.upper_bound_vars[dim], coords[id][dim])
            .OnlyEnforceIf(vars.point_coord_leq_ub_var[id][dim].Not());
      }
    }
    // If coords are inside the bounds of the rectangle, it must be included.
    // point_coord_geq_lb_var and point_coord_leq_ub_var => point_included_var.
    for (int id = 0; id < num_points; ++id) {
      std::vector<BoolVar> all_bound_vars;
      all_bound_vars.reserve(2 * dimension);
      for (int dim = 0; dim < dimension; ++dim) {
        all_bound_vars.push_back(vars.point_coord_geq_lb_var[id][dim]);
        all_bound_vars.push_back(vars.point_coord_leq_ub_var[id][dim]);
      }
      cp_model->AddImplication(all_bound_vars, {vars.point_included_var[id]});
    }
  }
}

}  // namespace

MaxHyperRectangularCpSat::MaxHyperRectangularCpSat(
    const std::vector<std::vector<int64_t>>& coords,
    bool enforce_enclosed_points)
    : MaxHyperRectangular(coords),
      coords_(coords),
      cp_model_(std::make_unique<sat::CpModelBuilder>()),
      vars_(CreateModelVars(coords_, cp_model_, enforce_enclosed_points)) {
  AddModelConstraints(coords_, vars_, cp_model_, enforce_enclosed_points);
}

absl::StatusOr<MaxHyperRectangularSolution>
MaxHyperRectangularCpSat::SolveMaxHyperRectangular(
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
  DoubleLinearExpr objective;
  size_t dimension = coords_[0].size();
  for (size_t id = 0; id < weights.size(); ++id) {
    objective.AddTerm(vars_.point_included_var[id], weights[id]);
  }
  if (!ignore_perimeter_in_objective) {
    for (size_t dim = 0; dim < dimension; ++dim) {
      objective.AddTerm(vars_.length_vars[dim], -1.0);
    }
  }
  cp_model_->Maximize(objective);
  // Include locally added constraints.
  std::vector<Constraint> locally_added_constraints;
  locally_added_constraints.reserve(forbidden_points.size() +
                                    points_same_cluster.size() +
                                    points_diff_cluster.size());
  // Set forced and forbidden points.
  for (int point : forbidden_points)
    locally_added_constraints.push_back(
        cp_model_->AddEquality(vars_.point_included_var[point], 0));

  // Set pairs of points to be included on same cluster.
  for (auto [point1, point2] : points_same_cluster) {
    locally_added_constraints.push_back(cp_model_->AddEquality(
        vars_.point_included_var[point1], vars_.point_included_var[point2]));
  }
  for (auto [point1, point2] : points_diff_cluster) {
    locally_added_constraints.push_back(cp_model_->AddLessOrEqual(
        vars_.point_included_var[point1] + vars_.point_included_var[point2],
        1.0));
  }
  // TODO(edomoreno) Set parameters to set num workers.
  // SatParameters parameters;
  // parameters.set_log_search_progress(true);
  // parameters.set_log_to_stdout(true);
  // parameters.set_num_workers(32);
  const CpSolverResponse response = Solve(cp_model_->Build());
  // Clear locally added constraints.
  for (Constraint& constraint : locally_added_constraints) {
    constraint.MutableProto()->Clear();
  }
  if (response.status() == CpSolverStatus::INFEASIBLE) {
    return absl::NotFoundError("Infeasible problem");
  }
  if (response.status() != CpSolverStatus::OPTIMAL) {
    return absl::InternalError("Not optimal solution found.");
  }
  std::vector<bool> is_included(coords_.size());
  for (int id = 0; id < coords_.size(); ++id) {
    is_included[id] =
        SolutionBooleanValue(response, vars_.point_included_var[id]);
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
