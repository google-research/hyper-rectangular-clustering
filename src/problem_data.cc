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
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "pricer/pricer.h"
#include "scip/cons_linear.h"
#include "scip/pub_misc.h"
#include "scip/scip_cons.h"
#include "scip/scip_numerics.h"
#include "scip/scip_prob.h"
#include "scip/scip_var.h"
#include "scip/type_cons.h"
#include "scip/type_scip.h"
#include "scip/type_var.h"
#include "src/problem_data.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "third_party/scip/src/objscip/objvardata.h"
#include "third_party/scip/src/scip/cons_linear.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/pub_misc.h"
#include "third_party/scip/src/scip/scip_cons.h"
#include "third_party/scip/src/scip/scip_numerics.h"
#include "third_party/scip/src/scip/scip_prob.h"
#include "third_party/scip/src/scip/scip_var.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#include "third_party/scip/src/scip/type_var.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {

MaxHyperRectangularSolution GetSolutionIncludingAllPoints(
    std::vector<std::vector<int64_t>> coords) {
  MaxHyperRectangularSolution solution;
  solution.included_points.resize(coords.size(), true);
  size_t dimension = coords[0].size();
  int64_t half_perimeter = 0;
  for (int dim = 0; dim < dimension; ++dim) {
    auto min_coord = std::min_element(
        coords.begin(), coords.end(),
        [&dim](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
          return a[dim] < b[dim];
        });
    auto max_coord = std::max_element(
        coords.begin(), coords.end(),
        [&dim](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
          return a[dim] < b[dim];
        });
    half_perimeter += (max_coord->at(dim) - min_coord->at(dim));
    solution.limits_solution.push_back(
        {min_coord->at(dim), max_coord->at(dim)});
  }
  solution.objective_value = static_cast<double>(half_perimeter);
  return solution;
}

void ConvertSolutionToNiceName(
    const std::vector<std::pair<int64_t, int64_t>>& solution, char* var_name) {
  constexpr int kSize = 255;
  char tmp_name[kSize];
  (void)SCIPsnprintf(var_name, kSize, "C");
  for (const std::pair<int64_t, int64_t>& bound : solution) {
    std::strncpy(tmp_name, var_name, kSize);
    (void)SCIPsnprintf(var_name, kSize, "%s[%d,%d]", tmp_name, bound.first,
                       bound.second);
  }
}
}  // namespace

SCIP_RETCODE ProbDataMaxHyperRectangular::CreateCoverageVariables(SCIP* scip) {
  point_coverage_variables_.resize(num_points_,
                                   static_cast<SCIP_VAR*>(nullptr));
  // Add coverage of point variables.
  char var_name[255];
  SCIP_VARTYPE var_type =
      relax_coverage_vars_ ? SCIP_VARTYPE_CONTINUOUS : SCIP_VARTYPE_BINARY;
  for (int point_id = 0; point_id < num_points_; ++point_id) {
    VarDataMaxHyperRectangular* var_data_ptr = new VarDataMaxHyperRectangular();
    (void)SCIPsnprintf(var_name, sizeof(var_name), "covered_%d", point_id);
    SCIP_CALL(SCIPcreateObjVar(scip, &(point_coverage_variables_[point_id]),
                               var_name, /*lb=*/0.0, /*ub=*/1.0,
                               /*obj=*/0.0, /*type=*/var_type,
                               /*initial=*/true, /*removable=*/false,
                               var_data_ptr, /*deleteobject=*/true));
    SCIP_CALL(SCIPaddVar(scip, point_coverage_variables_[point_id]));
  }
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::CreateInitialSetColumnsVars(
    SCIP* scip) {
  // Add col representing a cluster with a single point.
  for (int point_id = 0; point_id < num_points_; ++point_id) {
    MaxHyperRectangularSolution single_solution;
    single_solution.included_points.resize(num_points_, false);
    single_solution.included_points[point_id] = true;
    single_solution.objective_value = 0.0;
    const std::vector<int64_t>& coords = point_coords_[point_id];
    for (int64_t coord : coords)
      single_solution.limits_solution.push_back({coord, coord});
    SCIP_CALL(AddClusterVariables(scip, single_solution, /*is_initial=*/true));
  }
  MaxHyperRectangularSolution solution_all_points =
      GetSolutionIncludingAllPoints(point_coords_);
  SCIP_CALL(AddClusterVariables(scip, solution_all_points,
                                /*is_initial=*/true));
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::AddMaxNumOutliersConstraint(
    SCIP* scip, int max_num_outliers) {
  char con_name[255];
  (void)SCIPsnprintf(con_name, sizeof(con_name), "max_num_outliers");
  std::vector<SCIP_Real> coeff_one(num_points_, 1.0);
  SCIP_CALL(SCIPcreateConsLinear(
      scip, &max_num_outliers_constraint_, con_name,
      /*nvars=*/num_points_, point_coverage_variables_.data(), coeff_one.data(),
      /*lhs*/ num_points_ - max_num_outliers, /*rhs*/ SCIPinfinity(scip),
      /*initial*/ true, /*separate*/ false,
      /*enforce*/ true, /*check*/ true, /*propagate*/ true, /*local*/ false,
      /*modifiable*/ false, /*dynamic*/ false, /*removable*/ false,
      /*stickingatnode*/ false));
  SCIP_CALL(SCIPaddCons(scip, max_num_outliers_constraint_));
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::AddMaxNumClustersConstraint(
    SCIP* scip, int max_num_clusters) {
  char con_name[255];
  (void)SCIPsnprintf(con_name, sizeof(con_name), "max_num_clusters");

  SCIP_CALL(SCIPcreateConsLinear(
      scip, &max_num_clusters_constraint_, con_name,
      /*nvars=*/0, /*vars=*/nullptr,
      /*vals*/ nullptr,
      /*lhs*/ 0, /*rhs*/ max_num_clusters,
      /*initial*/ true, /*separate*/ false,
      /*enforce*/ true, /*check*/ true, /*propagate*/ true, /*local*/ false,
      /*modifiable*/ true, /*dynamic*/ false, /*removable*/ false,
      /*stickingatnode*/ false));
  SCIP_CALL(SCIPaddCons(scip, max_num_clusters_constraint_));
  max_num_clusters_constraint_orig_ = max_num_clusters_constraint_;
  return SCIP_OKAY;
}

// Add coverage constraint for each point.
// sum_c is_i_in_cluster[c]*x[c] == is_covered[i]
SCIP_RETCODE ProbDataMaxHyperRectangular::AddCoverageConstraints(SCIP* scip) {
  coverage_of_points_constraint_.resize(num_points_,
                                        static_cast<SCIP_CONS*>(nullptr));
  coverage_of_points_constraint_orig_.resize(num_points_,
                                             static_cast<SCIP_CONS*>(nullptr));
  char con_name[255];
  for (int point_id = 0; point_id < num_points_; ++point_id) {
    (void)SCIPsnprintf(con_name, sizeof(con_name), "coverage_%d", point_id);
    SCIP_Real minus_one = -1.0;
    SCIP_CALL(SCIPcreateConsLinear(
        scip, &(coverage_of_points_constraint_[point_id]), con_name,
        /*nvars=*/1, /*vars=*/&point_coverage_variables_[point_id],
        /*vals*/ &minus_one,
        /*lhs*/ 0, /*rhs*/ std::numeric_limits<double>::max(),
        /*initial*/ true, /*separate*/ false,
        /*enforce*/ true, /*check*/ true, /*propagate*/ true, /*local*/ false,
        /*modifiable*/ true, /*dynamic*/ false, /*removable*/ false,
        /*stickingatnode*/ false));
    SCIP_CALL(SCIPaddCons(scip, coverage_of_points_constraint_[point_id]));
    coverage_of_points_constraint_orig_[point_id] =
        coverage_of_points_constraint_[point_id];
  }
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::FormulateProblem(
    SCIP* scip, int max_num_outliers, int max_numclusters) {
  // Add coverage of point variables.
  SCIP_CALL(CreateCoverageVariables(scip));
  SCIP_CALL(AddMaxNumOutliersConstraint(scip, max_num_outliers));
  SCIP_CALL(AddMaxNumClustersConstraint(scip, max_numclusters));
  SCIP_CALL(AddCoverageConstraints(scip));
  SCIP_CALL(CreateInitialSetColumnsVars(scip));
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::FreeVarsAndConstraints(SCIP* scip) {
  for (SCIP_VAR*& var : pricer_vars_) SCIP_CALL(SCIPreleaseVar(scip, &var));
  for (int i = 0; i < point_coords_.size(); ++i) {
    SCIP_CALL(SCIPreleaseVar(scip, &point_coverage_variables_[i]));
    SCIP_CALL(SCIPreleaseCons(scip, &coverage_of_points_constraint_orig_[i]));
  }
  SCIP_CALL(SCIPreleaseCons(scip, &max_num_clusters_constraint_orig_));
  SCIP_CALL(SCIPreleaseCons(scip, &max_num_outliers_constraint_));
  return SCIP_OKAY;
}

SCIP_RETCODE ProbDataMaxHyperRectangular::AddClusterVariables(
    SCIP* scip, const MaxHyperRectangularSolution& solution, bool is_initial) {
  SCIP_VAR* var;
  char var_name[255];
  ConvertSolutionToNiceName(solution.limits_solution, var_name);
  // Recompute solution objective.
  MaxHyperRectangularSolution updated_solution = solution;
  double half_perimeter = 0;
  for (std::pair<int64_t, int64_t> bounds : solution.limits_solution) {
    half_perimeter += static_cast<double>(bounds.second - bounds.first);
  }
  updated_solution.objective_value = half_perimeter;
  VarDataMaxHyperRectangular* var_data_ptr =
      new VarDataMaxHyperRectangular(updated_solution);
  if (relax_priced_vars_) {
    // Created with upper bound infinity to avoid computing its reduced cost.
    SCIP_CALL(SCIPcreateObjVar(
        scip, &var, var_name, /*lb=*/0.0,
        /*ub=*/SCIPinfinity(scip), /*obj=*/half_perimeter,
        /*type=*/SCIP_VARTYPE_CONTINUOUS, /*initial=*/is_initial,
        /*removable=*/false, var_data_ptr, /*deleteobject=*/true));
  } else {
    SCIP_CALL(SCIPcreateObjVar(
        scip, &var, var_name, /*lb=*/0.0,
        /*ub=*/1.0, /*obj=*/half_perimeter,
        /*type=*/SCIP_VARTYPE_BINARY, /*initial=*/is_initial,
        /*removable=*/false, var_data_ptr, /*deleteobject=*/true));
  }
  if (is_initial) {
    SCIP_CALL(SCIPaddVar(scip, var));
  } else {
    // Add new variable to the list of variables to price into LP (score: 1).
    SCIP_CALL(SCIPaddPricedVar(scip, var, /*score=*/1.0));
  }
  // If not relaxed, set UB as lazy because is imposed by other constraints
  // and to avoid its reduced cost.
  if (!relax_priced_vars_) SCIP_CALL(SCIPchgVarUbLazy(scip, var, 1.0));

  // Add new variable into the set partition constraint (coverage).
  for (int point_id = 0; point_id < point_coords_.size(); ++point_id) {
    if (solution.included_points[point_id]) {
      SCIP_CALL(SCIPaddCoefLinear(
          scip, coverage_of_points_constraint_[point_id], var, 1.0));
    }
  }
  // Add new variable into max_num_cluster constraint.
  SCIP_CALL(SCIPaddCoefLinear(scip, max_num_clusters_constraint_, var, 1.0));
  pricer_vars_.push_back(var);
  return SCIP_OKAY;
}
}  // namespace hyperrectangular_clustering
}  // namespace operations_research
