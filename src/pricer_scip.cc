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

#include <cassert>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "objscip/objprobdata.h"
#include "scip/cons_linear.h"
#include "scip/def.h"
#include "scip/pub_var.h"
#include "scip/scip_cons.h"
#include "scip/scip_message.h"
#include "scip/scip_numerics.h"
#include "scip/type_pricer.h"
#include "scip/type_result.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#include "src/pricer_scip.h"
#include "src/problem_data.h"
#include "src/same_or_diff_constraint_handler.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer_scip.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/same_or_diff_constraint_handler.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/scip/src/objscip/objprobdata.h"
#include "third_party/scip/src/scip/cons_linear.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/pub_cons.h"
#include "third_party/scip/src/scip/pub_var.h"
#include "third_party/scip/src/scip/scip_cons.h"
#include "third_party/scip/src/scip/scip_message.h"
#include "third_party/scip/src/scip/scip_numerics.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_pricer.h"
#include "third_party/scip/src/scip/type_result.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"

#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {
struct PairedPointsConstraints {
  std::vector<std::pair<int, int>> points_same_cluster;
  std::vector<std::pair<int, int>> points_diff_cluster;
};

// Return the pairs of points that must be in the same/diff cluster due
// to the application of branching rules in the B&B tree.
PairedPointsConstraints GetPairedPointsFromBranchingRules(SCIP* scip) {
  PairedPointsConstraints paired_points_constraints;
  SCIP_CONSHDLR* same_diff_constraint_handler =
      SCIPfindConshdlr(scip, "SameDiffContHdlr");
  assert(same_diff_constraint_handler != nullptr);
  SCIP_CONS** branch_constraints =
      SCIPconshdlrGetConss(same_diff_constraint_handler);
  int num_branch_constraints =
      SCIPconshdlrGetNConss(same_diff_constraint_handler);
  // Loop over branching constraints to find paired points.
  for (int const_id = 0; const_id < num_branch_constraints; ++const_id) {
    SCIP_CONS* branch_cons = branch_constraints[const_id];
    // Ignore if not active.
    if (!SCIPconsIsActive(branch_cons)) continue;
    const SameDiffConstraintDetails constraint_details =
        GetConstraintDetails(branch_cons);
    SCIPdebugMsg(scip, "imposing branch constraint for %s(%d,%d)\n",
                 constraint_details.constraint_type ==
                         SameDiffConstraintType::kPointsOnSameCluster
                     ? "same"
                     : "diff",
                 constraint_details.first_point,
                 constraint_details.second_point);
    if (constraint_details.constraint_type ==
        SameDiffConstraintType::kPointsOnSameCluster)
      paired_points_constraints.points_same_cluster.push_back(
          {constraint_details.first_point, constraint_details.second_point});
    else
      paired_points_constraints.points_diff_cluster.push_back(
          {constraint_details.first_point, constraint_details.second_point});
  }
  return paired_points_constraints;
}
}  // namespace

bool AbslParseFlag(absl::string_view text, PricerOptions* pricer,
                   std::string* error) {
  if (text == "cpsat") {
    *pricer = PricerOptions::kPricerCpSat;
    return true;
  }
  if (text == "cpsat_enforcing") {
    *pricer = PricerOptions::kPricerCpSatEnforcing;
    return true;
  }
  if (text == "maxclosure") {
    *pricer = PricerOptions::kPricerMaxClosure;
    return true;
  }
  if (text == "mip") {
    *pricer = PricerOptions::kPricerMIP;
    return true;
  }
  *error = "unknown value for pricer";
  return false;
}

std::string AbslUnparseFlag(PricerOptions pricer) {
  switch (pricer) {
    case PricerOptions::kPricerCpSat:
      return "cpsat";
    case PricerOptions::kPricerCpSatEnforcing:
      return "cpsat_enforcing";
    case PricerOptions::kPricerMaxClosure:
      return "maxclosure";
    case PricerOptions::kPricerMIP:
      return "mip";
    default:
      return "error";
  }
}

// initialization of variable pricer (called after problem was transformed)
// Required to recover original constraints from transformed references.
SCIP_DECL_PRICERINIT(ObjPricerMaxClustering::scip_init) {
  VLOG(1) << "Init pricer";
  ProbDataMaxHyperRectangular* problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular*>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  SCIP_CALL(SCIPgetTransformedCons(
      scip, *(problem_data->GetMaxNumClustersConstraintPtr()),
      problem_data->GetMaxNumClustersConstraintPtr()));
  for (int point_id = 0; point_id < point_coords_.size(); ++point_id) {
    SCIP_CALL(SCIPgetTransformedCons(
        scip, *(problem_data->GetCoverageConstraintPtr(point_id)),
        problem_data->GetCoverageConstraintPtr(point_id)));
  }
  return SCIP_OKAY;
}

// It should not be an infeasible case, so isfarkas is ignored.
SCIP_RETCODE ObjPricerMaxClustering::pricing(SCIP* scip, bool isfarkas) {
  VLOG(1) << "Calling pricing with isfarkas=" << isfarkas;
  size_t num_points = point_coords_.size();
  std::vector<double> dual_coverage_constraints(num_points);
  ProbDataMaxHyperRectangular* problem_data;
  problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular*>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  VLOG(1) << "dual of coverage_constraints: ";
  for (int point_id = 0; point_id < point_coords_.size(); ++point_id) {
    if (isfarkas) {
      dual_coverage_constraints[point_id] =
          static_cast<double>(SCIPgetDualfarkasLinear(
              scip, *(problem_data->GetCoverageConstraintPtr(point_id))));
    } else {
      dual_coverage_constraints[point_id] =
          static_cast<double>(SCIPgetDualsolLinear(
              scip, *(problem_data->GetCoverageConstraintPtr(point_id))));
    }
    VLOG(1) << "\tPoint " << point_id << ": "
            << dual_coverage_constraints[point_id];
  }
  // Check for local forbidden ports.
  std::vector<int> forbidden_points;
  std::vector<int> forced_points;
  for (int point_id = 0; point_id < point_coords_.size(); ++point_id) {
    const double var_local_ub =
        SCIPvarGetUbLocal(problem_data->GetCoverageVar(point_id));
    if (SCIPisLE(scip, var_local_ub, 0.0)) {
      VLOG(1) << "\tForbidden point " << point_id;
      forbidden_points.push_back(point_id);
    }
    const double var_local_lb =
        SCIPvarGetLbLocal(problem_data->GetCoverageVar(point_id));
    if (SCIPisGE(scip, var_local_lb, 1.0)) {
      VLOG(1) << "\tNeed to cover (not enforced): " << point_id;
    }
  }
  PairedPointsConstraints paired_points_constraints;
  if (use_paired_pairs_branching_) {
    paired_points_constraints = GetPairedPointsFromBranchingRules(scip);
  }
  // Solve problem with these weights.
  VLOG(1) << "Solving MaxRectangular: ";
  absl::StatusOr<MaxHyperRectangularSolution> cluster_solution =
      max_hyper_rectangular_->SolveMaxHyperRectangular(
          dual_coverage_constraints, forbidden_points,
          paired_points_constraints.points_same_cluster,
          paired_points_constraints.points_diff_cluster,
          /*ignore_perimeter_in_objective=*/isfarkas);
  // If infeasible, return ok without adding any column.
  if (cluster_solution.status().code() == absl::StatusCode::kNotFound) {
    VLOG(1) << "Infeasible or empty subproblem, returning no new column";
    return SCIP_OKAY;
  }
  if (!cluster_solution.ok()) return SCIP_ERROR;
  VLOG(1) << "Solved with objective value="
          << cluster_solution->objective_value;
  for (int point_id = 0; point_id < point_coords_.size(); ++point_id)
    VLOG(1) << "\tPoint " << point_id << ": "
            << cluster_solution->included_points[point_id];
  for (auto bounds : cluster_solution->limits_solution) {
    VLOG(1) << "\tBounds: " << bounds.first << ", " << bounds.second;
  }

  // Check if reduced cost is negative, to include it on the solution.
  const SCIP_Real max_num_clusters_dual_ =
      (isfarkas) ? SCIPgetDualfarkasLinear(
                       scip, *(problem_data->GetMaxNumClustersConstraintPtr()))
                 : SCIPgetDualsolLinear(
                       scip, *(problem_data->GetMaxNumClustersConstraintPtr()));
  VLOG(1) << "max_num_clusters_dual_: " << max_num_clusters_dual_;
  VLOG(1) << "reduced_cost: "
          << -max_num_clusters_dual_ -
                 static_cast<SCIP_Real>(cluster_solution->objective_value);
  if (SCIPisSumGT(scip, cluster_solution->objective_value,
                  -max_num_clusters_dual_)) {
    return problem_data->AddClusterVariables(
        scip, cluster_solution.value());
  }
  return SCIP_OKAY;
}

// Pricing of additional variables if LP is feasible.
SCIP_DECL_PRICERREDCOST(ObjPricerMaxClustering::scip_redcost) {
  SCIPdebugMsg(scip, "call scip_redcost ...\n");
  *result = SCIP_SUCCESS;
  SCIP_CALL(pricing(scip, false));
  return SCIP_OKAY;
}
SCIP_DECL_PRICERFARKAS(ObjPricerMaxClustering::scip_farkas) {
  SCIPdebugMsg(scip, "call scip_farkas ...\n");
  SCIP_CALL(pricing(scip, true));
  return SCIP_OKAY;
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
