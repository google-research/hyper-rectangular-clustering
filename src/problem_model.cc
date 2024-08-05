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
#include <string>

#ifdef LOCAL
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "blockmemshell/memory.h"
#include "objscip/objpricer.h"
#include "scip/def.h"
#include "scip/scip_general.h"
#include "scip/scip_message.h"
#include "scip/scip_param.h"
#include "scip/scip_pricer.h"
#include "scip/scip_prob.h"
#include "scip/scip_sol.h"
#include "scip/scip_solve.h"
#include "scip/scipdefplugins.h"
#include "scip/type_retcode.h"
#include "scip/type_sol.h"
#include "src/branching_handler.h"
#include "src/pricer_scip.h"
#include "src/problem_model.h"
#include "src/same_or_diff_constraint_handler.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/branching_handler.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer_scip.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_model.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/same_or_diff_constraint_handler.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/scip/src/blockmemshell/memory.h"
#include "third_party/scip/src/objscip/objbranchrule.h"
#include "third_party/scip/src/objscip/objconshdlr.h"
#include "third_party/scip/src/objscip/objpricer.h"
#include "third_party/scip/src/objscip/objprobdata.h"
#include "third_party/scip/src/objscip/objvardata.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/scip_general.h"
#include "third_party/scip/src/scip/scip_message.h"
#include "third_party/scip/src/scip/scip_numerics.h"
#include "third_party/scip/src/scip/scip_param.h"
#include "third_party/scip/src/scip/scip_pricer.h"
#include "third_party/scip/src/scip/scip_prob.h"
#include "third_party/scip/src/scip/scip_sol.h"
#include "third_party/scip/src/scip/scip_solve.h"
#include "third_party/scip/src/scip/scip_solvingstats.h"
#include "third_party/scip/src/scip/scipdefplugins.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_sol.h"
#include "third_party/scip/src/scip/type_var.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

bool AbslParseFlag(absl::string_view text, BranchingOptions* branching,
                   std::string* error) {
  if (text == "none") {
    *branching = BranchingOptions::kNone;
    return true;
  }
  if (text == "ryanfoster") {
    *branching = BranchingOptions::kRyanFoster;
    return true;
  }
  *error = "unknown value for pricer";
  return false;
}

std::string AbslUnparseFlag(BranchingOptions branching) {
  switch (branching) {
    case BranchingOptions::kNone:
      return "none";
    case BranchingOptions::kRyanFoster:
      return "ryanfoster";
    default:
      return "error";
  }
}

SCIP_RETCODE
ClusteringMaxHyperRectangular::FormulateProblemWithTrivialColumns() {
  const int num_points = static_cast<int>(coords_.size());
  const std::string data_name = "my_problem";
  VLOG(1) << "Formulating problem with " << num_points << " points.";
  // Initialize SCIP environment.
  SCIP_CALL(SCIPcreate(&scip_));
  SCIPprintVersion(scip_, nullptr);
  SCIPinfoMessage(scip_, nullptr, "\n");
  SCIP_CALL(SCIPincludeDefaultPlugins(scip_));
  // Create empty problem.
  SCIP_CALL(SCIPcreateProb(scip_, "Problem", nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, nullptr));
  // Create Problem Data
  SCIP_CALL(SCIPcreateObjProb(scip_, data_name.c_str(), &problem_data_,
                              /*deleteobject=*/false));
  SCIP_CALL(problem_data_.FormulateProblem(scip_, max_num_outliers_,
                                           max_num_clusters_));
  return SCIP_OKAY;
}

SCIP_RETCODE ClusteringMaxHyperRectangular::SetMaxClusteringPricer(
    PricerOptions pricer_option) {
  VLOG(1) << "Creating MaxClustering_Pricer.";
  auto* pricer =
      new ObjPricerMaxClustering(scip_, kPRICER_NAME, coords_, pricer_option);
  SCIP_CALL(SCIPincludeObjPricer(scip_, pricer, true));
  SCIP_CALL(SCIPactivatePricer(scip_, SCIPfindPricer(scip_, kPRICER_NAME)));
  VLOG(1) << "Done!";
  return SCIP_OKAY;
}

SCIP_RETCODE ClusteringMaxHyperRectangular::SolveProblem() {
  VLOG(1) << "Solving.";
  SCIP_CALL(SCIPsolve(scip_));
  SCIP_CALL(SCIPprintStatistics(scip_, nullptr));
  SCIP_CALL(SCIPprintBestSol(scip_, nullptr, FALSE));
  VLOG(1) << "Done.";
  return SCIP_OKAY;
}

ProblemSolution ClusteringMaxHyperRectangular::GetSolution() {
  ProblemSolution solution;
  const size_t num_points = coords_.size();
  solution.point_coverage.resize(num_points, 0.0);
  SCIP_SOL* best_sol = SCIPgetBestSol(scip_);
  for (int point_id = 0; point_id < num_points; ++point_id) {
    solution.point_coverage[point_id] =
        SCIPgetSolVal(scip_, best_sol, problem_data_.GetCoverageVar(point_id));
  }
  solution.half_perimeter = SCIPgetSolOrigObj(scip_, best_sol);
  VLOG(1) << "Getting solution from problem data.";
  for (SCIP_VAR* cluster_var : problem_data_.GetPricerVars()) {
    const double var_value = SCIPgetSolVal(scip_, best_sol, cluster_var);
    if (SCIPisZero(scip_, var_value)) continue;
    VarDataMaxHyperRectangular* var_data =
        dynamic_cast<VarDataMaxHyperRectangular*>(
            SCIPgetObjVardata(scip_, cluster_var));
    CHECK(var_data != nullptr);
    solution.cluster_limits.push_back(
        {var_data->GetClusterLimits(), var_value});
  }
  return solution;
}

absl::StatusOr<ProblemSolution> ClusteringMaxHyperRectangular::Solve(
    SolveParameters parameters) {
  if (parameters.pricer != PricerOptions::kNone)
    SetMaxClusteringPricer(parameters.pricer);
  if (parameters.branching == BranchingOptions::kRyanFoster) {
    // Check pricing is correct and activate adding branching constraints.
    CHECK(parameters.pricer != PricerOptions::kPricerMaxClosure)
        << "Max Closure pricer not compatible with RyanFoster Branching.";
    auto pricer = dynamic_cast<ObjPricerMaxClustering*>(
        SCIPfindObjPricer(scip_, kPRICER_NAME));
    pricer->ActivatePairedPairsBranching();
    // Set branching rule.
    auto* branch_rule = new SameDiffBranchingHandler(scip_);
    CHECK_EQ(SCIPincludeObjBranchrule(scip_, branch_rule, true), SCIP_OKAY);
    auto* constraint_handler = new SameDiffConstraintHandler(scip_);
    CHECK_EQ(SCIPincludeObjConshdlr(scip_, constraint_handler, true),
             SCIP_OKAY);
  }
  // Set parameters
  CHECK_EQ(
      SCIPsetIntParam(scip_, "display/verblevel", parameters.display_verblevel),
      SCIP_OKAY);
  CHECK_EQ(SCIPsetRealParam(scip_, "limits/time", parameters.limits_time),
           SCIP_OKAY);
  // CHECK_EQ(SCIPsetBoolParam(scip, "display/lpinfo", TRUE), SCIP_OKAY);
  // Solve
  CHECK_EQ(SolveProblem(), SCIP_OKAY);
  ProblemSolution solution = GetSolution();
  problem_data_.FreeVarsAndConstraints(scip_);
  CHECK_EQ(SCIPfree(&scip_), SCIP_OKAY);
  BMScheckEmptyMemory();
  return solution;
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
