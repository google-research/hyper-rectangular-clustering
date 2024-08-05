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
#include <vector>

#ifdef LOCAL
#include "scip/scip.h"
#include "src/branching_handler.h"
#include "src/problem_data.h"
#include "src/same_or_diff_constraint_handler.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/branching_handler.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/same_or_diff_constraint_handler.h"
#include "third_party/absl/log/check.h"
#include "third_party/scip/src/objscip/objprobdata.h"
#include "third_party/scip/src/objscip/objvardata.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/scip_branch.h"
#include "third_party/scip/src/scip/scip_cons.h"
#include "third_party/scip/src/scip/scip_message.h"
#include "third_party/scip/src/scip/scip_numerics.h"
#include "third_party/scip/src/scip/scip_prob.h"
#include "third_party/scip/src/scip/scip_solvingstats.h"
#include "third_party/scip/src/scip/scip_tree.h"
#include "third_party/scip/src/scip/scip_var.h"
#include "third_party/scip/src/scip/type_branch.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_result.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#include "third_party/scip/src/scip/type_tree.h"
#include "third_party/scip/src/scip/type_var.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {}  // namespace

// This branching rule requires that coverage vars are equal to 0 or 1 because
// if not fractional disjoint cluster are valid. So, we set a higher priority
// to coverage vars, and we skip the execution of the branching rule if a
// fractional coverage var appears. So, when fractional coverage var appears,
// we are sure that we can use this branching rule properly.
void SameDiffBranchingHandler::SetVarPriorityBranching(SCIP *scip) {
  ProbDataMaxHyperRectangular *problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular *>(SCIPgetObjProbData(scip));
  CHECK(problem_data != nullptr);
  for (int point_id = 0; point_id < problem_data->GetNumPoints(); ++point_id) {
    SCIP_VAR *coverage_var = problem_data->GetCoverageVar(point_id);
    CHECK_EQ(SCIPchgVarBranchPriority(scip, coverage_var, 100), SCIP_OKAY);
  }
}

SCIP_DECL_BRANCHEXECLP(SameDiffBranchingHandler::scip_execlp) {
  SCIPdebugMsg(scip, "start branching at node %lld, depth %d\n",
               SCIPgetNNodes(scip), SCIPgetDepth(scip));
  *result = SCIP_DIDNOTRUN;
  ProbDataMaxHyperRectangular *problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular *>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  const int num_points = problem_data->GetNumPoints();
  // Matrix to storage the sum of variables covering each pairs of points.
  std::vector<std::vector<SCIP_Real>> matrix_cumm_coverage(num_points);
  for (int point_id = 0; point_id < num_points; ++point_id)
    matrix_cumm_coverage[point_id].resize(point_id + 1);
  // Get fractional LP candidates.
  SCIP_VAR **lp_candidates;
  SCIP_Real *lp_candidates_frac_val;
  int num_lp_candidates;

  SCIP_CALL(SCIPgetLPBranchCands(scip, &lp_candidates, /*lpcandssol=*/
                                 nullptr, &lp_candidates_frac_val, /*nlpcands=*/
                                 nullptr, &num_lp_candidates, /*nfracimplvars=*/
                                 nullptr));
  assert(num_lp_candidates > 0);
  // Compute weights for each order pairs.
  for (int lp_candidate_id = 0; lp_candidate_id < num_lp_candidates;
       ++lp_candidate_id) {
    const SCIP_Real solution_val = lp_candidates_frac_val[lp_candidate_id];
    const VarDataMaxHyperRectangular *var_data =
        dynamic_cast<VarDataMaxHyperRectangular *>(
            SCIPgetObjVardata(scip, lp_candidates[lp_candidate_id]));
    assert(var_data != nullptr);
    // Check if is a pricer var. If not, exit with result SCIP_DIDNOTRUN.
    if (!var_data->IsPricerVar()) return SCIP_OKAY;
    // Construct weight matrix.
    for (int point_id = 0; point_id < num_points; ++point_id) {
      if (!var_data->IsPointIncluded(point_id)) continue;
      // Diagonal: sum of all variables including the point.
      matrix_cumm_coverage[point_id][point_id] += solution_val;
      // In coord [i][j] we store the sum of vars including both points.
      for (int other_point_id = point_id + 1; other_point_id < num_points;
           ++other_point_id) {
        if (!var_data->IsPointIncluded(other_point_id)) continue;
        matrix_cumm_coverage[other_point_id][point_id] += solution_val;
        assert(SCIPisFeasGE(
            scip, matrix_cumm_coverage[other_point_id][point_id], 0.0));
      }
    }
  }

  // Select best pair of points for branching. That is where the matrix value
  // is closer to 0.5. However, only select this pair if there is a variable
  // not including boths.
  SCIP_Real best_value = 0.0;
  int best_first_point = -1;
  int best_second_point = -1;
  for (int point_id = 0; point_id < num_points; ++point_id) {
    for (int other_point_id = 0; other_point_id < point_id; ++other_point_id) {
      SCIP_Real coverage_val =
          MIN(matrix_cumm_coverage[point_id][other_point_id],
              1.0 - matrix_cumm_coverage[point_id][other_point_id]);
      if (best_value < coverage_val) {
        if (SCIPisEQ(scip, matrix_cumm_coverage[point_id][other_point_id],
                     matrix_cumm_coverage[point_id][point_id]) &&
            SCIPisEQ(scip, matrix_cumm_coverage[point_id][other_point_id],
                     matrix_cumm_coverage[other_point_id][other_point_id]))
          continue;
        best_value = coverage_val;
        best_first_point = other_point_id;
        best_second_point = point_id;
      }
    }
  }
  // If properly executed, it must be a branching option.
  if (best_value == 0) return SCIP_ERROR;
  assert(best_value > 0.0);
  SCIPdebugMsg(scip, "branch on order pair <%d,%d> with weight <%g>\n",
               best_first_point, best_second_point, best_value);
  // Create the branch-and-bound tree child nodes of the current node.
  SCIP_NODE *child_same;
  SCIP_NODE *child_diff;
  SCIP_CALL(SCIPcreateChild(scip, &child_same, /*nodeselprio=*/0.0,
                            SCIPgetLocalTransEstimate(scip)));
  SCIP_CALL(SCIPcreateChild(scip, &child_diff, /*nodeselprio=*/0.0,
                            SCIPgetLocalTransEstimate(scip)));
  // Create the constraints;
  SCIP_CONS *constraint_same;
  SCIP_CONS *constraint_diff;
  SCIP_CALL(CreateSameDiffConstraint(
      scip, &constraint_same, "same", best_first_point, best_second_point,
      SameDiffConstraintType::kPointsOnSameCluster, child_same,
      /*local_constraint_only=*/true));
  SCIP_CALL(CreateSameDiffConstraint(
      scip, &constraint_diff, "diff", best_first_point, best_second_point,
      SameDiffConstraintType::kPointsOnDiffCluster, child_diff,
      /*local_constraint_only=*/true));
  // Add constraint to nodes.
  SCIP_CALL(SCIPaddConsNode(scip, child_same, constraint_same, nullptr));
  SCIP_CALL(SCIPaddConsNode(scip, child_diff, constraint_diff, nullptr));
  // Release constraints.
  SCIP_CALL(SCIPreleaseCons(scip, &constraint_same));
  SCIP_CALL(SCIPreleaseCons(scip, &constraint_diff));
  *result = SCIP_BRANCHED;
  return SCIP_OKAY;
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
