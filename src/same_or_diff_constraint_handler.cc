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
#include "src/problem_data.h"
#include "src/same_or_diff_constraint_handler.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_data.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/same_or_diff_constraint_handler.h"
#include "third_party/absl/log/check.h"
#include "third_party/scip/src/objscip/objprobdata.h"
#include "third_party/scip/src/objscip/objvardata.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/pub_cons.h"
#include "third_party/scip/src/scip/pub_message.h"
#include "third_party/scip/src/scip/pub_tree.h"
#include "third_party/scip/src/scip/pub_var.h"
#include "third_party/scip/src/scip/scip_cons.h"
#include "third_party/scip/src/scip/scip_mem.h"
#include "third_party/scip/src/scip/scip_message.h"
#include "third_party/scip/src/scip/scip_tree.h"
#include "third_party/scip/src/scip/scip_var.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_result.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#include "third_party/scip/src/scip/type_tree.h"
#include "third_party/scip/src/scip/type_var.h"
#endif

using operations_research::hyperrectangular_clustering::SameDiffConstraintType;

// Redefine SCIP_CONSDATA structure for this type of constraint.
/* num_propagated_vars :=number of variables that existed, the last time, the
 * related node was propagated, used to determine whether the constraint
 * should be repropagated */
struct SCIP_ConsData {
  int first_point;
  int second_point;
  SameDiffConstraintType constraint_type;
  int num_propagated_vars;
  int num_propagations;  // Number propagations runs of this constraint.
  bool is_propagated;    // is constraint already propagated?
  SCIP_NODE* node;  // The node in the B&B-tree at which the cons is sticking
};

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {
// Check if variable satisfies the constraint. If not, it is fixed to zero.
SCIP_RETCODE CheckValidityVariableInConstraint(SCIP* scip,
                                               SCIP_CONSDATA* consdata,
                                               SCIP_VAR* var,
                                               int* num_fixed_variables,
                                               SCIP_Bool* detected_cutoff) {
  assert(scip != nullptr);
  assert(consdata != nullptr);
  assert(var != nullptr);
  assert(num_fixed_variables != nullptr);
  assert(detected_cutoff != nullptr);
  // Check if variable is already zero.
  if (SCIPvarGetUbLocal(var) < 0.5) return SCIP_OKAY;
  // Check if variable contains points involved in the constraint.
  const VarDataMaxHyperRectangular* var_data =
      dynamic_cast<VarDataMaxHyperRectangular*>(SCIPgetObjVardata(scip, var));
  CHECK(var_data != nullptr);
  const bool is_point_one = var_data->IsPointIncluded(consdata->first_point);
  const bool is_point_two = var_data->IsPointIncluded(consdata->second_point);
  if ((consdata->constraint_type ==
           SameDiffConstraintType::kPointsOnSameCluster &&
       is_point_one != is_point_two) ||
      (consdata->constraint_type ==
           SameDiffConstraintType::kPointsOnDiffCluster &&
       is_point_one && is_point_two)) {
    SCIP_Bool has_been_fixed;
    SCIP_Bool is_infeasible;
    SCIP_CALL(SCIPfixVar(scip, var, 0.0, &is_infeasible, &has_been_fixed));
    if (is_infeasible) {
      assert(SCIPvarGetLbLocal(var) > 0.5);
      SCIPdebugMsg(scip, "-> cutoff\n");
      (*detected_cutoff) = TRUE;
    } else {
      assert(has_been_fixed);
      (*num_fixed_variables)++;
    }
  }
  return SCIP_OKAY;
}

// Check all variables for the constraint fixing and pruning if needed.
SCIP_RETCODE FixAllInvalidVariables(SCIP* scip, SCIP_CONSDATA* consdata,
                                    const std::vector<SCIP_VAR*>& vars,
                                    SCIP_RESULT* result) {
  int num_fixed_variables = 0;
  SCIP_Bool detected_cutoff = FALSE;
  SCIPdebugMsg(scip, "check variables %d to %d\n",
               consdata->num_propagated_vars, static_cast<int>(vars.size()));
  for (int v = consdata->num_propagated_vars;
       v < vars.size() && !detected_cutoff; ++v) {
    SCIP_CALL(CheckValidityVariableInConstraint(
        scip, consdata, vars[v], &num_fixed_variables, &detected_cutoff));
  }
  SCIPdebugMsg(scip, "fixed %d variables locally\n", num_fixed_variables);
  if (detected_cutoff)
    *result = SCIP_CUTOFF;
  else if (num_fixed_variables > 0)
    *result = SCIP_REDUCEDDOM;
  return SCIP_OKAY;
}

SCIP_RETCODE CreateSameDiffConstraintData(
    SCIP* scip, SCIP_CONSDATA** constraint_data, const int first_point,
    const int second_point, const SameDiffConstraintType constraint_type,
    SCIP_NODE* node_where_created) {
  assert(scip != nullptr);
  assert(constraint_data != nullptr);
  assert(first_point >= 0);
  assert(second_point >= 0);
  assert(first_point < second_point);

  SCIP_CALL(SCIPallocBlockMemory(scip, constraint_data));
  (*constraint_data)->first_point = first_point;
  (*constraint_data)->second_point = second_point;
  (*constraint_data)->constraint_type = constraint_type;
  (*constraint_data)->num_propagated_vars = 0;
  (*constraint_data)->num_propagations = 0;
  (*constraint_data)->is_propagated = FALSE;
  (*constraint_data)->node = node_where_created;
  return SCIP_OKAY;
}

SCIP_RETCODE FreeSameDiffConstraintData(SCIP* scip,
                                        SCIP_CONSDATA** constraint_data) {
  assert(constraint_data != nullptr);
  assert(*constraint_data != nullptr);
  SCIPfreeBlockMemory(scip, constraint_data);
  return SCIP_OKAY;
}

#ifdef SCIP_DEBUG
void DisplaySameDiffConstraintData(SCIP* scip, SCIP_CONSDATA* constraint_data,
                                   FILE* file_stream) {
  SCIPinfoMessage(scip, file_stream, "%s(%d,%d) at node %lld\n",
                  (constraint_data->constraint_type ==
                   SameDiffConstraintType::kPointsOnSameCluster)
                      ? "same"
                      : "diff",
                  constraint_data->first_point, constraint_data->second_point,
                  SCIPnodeGetNumber(constraint_data->node));
}
#endif  // SCIP_DEBUG
}  // namespace

// Required Callbacks for Constraint Handler
// To delete Constraint Data
SCIP_DECL_CONSDELETE(SameDiffConstraintHandler::scip_delete) {
  SCIP_CALL(FreeSameDiffConstraintData(scip, consdata));
  return SCIP_OKAY;
}

// Transforms constraint data into data belonging to the transformed problem.
// Required because ConsData is modified by other callbacks.
SCIP_DECL_CONSTRANS(SameDiffConstraintHandler::scip_trans) {
  SCIP_CONSDATA* source_data = SCIPconsGetData(sourcecons);
  SCIP_CONSDATA* target_data;
  // Create constraint data copying original values.
  SCIP_CALL(CreateSameDiffConstraintData(
      scip, &target_data, source_data->first_point, source_data->second_point,
      source_data->constraint_type, source_data->node));
  // Create target constraint.
  SCIP_CALL(SCIPcreateCons(
      scip, targetcons, SCIPconsGetName(sourcecons), conshdlr, target_data,
      SCIPconsIsInitial(sourcecons), SCIPconsIsSeparated(sourcecons),
      SCIPconsIsEnforced(sourcecons), SCIPconsIsChecked(sourcecons),
      SCIPconsIsPropagated(sourcecons), SCIPconsIsLocal(sourcecons),
      SCIPconsIsModifiable(sourcecons), SCIPconsIsDynamic(sourcecons),
      SCIPconsIsRemovable(sourcecons), SCIPconsIsStickingAtNode(sourcecons)));
  return SCIP_OKAY;
}

// Main callback. It fixes variables violating the constraint.
SCIP_DECL_CONSPROP(SameDiffConstraintHandler::scip_prop) {
  *result = SCIP_DIDNOTFIND;
  ProbDataMaxHyperRectangular* problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular*>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  for (int cons_id = 0; cons_id < nconss; ++cons_id) {
    SCIP_CONSDATA* consdata = SCIPconsGetData(conss[cons_id]);
    if (!consdata->is_propagated) {
      SCIPdebugMsg(scip, "propagate constraint <%s> ",
                   SCIPconsGetName(conss[cons_id]));
      SCIPdebug(DisplaySameDiffConstraintData(scip, consdata, NULL));
      SCIP_CALL(FixAllInvalidVariables(scip, consdata,
                                       problem_data->GetPricerVars(), result));
      consdata->num_propagations++;
      if (*result != SCIP_CUTOFF) {
        consdata->is_propagated = TRUE;
        consdata->num_propagated_vars = problem_data->GetNumPricerVars();
      } else {
        break;
      }
    }
  }
  return SCIP_OKAY;
}

SCIP_DECL_CONSACTIVE(SameDiffConstraintHandler::scip_active) {
  SCIP_CONSDATA* consdata = SCIPconsGetData(cons);
  ProbDataMaxHyperRectangular* problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular*>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  SCIPdebugMsg(scip, "activate constraint <%s> at node <%lld> in depth <%d>: ",
               SCIPconsGetName(cons), SCIPnodeGetNumber(consdata->node),
               SCIPnodeGetDepth(consdata->node));
  SCIPdebug(DisplaySameDiffConstraintData(scip, consdata, NULL));
  if (consdata->num_propagated_vars != problem_data->GetNumPricerVars()) {
    SCIPdebugMsg(scip, "-> mark constraint to be repropagated\n");
    consdata->is_propagated = FALSE;
    SCIP_CALL(SCIPrepropagateNode(scip, consdata->node));
  }
  return SCIP_OKAY;
}

SCIP_DECL_CONSDEACTIVE(SameDiffConstraintHandler::scip_deactive) {
  SCIP_CONSDATA* consdata = SCIPconsGetData(cons);
  ProbDataMaxHyperRectangular* problem_data =
      dynamic_cast<ProbDataMaxHyperRectangular*>(SCIPgetObjProbData(scip_));
  CHECK(problem_data != nullptr);
  SCIPdebugMsg(scip,
               "deactivate constraint <%s> at node <%lld> in depth "
               "<%d>: ",
               SCIPconsGetName(cons), SCIPnodeGetNumber(consdata->node),
               SCIPnodeGetDepth(consdata->node));
  SCIPdebug(DisplaySameDiffConstraintData(scip, consdata, NULL));
  // set the number of propagated variables to current number of variables.
  consdata->num_propagated_vars = problem_data->GetNumPricerVars();
  return SCIP_OKAY;
}

// Interface Methods
// Creates and captures a samediff constraint.
SCIP_RETCODE CreateSameDiffConstraint(
    SCIP* scip, SCIP_CONS** new_constraint_ptr, const char* constraint_name,
    const int first_point, const int second_point,
    const SameDiffConstraintType constraint_type, SCIP_NODE* node_where_created,
    SCIP_Bool local_constraint_only) {
  // Find the samediff constraint handler.
  SCIP_CONSHDLR* constraint_handler =
      SCIPfindConshdlr(scip, "SameDiffContHdlr");
  if (constraint_handler == nullptr) {
    SCIPerrorMessage("samediff constraint handler not found\n");
    return SCIP_PLUGINNOTFOUND;
  }
  /* create the constraint specific data */
  SCIP_CONSDATA* constraint_data;
  SCIP_CALL(CreateSameDiffConstraintData(scip, &constraint_data, first_point,
                                         second_point, constraint_type,
                                         node_where_created));
  /* create constraint */
  SCIP_CALL(SCIPcreateCons(scip, new_constraint_ptr, constraint_name,
                           constraint_handler, constraint_data,
                           /*initial=*/FALSE, /*separate=*/FALSE,
                           /*enforce=*/FALSE, /*check=*/FALSE,
                           /*propagate=*/TRUE, /*local=*/local_constraint_only,
                           /*modifiable=*/FALSE, /*dynamic=*/FALSE,
                           /*removable=*/FALSE, /*stickingatnode=*/TRUE));
  SCIPdebugMsg(scip, "created constraint: ");
  SCIPdebug(DisplaySameDiffConstraintData(scip, constraint_data, NULL));
  return SCIP_OKAY;
}

// Return details of the constraint.
SameDiffConstraintDetails GetConstraintDetails(SCIP_CONS* constraint) {
  SCIP_CONSDATA* constraint_data;
  assert(constraint != nullptr);
  constraint_data = SCIPconsGetData(constraint);
  assert(constraint_data != nullptr);
  return {constraint_data->first_point, constraint_data->second_point,
          constraint_data->constraint_type};
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
