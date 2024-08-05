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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_SAME_OR_DIFF_CONSTRAINT_HANDLER_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_SAME_OR_DIFF_CONSTRAINT_HANDLER_H_

#ifdef LOCAL
#include "objscip/objconshdlr.h"
#include "scip/scip.h"
#else
#include "third_party/scip/src/objscip/objconshdlr.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#include "third_party/scip/src/scip/type_timing.h"
#include "third_party/scip/src/scip/type_tree.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

// Type of constraint
enum class SameDiffConstraintType {
  kPointsOnSameCluster = 0,
  kPointsOnDiffCluster
};

struct SameDiffConstraintDetails {
  int first_point;
  int second_point;
  SameDiffConstraintType constraint_type;
};

class SameDiffConstraintHandler : public scip::ObjConshdlr {
 public:
  explicit SameDiffConstraintHandler(SCIP* scip)
      : ObjConshdlr(scip, "SameDiffContHdlr",
                    "Constraint Handler for Ryan Foster Branching",
                    /*sepapriority=?*/ 1000000, /*enfopriority=*/0,
                    /*checkpriority=*/9999999, /*sepafreq=?*/ 1,
                    /*propfreq=*/1,
                    /*eagerfreq=*/1, /*maxprerounds=?*/ 0,
                    /*delaysepa=?*/ false,
                    /*delayprop=*/false, /*needscons=*/true,
                    /*proptiming=*/SCIP_PROPTIMING_BEFORELP,
                    /*presoltiming=?*/ SCIP_PRESOLTIMING_FAST) {}
  // Free specific constraint data.
  virtual SCIP_DECL_CONSDELETE(scip_delete);
  // Transforms constraint data into data belonging to the transformed problem.
  virtual SCIP_DECL_CONSTRANS(scip_trans);
  // Domain propagation method. It fixes variables violating the constraint.
  virtual SCIP_DECL_CONSPROP(scip_prop);
  // Constraint activation method.
  virtual SCIP_DECL_CONSACTIVE(scip_active);
  // Constraint deactivation method.
  virtual SCIP_DECL_CONSDEACTIVE(scip_deactive);
  // Callback not implemented because not required.
  virtual SCIP_DECL_CONSENFOLP(scip_enfolp) { return SCIP_ERROR; }
  virtual SCIP_DECL_CONSENFOPS(scip_enfops) { return SCIP_ERROR; }
  virtual SCIP_DECL_CONSCHECK(scip_check) { return SCIP_ERROR; }
  virtual SCIP_DECL_CONSLOCK(scip_lock) { return SCIP_ERROR; }
};

// Creates and captures a samediff constraint.
SCIP_RETCODE CreateSameDiffConstraint(
    SCIP* scip, SCIP_CONS** new_constraint_ptr, const char* constraint_name,
    int first_point, int second_point, SameDiffConstraintType constraint_type,
    SCIP_NODE* node_where_created, SCIP_Bool local_constraint_only);

// Return details of the constraint.
SameDiffConstraintDetails GetConstraintDetails(SCIP_CONS* constraint);

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_SAME_OR_DIFF_CONSTRAINT_HANDLER_H_
