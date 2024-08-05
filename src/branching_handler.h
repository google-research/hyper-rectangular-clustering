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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_BRANCHING_HANDLER_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_BRANCHING_HANDLER_H_

#ifdef LOCAL
#include "objscip/objbranchrule.h"
#include "scip/scip.h"
#else
#include "third_party/scip/src/objscip/objbranchrule.h"
#include "third_party/scip/src/scip/type_branch.h"
#include "third_party/scip/src/scip/type_scip.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

class SameDiffBranchingHandler : public scip::ObjBranchrule {
 public:
  explicit SameDiffBranchingHandler(SCIP* scip)
      : ObjBranchrule(scip, "RyanFoster", "Ryan/Foster branching rule",
                      /*priority=*/50000, /*maxdepth=*/-1,
                      /*maxbounddist=*/1.0) {
    SetVarPriorityBranching(scip);
  }
  virtual SCIP_DECL_BRANCHEXECLP(scip_execlp);

 private:
  static void SetVarPriorityBranching(SCIP* scip);
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_BRANCHING_HANDLER_H_
