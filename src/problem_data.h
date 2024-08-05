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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_DATA_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_DATA_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "objscip/objscip.h"
#include "pricer/pricer.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "third_party/scip/src/objscip/objprobdata.h"
#include "third_party/scip/src/objscip/objvardata.h"
#include "third_party/scip/src/scip/type_cons.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#include "third_party/scip/src/scip/type_var.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

// Object to store variable data. In the case of pricer vars, it stores
// the limits, the coverage of vars and objective. If other variable (e.g.
// coverage vars), is nullopt.
class VarDataMaxHyperRectangular : public scip::ObjVardata {
 public:
  explicit VarDataMaxHyperRectangular(
      MaxHyperRectangularSolution cluster_solution)
      : cluster_data_(std::move(cluster_solution)) {}
  VarDataMaxHyperRectangular() : cluster_data_(std::nullopt) {}
  bool IsPricerVar() const { return cluster_data_.has_value(); }
  bool IsPointIncluded(int point_id) const {
    return cluster_data_->included_points[point_id];
  }
  ClusterLimits GetClusterLimits() const {
    return cluster_data_->limits_solution;
  }

 private:
  const std::optional<MaxHyperRectangularSolution> cluster_data_;
};

/** SCIP user problem data */
class ProbDataMaxHyperRectangular : public scip::ObjProbData {
 public:
  explicit ProbDataMaxHyperRectangular(
      const std::vector<std::vector<int64_t>>& point_coords,
      const bool relax_coverage_vars, const bool relax_priced_vars)
      : num_points_(point_coords.size()),
        point_coords_(point_coords),
        relax_coverage_vars_(relax_coverage_vars),
        relax_priced_vars_(relax_priced_vars) {}

  SCIP_CONS** GetMaxNumClustersConstraintPtr() {
    return &max_num_clusters_constraint_;
  }
  SCIP_VAR* GetCoverageVar(const int point_id) {
    return point_coverage_variables_[point_id];
  }
  SCIP_CONS** GetCoverageConstraintPtr(const int point_id) {
    return &coverage_of_points_constraint_[point_id];
  }
  const std::vector<SCIP_VAR*>& GetPricerVars() { return pricer_vars_; }
  int GetNumPricerVars() { return pricer_vars_.size(); }
  int GetNumPoints() { return num_points_; }

  SCIP_RETCODE FormulateProblem(SCIP* scip, int max_num_outliers,
                                int max_numclusters);
  SCIP_RETCODE FreeVarsAndConstraints(SCIP* scip);
  // Add a new cluster variable to problem.
  SCIP_RETCODE AddClusterVariables(SCIP* scip,
                                   const MaxHyperRectangularSolution& solution,
                                   bool is_initial = false);

 private:
  // Add initial set including single points and a cluster with all points.
  SCIP_RETCODE CreateInitialSetColumnsVars(SCIP* scip);
  SCIP_RETCODE CreateCoverageVariables(SCIP* scip);
  SCIP_RETCODE AddMaxNumOutliersConstraint(SCIP* scip, int max_num_outliers);
  SCIP_RETCODE AddMaxNumClustersConstraint(SCIP* scip, int max_num_clusters);
  SCIP_RETCODE AddCoverageConstraints(SCIP* scip);

  const size_t num_points_;
  const std::vector<std::vector<int64_t>>& point_coords_;
  std::unique_ptr<MaxHyperRectangular> max_hyper_rectangular_;
  std::vector<SCIP_VAR*> point_coverage_variables_;
  std::vector<SCIP_CONS*> coverage_of_points_constraint_;
  std::vector<SCIP_CONS*> coverage_of_points_constraint_orig_;
  SCIP_CONS* max_num_clusters_constraint_;
  SCIP_CONS* max_num_clusters_constraint_orig_;
  SCIP_CONS* max_num_outliers_constraint_;
  std::vector<SCIP_VAR*> pricer_vars_;
  const bool relax_coverage_vars_;
  const bool relax_priced_vars_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PROBLEM_DATA_H_
