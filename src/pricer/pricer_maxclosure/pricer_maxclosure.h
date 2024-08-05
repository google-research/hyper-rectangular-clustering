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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MAXCLOSURE_PRICER_MAXCLOSURE_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MAXCLOSURE_PRICER_MAXCLOSURE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ortools/graph/max_flow.h"
#include "pricer/pricer.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/statusor.h"
#include "util/operations_research/graph/ebert_graph.h"
#include "util/operations_research/graph/max_flow.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

class MaxHyperRectangularMaxClosure : public MaxHyperRectangular {
 public:
  explicit MaxHyperRectangularMaxClosure(
      const std::vector<std::vector<int64_t>>& coords);

  // Find the max hyper rectangular.
  absl::StatusOr<MaxHyperRectangularSolution> SolveMaxHyperRectangular(
      const std::vector<double>& weights,
      const std::vector<int>& forbidden_points,
      const std::vector<std::pair<int, int>>& points_same_cluster,
      const std::vector<std::pair<int, int>>& points_diff_cluster,
      bool ignore_perimeter_in_objective) override;

 private:
  const std::vector<std::vector<int64_t>>& coords_;
  const std::vector<std::vector<size_t>> sorted_coords_indices_;
  const std::vector<std::vector<NodeIndex>> precedences_;
  const std::vector<int64_t> maxclosure_base_weights_;
  const int64_t objval_null_solution_;
  const std::unique_ptr<SimpleMaxFlow> max_flow_;
  const std::vector<std::pair<NodeIndex, NodeIndex>> flow_arcs_to_points_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_PRICER_MAXCLOSURE_PRICER_MAXCLOSURE_H_
