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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "ortools/graph/ebert_graph.h"
#include "ortools/graph/max_flow.h"
#include "pricer/pricer_maxclosure/pricer_maxclosure.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_maxclosure/pricer_maxclosure.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_format.h"
#include "util/operations_research/graph/ebert_graph.h"
#include "util/operations_research/graph/max_flow.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace {

template <typename T>
std::vector<size_t> SortIndicesDecreasing(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

std::vector<std::vector<size_t>> SortCoordsIndices(
    const std::vector<std::vector<int64_t>>& original_coords) {
  const size_t num_points = original_coords.size();
  // Copy coords of each point to sort them
  std::vector<std::vector<size_t>> sorted_indices(
      original_coords[0].size(), std::vector<size_t>(num_points));

  std::vector<int64_t> single_dim_coords(num_points);
  std::vector<size_t> single_dim_sorted_idx(num_points);
  for (size_t dim = 0; dim < original_coords[0].size(); ++dim) {
    // Copy one axis coords to sort them
    for (size_t idx = 0; idx < num_points; ++idx)
      single_dim_coords[idx] = original_coords[idx][dim];
    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::iota(single_dim_sorted_idx.begin(), single_dim_sorted_idx.end(), 0);
    std::stable_sort(single_dim_sorted_idx.begin(), single_dim_sorted_idx.end(),
                     [&single_dim_coords](size_t i1, size_t i2) {
                       return single_dim_coords[i1] < single_dim_coords[i2];
                     });
    std::copy(single_dim_sorted_idx.begin(), single_dim_sorted_idx.end(),
              sorted_indices[dim].begin());
  }
  return sorted_indices;
}

/* Construct the nodes and precedences of the internal graph
 * There are #points x (2*m_numDim+1) nodes, in the following order
 *   - #points representing point,  with 2xm_numDim precedences to each coords
 *   - For each dimension
 *      - #points LeftCoords, in order, with one precedence to the previous one
 *      - #points RightCoords, in order, with one precedence to the next one
 */
// TODO(edomoreno) Potential bug. If two points have the same coords.
// In this case will have weight 0 and the other will have the distance to
// the next one. Of this occurs, it could select one of them but not the other
// tricking the solution. It should use a single node if coords coincides.
std::vector<std::vector<NodeIndex>> ConstructPrecedences(
    const std::vector<std::vector<int64_t>>& coords,
    const std::vector<std::vector<size_t>>& sorted_indices) {
  const NodeIndex num_points = static_cast<NodeIndex>(coords.size());
  const size_t num_dimensions = coords[0].size();
  const size_t num_nodes = num_points * (2 * num_dimensions + 1);
  std::vector<std::vector<NodeIndex>> precedences(num_nodes);
  for (NodeIndex dim = 0; dim < coords[0].size(); ++dim) {
    const NodeIndex offset_dim = num_points * (2 * dim + 1);
    for (NodeIndex pos = 0; pos < num_points; ++pos) {
      const size_t point_idx = sorted_indices[dim][pos];
      // add precedence between y[point_idx] and left[dim][pos]
      precedences[point_idx].emplace_back(offset_dim + pos);
      // add precedence between y[point_idx] and right[dim][pos]
      precedences[point_idx].emplace_back(offset_dim + num_points + pos);
    }
    // Now precedences between left/right coordinates
    for (NodeIndex pos = 0; pos < num_points - 1; pos++)
      precedences[offset_dim + pos].emplace_back(offset_dim + pos + 1);
    for (NodeIndex pos = 1; pos < num_points; pos++)
      precedences[offset_dim + num_points + pos].emplace_back(
          offset_dim + num_points + pos - 1);
  }
  return precedences;
}

std::vector<int64_t> ComputeMaxClosureBaseWeights(
    const std::vector<std::vector<int64_t>>& coords,
    const std::vector<std::vector<size_t>>& sorted_indices,
    const std::vector<std::vector<NodeIndex>>& precedences) {
  const size_t num_points = coords.size();
  std::vector<int64_t> base_weights(precedences.size(), 0);
  for (size_t dim = 0; dim < coords[0].size(); ++dim) {
    const size_t pos_dim = num_points * (2 * dim + 1);
    // First for L (in reverse order)
    // Rightmost have its true value (same sign because maximizing)
    // Previous has the difference (negative vals, because worse)
    const size_t rightMost = pos_dim + num_points - 1;
    int64_t lastVal = coords[sorted_indices[dim][num_points - 1]][dim];
    base_weights[rightMost] = lastVal;
    for (size_t i = 1; i < num_points; ++i) {
      const size_t idx = num_points - 1 - i;
      const size_t sorted_index = sorted_indices[dim][idx];
      const int64_t val = coords[sorted_index][dim];
      base_weights[pos_dim + idx] = (val - lastVal);
      lastVal = val;
    }
    // then for R (in order)
    // Leftmost has its true value (minus, because maximizing)
    // Next one has the difference (negative vals, because worse)
    const size_t leftMost = pos_dim + num_points;
    lastVal = coords[sorted_indices[dim][0]][dim];
    base_weights[leftMost] = -lastVal;
    for (size_t idx = 1; idx < num_points; ++idx) {
      const size_t sorted_index = sorted_indices[dim][idx];
      const int64_t val = coords[sorted_index][dim];
      base_weights[pos_dim + num_points + idx] = -(val - lastVal);
      lastVal = val;
    }
  }
  return base_weights;
}

// If solution is empty for original points, max closure still will choose
// the only nodes with positive weight, which are the rightmost L[dim,x]
// (if the leftmost coordinate is positive) or the sum of both if negative.
int64_t GetValueOfNullSolution(const std::vector<int64_t>& base_weights,
                               const size_t num_points,
                               const size_t num_dimensions) {
  int64_t objval_null_solution = 0;
  for (size_t dim = 0; dim < num_dimensions; ++dim) {
    const size_t pos_dim = num_points * (2 * dim + 1);
    const size_t rightMost = pos_dim + num_points - 1;
    const size_t leftMost = pos_dim + num_points;
    objval_null_solution +=
        ((base_weights[rightMost] > 0) ? base_weights[rightMost] : 0) +
        ((base_weights[leftMost] > 0) ? base_weights[leftMost] : 0);
  }
  return objval_null_solution;
}

std::vector<std::pair<NodeIndex, NodeIndex>> ConstructFlowProblem(
    const std::vector<std::vector<NodeIndex>>& precedences,
    const std::unique_ptr<SimpleMaxFlow>& max_flow) {
  const NodeIndex num_nodes = static_cast<NodeIndex>(precedences.size());
  // Create arcs from precedences with capacity infinity.
  for (NodeIndex idx = 0; idx < num_nodes; ++idx) {
    for (NodeIndex prec_idx : precedences[idx]) {
      max_flow->AddArcWithCapacity(idx, prec_idx,
                                   std::numeric_limits<int64_t>::max());
    }
  }
  NodeIndex source_node = num_nodes;
  NodeIndex tail_node = source_node + 1;
  std::vector<std::pair<NodeIndex, NodeIndex>> flow_arcs_to_points;
  flow_arcs_to_points.reserve(num_nodes);
  // We create the node with 0 capacity. To be updated with weights.
  for (NodeIndex idx = 0; idx < num_nodes; ++idx) {
    ArcIndex source_arc = max_flow->AddArcWithCapacity(source_node, idx, 0);
    ArcIndex tail_arc = max_flow->AddArcWithCapacity(idx, tail_node, 0);
    flow_arcs_to_points.push_back({source_arc, tail_arc});
  }
  return flow_arcs_to_points;
}

// Current idea: max flow is at most the sum of all positive weights (including
// null solution objval). So we multiply by a factor on the limits of a int64_t.
FlowQuantity GetScalingFactor(const std::vector<double>& weights,
                              const std::vector<int64_t>& base_weights,
                              const int64_t objval_null_solution) {
  double sum_positive_weights = std::accumulate(
      weights.begin(), weights.end(), static_cast<double>(objval_null_solution),
      [](double a, double b) { return a + (b > 0.0 ? b : 0.0); });
  for (int64_t base_weight : base_weights)
    if (std::abs(static_cast<double>(base_weight)) > sum_positive_weights)
      sum_positive_weights = std::abs(static_cast<double>(base_weight));
  int exponent = std::ilogb(sum_positive_weights);
  // TODO(edomoreno) Fix this. It should be 1LL << (62-exponent) but failing.
  // Ensure the exponent doesn't exceed the int64_t limit
  //  exponent = std::min(exponent, 62); // 63 is reserved for the sign bit.
  //  int64_t factor = 1LL << (62- exponent);
  //  for (double weight : weights)
  //    CHECK(weight * factor >= 0);
  return 1 << (61 - exponent);
}

FlowQuantity SolveMaxFlow(
    const std::vector<FlowQuantity>& weights,
    const std::vector<std::pair<NodeIndex, NodeIndex>>& flow_arcs_to_points,
    const std::unique_ptr<SimpleMaxFlow>& max_flow) {
  const size_t num_nodes = weights.size();
  for (int idx = 0; idx < num_nodes; ++idx) {
    if (weights[idx] > 0)
      max_flow->SetArcCapacity(flow_arcs_to_points[idx].first, weights[idx]);
    else
      max_flow->SetArcCapacity(flow_arcs_to_points[idx].first, 0);
    if (weights[idx] < 0)
      max_flow->SetArcCapacity(flow_arcs_to_points[idx].second, -weights[idx]);
    else
      max_flow->SetArcCapacity(flow_arcs_to_points[idx].second, 0);
  }
  // Solve max closure
  NodeIndex source_node = static_cast<NodeIndex>(num_nodes);
  NodeIndex tail_node = source_node + 1;
  int status = max_flow->Solve(source_node, tail_node);
  CHECK_EQ(status, 0);
  FlowQuantity sum_positive_weights = std::accumulate(
      weights.begin(), weights.end(), 0,
      [](FlowQuantity a, FlowQuantity b) { return a + (b > 0 ? b : 0); });
  CHECK_GE(sum_positive_weights, 0);
  FlowQuantity optimal_flow = max_flow->OptimalFlow();
  CHECK_LE(optimal_flow, sum_positive_weights);
  return (sum_positive_weights - optimal_flow);
}

std::vector<bool> GetIncludedPointsFromMaxFlow(
    const size_t num_points, const std::unique_ptr<SimpleMaxFlow>& max_flow) {
  std::vector<bool> included_points(num_points, false);
  std::vector<NodeIndex> reachable_from_source;
  max_flow->GetSourceSideMinCut(&reachable_from_source);
  for (NodeIndex node : reachable_from_source) {
    if (node > num_points) continue;
    included_points[node] = true;
  }
  return included_points;
}

}  // namespace

MaxHyperRectangularMaxClosure::MaxHyperRectangularMaxClosure(
    const std::vector<std::vector<int64_t>>& coords)
    : MaxHyperRectangular(coords),
      coords_(coords),
      sorted_coords_indices_(SortCoordsIndices(coords_)),
      precedences_(ConstructPrecedences(coords_, sorted_coords_indices_)),
      maxclosure_base_weights_(ComputeMaxClosureBaseWeights(
          coords_, sorted_coords_indices_, precedences_)),
      objval_null_solution_(GetValueOfNullSolution(
          maxclosure_base_weights_, coords.size(), coords[0].size())),
      max_flow_(std::make_unique<SimpleMaxFlow>()),
      flow_arcs_to_points_(ConstructFlowProblem(precedences_, max_flow_)) {}

absl::StatusOr<MaxHyperRectangularSolution>
MaxHyperRectangularMaxClosure::SolveMaxHyperRectangular(
    const std::vector<double>& weights,
    const std::vector<int>& forbidden_points,
    const std::vector<std::pair<int, int>>& points_same_cluster,
    const std::vector<std::pair<int, int>>& points_diff_cluster,
    bool ignore_perimeter_in_objective) {
  if (!points_diff_cluster.empty() || !points_same_cluster.empty())
    return absl::InvalidArgumentError(absl::StrFormat(
        "Max Closure pricer does not support paired points constraints"));
  const size_t num_points = coords_.size();
  if (weights.size() != num_points)
    return absl::InvalidArgumentError(absl::StrFormat(
        "weights and coords must have the same size. (%d vs %d)",
        weights.size(), num_points));

  MaxHyperRectangularSolution solution;
  // Merge point weights with base weights for L/R nodes
  std::vector<FlowQuantity> full_weights(precedences_.size(), 0);
  // We need to scale the provided weights to use integer values.
  const FlowQuantity scaling_factor = GetScalingFactor(
      weights, maxclosure_base_weights_, objval_null_solution_);
  if (!ignore_perimeter_in_objective) {
    std::transform(
        maxclosure_base_weights_.begin(), maxclosure_base_weights_.end(),
        full_weights.begin(),
        [scaling_factor](FlowQuantity val) { return val * scaling_factor; });
  }
  std::transform(weights.begin(), weights.end(), full_weights.begin(),
                 [scaling_factor](double val) {
                   return static_cast<FlowQuantity>(
                       std::round(val * static_cast<double>(scaling_factor)));
                 });
  // To forbid a point, we assign a negative value.
  // Note: this assumes cover constraint in the main problem is <=.
  for (int forbidden_idx : forbidden_points) full_weights[forbidden_idx] = -1;

  FlowQuantity closure_value =
      SolveMaxFlow(full_weights, flow_arcs_to_points_, max_flow_);
  if (closure_value > (objval_null_solution_ * scaling_factor)) {
    VLOG(1) << "OPTIMAL. Closure value=" << closure_value
            << " critical=" << objval_null_solution_ * scaling_factor
            << " scaling=" << scaling_factor << "\n";
    VLOG(1) << "Unscaled: Closure value=" << closure_value / scaling_factor
            << " critical=" << objval_null_solution_ << "\n";
    std::vector<bool> included_points =
        GetIncludedPointsFromMaxFlow(num_points, max_flow_);
    return GenSolutionFromPointCoverage(weights, included_points,
                                        forbidden_points,
                                        ignore_perimeter_in_objective);
  } else {
    VLOG(1) << "Closure value " << closure_value << " below threshold "
            << objval_null_solution_ * scaling_factor
            << "scaling=" << scaling_factor
            << " re-optimizing forcing points\n";
    FlowQuantity best_value = 0;
    std::vector<bool> best_sol(coords_.size(), false);
    for (size_t idx : SortIndicesDecreasing(weights)) {
      // If weights are negative, then all remaining have negative weight.
      if (weights[idx] < 0) break;
      // If forbidden, skip.
      if (full_weights[idx] < 0) continue;
      VLOG(2) << "\tForcing" << idx << " with weight=" << weights[idx] << "\n";
      const FlowQuantity previous_value = full_weights[idx];
      full_weights[idx] += (objval_null_solution_ * scaling_factor);
      closure_value =
          SolveMaxFlow(full_weights, flow_arcs_to_points_, max_flow_);
      if (closure_value > best_value) {
        best_value = closure_value;
        best_sol = GetIncludedPointsFromMaxFlow(num_points, max_flow_);
      }
      full_weights[idx] = previous_value;
    }
    VLOG(1) << "\tOPTIMAL. Best closure value="
            << closure_value - objval_null_solution_ * scaling_factor
            << " critical=" << objval_null_solution_ * scaling_factor
            << " scaling=" << scaling_factor << "\n";
    const bool include_covered_with_zero_weight =
        forbidden_points.empty() && points_same_cluster.empty() &&
        points_diff_cluster.empty();
    return GenSolutionFromPointCoverage(weights, best_sol, forbidden_points,
                                        ignore_perimeter_in_objective,
                                        include_covered_with_zero_weight);
  }
}

}  // namespace hyperrectangular_clustering
}  // namespace operations_research
