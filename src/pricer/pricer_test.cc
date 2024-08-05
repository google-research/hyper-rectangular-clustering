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

#include <cstdint>
#include <vector>

#ifdef LOCAL
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pricer_cpsat/pricer_cpsat.h"
#include "pricer_maxclosure/pricer_maxclosure.h"
#include "pricer_mip/pricer_mip.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_cpsat/pricer_cpsat.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_maxclosure/pricer_maxclosure.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_mip/pricer_mip.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {
namespace sat {
namespace {

using ::testing::AllOf;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Pair;
using ::testing::SizeIs;

constexpr double kObjvalSolverTolerance = 1e-10;

template <typename PricerType>
class GeneralPricerTest : public testing::Test {};
using GeneralPricerTypes =
    ::testing::Types<MaxHyperRectangularCpSat, MaxHyperRectangularMaxClosure,
                     MaxHyperRectangularMIP>;
TYPED_TEST_SUITE(GeneralPricerTest, GeneralPricerTypes);

// Test for pricer enforcing that pairs of points in same/diff cluster.
template <typename PricerType>
class PairPointsPricerTest : public testing::Test {};
using PairPointsPricerTypes =
    ::testing::Types<MaxHyperRectangularCpSat, MaxHyperRectangularMIP>;
TYPED_TEST_SUITE(PairPointsPricerTest, PairPointsPricerTypes);

// Test for pricer enforcing that points enclosed by rectangle must be selected.
template <typename PricerType>
class EnforcingEnclosedPricerTest : public testing::Test {};
using EnforcingEnclosedPricerTypes = ::testing::Types<MaxHyperRectangularCpSat>;
TYPED_TEST_SUITE(EnforcingEnclosedPricerTest, EnforcingEnclosedPricerTypes);

TYPED_TEST(GeneralPricerTest, ErrorOnWeightSize) {
  TypeParam max_hyper_rectangular({{1, 1}});

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {0.5, 0.5}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_FALSE(solution.ok());
  EXPECT_EQ(solution.status().code(), absl::StatusCode::kInvalidArgument);
}

TYPED_TEST(GeneralPricerTest, AllPointsSelected) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 5}, {4, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {100, 100, 100}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(1, 4), Pair(1, 5)),
                        ElementsAre(true, true, true),
                        DoubleNear(300 - 7, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, AllNegativeWeights) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 5}, {4, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {-1, -2, -3}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  // TODO(edomoreno) Check they have same value on first field.
  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution, FieldsAre(SizeIs(2), ElementsAre(false, false, false),
                                   DoubleNear(0, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, WeightNotLargeForExtendingSolution) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 5}, {4, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {100, 1, 100}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(1, 4), Pair(1, 3)),
                        ElementsAre(true, false, true),
                        DoubleNear(200 - 5, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, PositiveAndNegativeCoords) {
  std::vector<std::vector<int64_t>> points = {
      {-3, -3}, {-1, -1}, {1, 1}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {1, 100, 100, 1}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(-1, 1), Pair(-1, 1)),
                        ElementsAre(false, true, true, false),
                        DoubleNear(200 - 4, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, AllNegativeCoords) {
  std::vector<std::vector<int64_t>> points = {
      {-7, -7}, {-5, -5}, {-3, -3}, {-1, -1}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {1, 100, 100, 1}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(-5, -3), Pair(-5, -3)),
                        ElementsAre(false, true, true, false),
                        DoubleNear(200 - 4, kObjvalSolverTolerance)));
}

TYPED_TEST(EnforcingEnclosedPricerTest,
           PointInMiddleWithNegativeWeightMustBeAlsoIncluded) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 5}, {4, 3}, {3, 3}};
  TypeParam max_hyper_rectangular(points, /*enforce_enclosed_points=*/true);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {100, 100, 100, -1}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(1, 4), Pair(1, 5)),
                        ElementsAre(true, true, true, true),
                        DoubleNear(299 - 7, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, FractionValuesOnObjective) {
  std::vector<std::vector<int64_t>> points = {
      {1, 1}, {1, 2}, {2, 2}, {2, 1}, {4, 4}};
  std::vector<double> objective_coeficients(5, 0.7);
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          objective_coeficients, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->included_points,
              ElementsAre(true, true, true, true, false));
  EXPECT_THAT(solution->limits_solution, ElementsAre(Pair(1, 2), Pair(1, 2)));
  EXPECT_THAT(solution->objective_value,
              DoubleNear(2.8 - 2, kObjvalSolverTolerance));
  EXPECT_THAT(*solution,
              FieldsAre(ElementsAre(Pair(1, 2), Pair(1, 2)),
                        ElementsAre(true, true, true, true, false),
                        DoubleNear(2.8 - 2, kObjvalSolverTolerance)));
}

TYPED_TEST(GeneralPricerTest, ForbiddingPointNotEnforcing) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {50, 100, 40}, /*forbidden_points=*/{1},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 3), Pair(1, 3))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, false, true)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(90 - 4, kObjvalSolverTolerance))));
}

TYPED_TEST(EnforcingEnclosedPricerTest, ForbiddingPointEnforcing) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points, /*enforce_enclosed_points=*/true);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {50, 100, 40}, /*forbidden_points=*/{1},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 1), Pair(1, 1))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, false, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(50 - 0, kObjvalSolverTolerance))));
}

TYPED_TEST(GeneralPricerTest, IgnorePerimeter) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {4, 2}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {10, 10, 1}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective==*/true);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 4), Pair(1, 2))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, true, true)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(21, kObjvalSolverTolerance))));
}

TYPED_TEST(GeneralPricerTest, RepeatedForbiddingAreIndependent) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution_prev =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {2, 2}, /*forbidden_points=*/{0},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);
  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {2, 2}, /*forbidden_points=*/{1},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 1), Pair(1, 1))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(2, kObjvalSolverTolerance))));
}

TYPED_TEST(PairPointsPricerTest, PairInSameCluster) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {5, 0, 0}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 1}},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 2), Pair(1, 2))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, true, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(5 - 2, kObjvalSolverTolerance))));
}

TYPED_TEST(PairPointsPricerTest, PairInDiffCluster) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {5, 6, 0}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{{0, 1}},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(2, 2), Pair(2, 2))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(false, true, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(6 - 0, kObjvalSolverTolerance))));
}

TYPED_TEST(PairPointsPricerTest, ValidPairInSameAndDiffCluster) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {10, 10, 10}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 2}},
          /*points_diff_cluster=*/{{0, 1}},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 3), Pair(1, 3))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, false, true)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(20 - 4, kObjvalSolverTolerance))));
}

// If same pairs is in same and diff cluster, then we cannot include them.
TYPED_TEST(PairPointsPricerTest, SamePairInSameAndDiffCluster) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {10, 10, 10}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 2}},
          /*points_diff_cluster=*/{{0, 2}},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(2, 2), Pair(2, 2))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(false, true, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(10 - 0, kObjvalSolverTolerance))));
}

TYPED_TEST(PairPointsPricerTest, SequencePairsMakeInfeasible) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {10, 10, 10}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 1}, {1, 2}},
          /*points_diff_cluster=*/{{0, 2}},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(0, 0), Pair(0, 0))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(false, false, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(0, kObjvalSolverTolerance))));
}

TYPED_TEST(PairPointsPricerTest, RepeatedCallsResetConstraints) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  TypeParam max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> prev_solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {15, 10, 10}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 1}, {1, 2}},
          /*points_diff_cluster=*/{{0, 2}},
          /*ignore_perimeter_in_objective=*/false);
  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {15, 10, 10}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{},
          /*points_diff_cluster=*/{{0, 2}},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(*solution,
              AllOf(Field("limits_solution",
                          &MaxHyperRectangularSolution::limits_solution,
                          ElementsAre(Pair(1, 2), Pair(1, 2))),
                    Field("included_points",
                          &MaxHyperRectangularSolution::included_points,
                          ElementsAre(true, true, false)),
                    Field("objective_value",
                          &MaxHyperRectangularSolution::objective_value,
                          DoubleNear(25 - 2, kObjvalSolverTolerance))));
}

TEST(IndividualPricerTests, MaxClosureDoesNotSupportPairs) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {2, 2}, {3, 3}};
  MaxHyperRectangularMaxClosure max_hyper_rectangular(points);

  absl::StatusOr<MaxHyperRectangularSolution> solution =
      max_hyper_rectangular.SolveMaxHyperRectangular(
          {5, 0, 0}, /*forbidden_points=*/{},
          /*points_same_cluster=*/{{0, 1}},
          /*points_diff_cluster=*/{},
          /*ignore_perimeter_in_objective=*/false);

  EXPECT_FALSE(solution.ok());
  EXPECT_EQ(solution.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace sat
}  // namespace hyperrectangular_clustering
}  // namespace operations_research
