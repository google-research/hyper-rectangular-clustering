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
#include <utility>
#include <vector>

#ifdef LOCAL
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/pricer_scip.h"
#include "src/problem_model.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer_scip.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_model.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/statusor.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

namespace {

using ::testing::Contains;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Pointwise;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

class HyperRectangularClusteringTest
    : public ::testing::TestWithParam<PricerOptions> {};
INSTANTIATE_TEST_SUITE_P(PricerOptionTest, HyperRectangularClusteringTest,
                         testing::Values(PricerOptions::kPricerCpSat,
                                         PricerOptions::kPricerMaxClosure,
                                         PricerOptions::kPricerMIP));

class HyperRectangularClusteringTestWithBranching
    : public ::testing::TestWithParam<PricerOptions> {};
INSTANTIATE_TEST_SUITE_P(PricerOptionTest,
                         HyperRectangularClusteringTestWithBranching,
                         testing::Values(PricerOptions::kPricerCpSat,
                                         PricerOptions::kPricerMIP));

TEST(HyperRectangularClusteringTest, BasicProblemWithoutPricer) {
  std::vector<std::vector<int64_t>> points = {
      {1, 1}, {1, 2}, {2, 2}, {2, 1}, {4, 4}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/1,
                                       /*max_num_outliers=*/1,
                                       /*relax_coverage_vars=*/true,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = PricerOptions::kNone,
                                .branching = BranchingOptions::kNone};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(4.5));
  EXPECT_THAT(solution->point_coverage,
              UnorderedPointwise(DoubleEq(), std::vector<double>{
                                                 0.75, 0.75, 0.75, 0.75, 1.0}));
  EXPECT_THAT(
      solution->cluster_limits,
      Contains(std::pair<ClusterLimits, double>({{{1, 4}, {1, 4}}, 0.75})));
}

TEST_P(HyperRectangularClusteringTest, BasicProblemWithPricer) {
  std::vector<std::vector<int64_t>> points = {
      {1, 1}, {1, 2}, {2, 2}, {2, 1}, {4, 4}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/1,
                                       /*max_num_outliers=*/1,
                                       /*relax_coverage_vars=*/true,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = GetParam(),
                                .branching = BranchingOptions::kNone};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(2));
  EXPECT_THAT(solution->point_coverage, ElementsAre(1, 1, 1, 1, 0));
  EXPECT_THAT(solution->cluster_limits,
              UnorderedElementsAre(
                  std::pair<ClusterLimits, double>({{{1, 2}, {1, 2}}, 1.0})));
}

TEST(HyperRectangularClusteringTest,
     BasicProblemWithoutPricerWithBinaryCoverage) {
  std::vector<std::vector<int64_t>> points = {
      {1, 1}, {1, 2}, {2, 2}, {2, 1}, {4, 4}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/1,
                                       /*max_num_outliers=*/1,
                                       /*relax_coverage_vars=*/false,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = PricerOptions::kNone,
                                .branching = BranchingOptions::kNone};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(6.0));
  EXPECT_THAT(solution->point_coverage, Contains(DoubleEq(1.0)).Times(4));
  EXPECT_THAT(solution->cluster_limits.front().first,
              Eq(ClusterLimits{{{1, 4}, {1, 4}}}));
  EXPECT_THAT(solution->cluster_limits.front().second, DoubleEq(1.0));
}

TEST_P(HyperRectangularClusteringTest, FractionalSolutionWithPricer) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {1, 2}, {2, 2},  {2, 1},
                                              {4, 1}, {5, 2}, {10, 10}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/1,
                                       /*max_num_outliers=*/2,
                                       /*relax_coverage_vars=*/true,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = GetParam(),
                                .branching = BranchingOptions::kNone};
  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(3.5));
  EXPECT_THAT(
      solution->point_coverage,
      Pointwise(DoubleEq(), std::vector<double>{1, 1, 1, 1, 0.5, 0.5, 0}));
  EXPECT_THAT(solution->cluster_limits,
              UnorderedElementsAre(
                  std::pair<ClusterLimits, double>({{{1, 5}, {1, 2}}, 0.5}),
                  std::pair<ClusterLimits, double>({{{1, 2}, {1, 2}}, 0.5})));
}

TEST_P(HyperRectangularClusteringTest,
       IntegerSolutionUsingPricerWithBinaryCoverage) {
  std::vector<std::vector<int64_t>> points = {{1, 1}, {1, 2}, {2, 2},  {2, 1},
                                              {4, 1}, {5, 2}, {10, 10}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/1,
                                       /*max_num_outliers=*/2,
                                       /*relax_coverage_vars=*/false,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = GetParam(),
                                .branching = BranchingOptions::kNone};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(4.0));
  EXPECT_THAT(solution->point_coverage,
              Pointwise(DoubleEq(), std::vector<double>{1, 1, 1, 1, 1, 0, 0}));
  EXPECT_THAT(solution->cluster_limits,
              UnorderedElementsAre(
                  std::pair<ClusterLimits, double>({{{1, 4}, {1, 2}}, 1.0})));
}

TEST_P(HyperRectangularClusteringTest,
       FractionalClustersSolutionUsingPricerWithBinaryCoverage) {
  std::vector<std::vector<int64_t>> points = {{10, 10}, {10, 20}, {20, 20},
                                              {20, 10}, {40, 10}, {38, 20}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/2,
                                       /*max_num_outliers=*/0,
                                       /*relax_coverage_vars=*/false,
                                       /*relax_priced_vars=*/true);
  SolveParameters parameters = {.pricer = GetParam(),
                                .branching = BranchingOptions::kNone};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  // Valid solution is {(10,20),(10,20)},{(38,40),{10,20}} with half perimeter
  // 32. But since pricer variables are fractional, there exist a solution with
  // half perimeter 30.
  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(30));
  EXPECT_THAT(solution->point_coverage,
              Pointwise(DoubleEq(), std::vector<double>{1, 1, 1, 1, 1, 1}));
  EXPECT_THAT(
      solution->cluster_limits,
      UnorderedElementsAre(
          std::pair<ClusterLimits, double>({{{10, 20}, {10, 20}}, 0.5}),
          std::pair<ClusterLimits, double>({{{10, 40}, {10, 20}}, 0.5}),
          std::pair<ClusterLimits, double>({{{40, 40}, {10, 10}}, 0.5}),
          std::pair<ClusterLimits, double>({{{38, 38}, {20, 20}}, 0.5})));
}

TEST_P(HyperRectangularClusteringTestWithBranching,
       IntegerClustersSolutionUsingPricerWithBinaryCoverage) {
  std::vector<std::vector<int64_t>> points = {{10, 10}, {10, 20}, {20, 20},
                                              {20, 10}, {40, 10}, {38, 20}};
  ClusteringMaxHyperRectangular master(points, /*max_num_clusters=*/2,
                                       /*max_num_outliers=*/0,
                                       /*relax_coverage_vars=*/false,
                                       /*relax_priced_vars=*/false);
  SolveParameters parameters = {.pricer = GetParam(),
                                .branching = BranchingOptions::kRyanFoster};

  absl::StatusOr<ProblemSolution> solution = master.Solve(parameters);

  EXPECT_TRUE(solution.ok());
  EXPECT_THAT(solution->half_perimeter, DoubleEq(32));
  EXPECT_THAT(solution->point_coverage,
              Pointwise(DoubleEq(), std::vector<double>{1, 1, 1, 1, 1, 1}));
  EXPECT_THAT(solution->cluster_limits,
              UnorderedElementsAre(
                  std::pair<ClusterLimits, double>({{{10, 20}, {10, 20}}, 1}),
                  std::pair<ClusterLimits, double>({{{38, 40}, {10, 20}}, 1})));
}

}  // namespace
}  // namespace hyperrectangular_clustering
}  // namespace operations_research
