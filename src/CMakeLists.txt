# Copyright 2010-2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#add_definitions(-DSCIP_DEBUG)

add_library(HyperRectangularClustering problem_model.cc pricer_scip.cc problem_data.cc
        same_or_diff_constraint_handler.cc branching_handler.cc)
target_link_libraries(HyperRectangularClustering
        absl::algorithm absl::log absl::status
        pricer pricer_cpsat pricer_maxclosure pricer_mip)

add_executable(problem_main problem_main.cc)
target_link_libraries(problem_main HyperRectangularClustering)

enable_testing()
add_executable(problem_model_test problem_model_test.cc)
target_link_libraries(problem_model_test HyperRectangularClustering GTest::gtest_main GTest::gmock_main)
include(GoogleTest)
gtest_discover_tests(problem_model_test)
