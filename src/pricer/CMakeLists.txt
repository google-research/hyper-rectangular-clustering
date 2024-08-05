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

add_library(pricer pricer.cc)
target_link_libraries(pricer absl::statusor absl::log)

add_subdirectory(pricer_cpsat)
add_subdirectory(pricer_maxclosure)
add_subdirectory(pricer_mip)


enable_testing()
add_executable(pricer_test pricer_test.cc)
target_link_libraries(pricer_test pricer_maxclosure pricer_cpsat pricer_mip pricer absl::statusor GTest::gtest_main )
include(GoogleTest)
gtest_discover_tests(pricer_test)
