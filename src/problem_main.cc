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
#include <fstream>
#include <string>
#include <vector>

#ifdef LOCAL
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "src/pricer_scip.h"
#include "src/problem_model.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer_scip.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/problem_model.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/flags/parse.h"
#include "third_party/absl/flags/usage.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/ascii.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#endif

using operations_research::hyperrectangular_clustering::BranchingOptions;
using operations_research::hyperrectangular_clustering::PricerOptions;

ABSL_FLAG(std::string, input_data_file, "", "Input file with problem data");
ABSL_FLAG(PricerOptions, pricer, PricerOptions::kPricerCpSat,
          "Pricer to use [cpsat|cpsat_enforcing,maxclosure,mip]");
ABSL_FLAG(BranchingOptions, branching, BranchingOptions::kRyanFoster,
          "Branching to use [none|ryanfoster]");
ABSL_FLAG(bool, relax_coverage_vars, false,
          "Relax integrality of coverage vars");
ABSL_FLAG(bool, relax_priced_vars, false, "Relax integrality of priced vars");
ABSL_FLAG(double, time_limit, 300.0, "time limit (in seconds)");

namespace {

struct ProblemData {
  std::vector<std::vector<int64_t>> coords;
  int num_points;
  int num_dimensions;
  int max_num_clusters;
  int max_num_outliers;
};

ProblemData read_input_data(const std::string& input_data_file) {
  ProblemData result;
  std::ifstream file(input_data_file);
  if (!file.is_open()) {
    LOG(FATAL) << "Error opening file: " << input_data_file;
    return result;
  }
  std::string line;
  int line_number = 1;
  while (std::getline(file, line)) {
    // Remove leading whitespace from the line
    line = absl::StripLeadingAsciiWhitespace(line);
    line = absl::StripTrailingAsciiWhitespace(line);
    std::vector<int64_t> row;
    for (const auto& token : absl::StrSplit(line, '\t', absl::SkipEmpty())) {
      int64_t value;
      if (absl::SimpleAtoi(token, &value)) {
        row.push_back(value);
      } else {
        LOG(FATAL) << "Error on line " << line_number
                   << " parsing value: " << token;
      }
    }
    if (line_number == 1) {
      result.num_points = static_cast<int>(row[0]);
      result.num_dimensions = static_cast<int>(row[1]);
      result.max_num_clusters = static_cast<int>(row[2]);
      result.max_num_outliers = static_cast<int>(row[3]);
      result.coords.reserve(result.num_points);
    } else {
      if (row.size() != result.num_dimensions)
        LOG(FATAL) << "Dimension incorrect on line " << line_number;
      result.coords.push_back(row);
    }
    line_number++;
  }
  if (result.coords.size() != result.num_points)
    LOG(FATAL) << "Mismatch number of points (" << result.num_points
               << ") and number of read lines (" << result.coords.size();

  return result;
}
}  // namespace

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "Hyperrectangular clustering."
      " --input_data_file=<file>. See also --help");
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_input_data_file).empty()) {
    LOG(INFO) << absl::ProgramUsageMessage();
    return 0;
  }

  ProblemData problem_data =
      read_input_data(absl::GetFlag(FLAGS_input_data_file));
  operations_research::hyperrectangular_clustering::
      ClusteringMaxHyperRectangular problem(
          problem_data.coords, problem_data.max_num_clusters,
          problem_data.max_num_outliers,
          absl::GetFlag(FLAGS_relax_coverage_vars),
          absl::GetFlag(FLAGS_relax_priced_vars));
  operations_research::hyperrectangular_clustering::SolveParameters parameters =
      {.pricer = absl::GetFlag(FLAGS_pricer),
       .branching = absl::GetFlag(FLAGS_branching),
       .limits_time = absl::GetFlag(FLAGS_time_limit)};
  absl::StatusOr<
      operations_research::hyperrectangular_clustering::ProblemSolution>
      result = problem.Solve(parameters);
  ;
  return 0;
}
