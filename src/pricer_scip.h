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

#ifndef EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_SCIP_H_
#define EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_SCIP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef LOCAL
#include "objscip/objpricer.h"
#include "pricer/pricer.h"
#include "pricer/pricer_cpsat/pricer_cpsat.h"
#include "pricer/pricer_maxclosure/pricer_maxclosure.h"
#include "pricer/pricer_mip/pricer_mip.h"
#include "scip/def.h"
#include "scip/type_pricer.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#else
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_cpsat/pricer_cpsat.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_maxclosure/pricer_maxclosure.h"
#include "experimental/users/edomoreno/HyperRectangularClustering/src/pricer/pricer_mip/pricer_mip.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/scip/src/objscip/objpricer.h"
#include "third_party/scip/src/scip/def.h"
#include "third_party/scip/src/scip/type_pricer.h"
#include "third_party/scip/src/scip/type_retcode.h"
#include "third_party/scip/src/scip/type_scip.h"
#endif

namespace operations_research {
namespace hyperrectangular_clustering {

static const char* kPRICER_NAME = "MaxClustering_Pricer";

enum class PricerOptions {
  kPricerCpSat = 0,
  kPricerCpSatEnforcing,
  kPricerMaxClosure,
  kPricerMIP,
  kNone
};

bool AbslParseFlag(absl::string_view text, PricerOptions* pricer,
                   std::string* error);
std::string AbslUnparseFlag(PricerOptions pricer);

class ObjPricerMaxClustering : public scip::ObjPricer {
 public:
  ObjPricerMaxClustering(SCIP* scip, const char* pricer_name,
                         const std::vector<std::vector<int64_t>>& point_coords,
                         PricerOptions pricer_option)
      : scip::ObjPricer(scip, pricer_name,
                        "Find cluster with negative reduced cost.",
                        /*priority=*/0, /*delay=*/TRUE),
        point_coords_(point_coords),
        use_paired_pairs_branching_(false) {
    switch (pricer_option) {
      case PricerOptions::kPricerCpSat:
        max_hyper_rectangular_ = std::make_unique<MaxHyperRectangularCpSat>(
            point_coords, /*enforce_enclosed_points=*/false);
        break;
      case PricerOptions::kPricerCpSatEnforcing:
        max_hyper_rectangular_ = std::make_unique<MaxHyperRectangularCpSat>(
            point_coords, /*enforce_enclosed_points=*/true);
        break;
      case PricerOptions::kPricerMaxClosure:
        max_hyper_rectangular_ =
            std::make_unique<MaxHyperRectangularMaxClosure>(point_coords);
        break;
      case PricerOptions::kPricerMIP:
        max_hyper_rectangular_ =
            std::make_unique<MaxHyperRectangularMIP>(point_coords);
        break;
      default:
        max_hyper_rectangular_ = nullptr;
    }
  }

  ~ObjPricerMaxClustering() override = default;

  // Initialization of variable pricer (called after problem was transformed).
  virtual SCIP_DECL_PRICERINIT(scip_init);

  // Reduced cost pricing method of variable pricer for feasible LPs.
  virtual SCIP_DECL_PRICERREDCOST(scip_redcost);

  // Farkas pricing method of variable pricer for infeasible LPs.
  virtual SCIP_DECL_PRICERFARKAS(scip_farkas);

  // Activate adding paired pairs branching constraints.
  void ActivatePairedPairsBranching() {
    use_paired_pairs_branching_ = true;
  }

  // Perform pricing.
  SCIP_RETCODE pricing(SCIP* scip, bool isfarkas);

 private:
  const std::vector<std::vector<int64_t>>& point_coords_;
  std::unique_ptr<MaxHyperRectangular> max_hyper_rectangular_;
  bool use_paired_pairs_branching_;
};

}  // namespace hyperrectangular_clustering
}  // namespace operations_research

#endif  // EXPERIMENTAL_USERS_EDOMORENO_HYPERRECTANGULARCLUSTERING_SRC_PRICER_SCIP_H_
